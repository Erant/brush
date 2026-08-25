//! Fused differentiable construction of per-splat camera-space pseudo-normals.

use brush_cube::{MainBackend, MainBackendBase};
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::FloatTensor,
        wgpu::WgpuRuntime,
    },
    tensor::{DType, Shape, Tensor},
};
use burn_cubecl::{CubeRuntime, fusion::FusionCubeRuntime, kernel::into_contiguous, tensor::CubeTensor};
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use glam::{Mat3, Vec3};

use crate::burn_glue::{unwrap_ad_wgpu_float, wrap_ad_wgpu_float};
use crate::camera::Camera;

mod kernels {
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::prelude::*;

    pub const WORKGROUP: u32 = 256;

    #[cube]
    fn shortest_axis<F: Float>(transforms: &Tensor<F>, base: usize) -> u32 {
        let sx = transforms[base + 7usize];
        let sy = transforms[base + 8usize];
        let sz = transforms[base + 9usize];
        if sx <= sy && sx <= sz { 0u32 } else if sy <= sz { 1u32 } else { 2u32 }
    }

    #[cube]
    fn axis_from_quat<F: Float>(w: F, x: F, y: F, z: F, axis: u32) -> (F, F, F) {
        let two = F::cast_from(2.0_f32);
        if axis == 0u32 {
            (w*w + x*x - y*y - z*z, two*(x*y + w*z), two*(x*z - w*y))
        } else if axis == 1u32 {
            (two*(x*y - w*z), w*w - x*x + y*y - z*z, two*(y*z + w*x))
        } else {
            (two*(x*z + w*y), two*(y*z - w*x), w*w - x*x - y*y + z*z)
        }
    }

    #[cube(launch)]
    pub fn normal_forward<F: Float>(
        transforms: &Tensor<F>, out: &mut Tensor<F>, n: u32,
        cam_x: f32, cam_y: f32, cam_z: f32,
        r00: f32, r01: f32, r02: f32,
        r10: f32, r11: f32, r12: f32,
        r20: f32, r21: f32, r22: f32,
    ) {
        let i = CUBE_POS_X * WORKGROUP + UNIT_POS_X;
        if i >= n { terminate!(); }
        let base = (i * 10u32) as usize;
        let w = transforms[base + 3usize]; let x = transforms[base + 4usize];
        let y = transforms[base + 5usize]; let z = transforms[base + 6usize];
        let axis = shortest_axis::<F>(transforms, base);
        let (ax, ay, az) = axis_from_quat::<F>(w, x, y, z, axis);
        let len = F::max(F::sqrt(ax*ax + ay*ay + az*az), F::cast_from(1.0e-12_f32));
        let ux = ax / len; let uy = ay / len; let uz = az / len;
        let facing_dot = (F::cast_from(cam_x) - transforms[base]) * ux
            + (F::cast_from(cam_y) - transforms[base + 1usize]) * uy
            + (F::cast_from(cam_z) - transforms[base + 2usize]) * uz;
        let zero = F::cast_from(0.0_f32);
        let one = F::cast_from(1.0_f32);
        let face = select(facing_dot < zero, -one, select(facing_dot > zero, one, zero));
        let ux = ux * face; let uy = uy * face; let uz = uz * face;
        let ob = (i * 3u32) as usize;
        out[ob] = F::cast_from(r00)*ux + F::cast_from(r01)*uy + F::cast_from(r02)*uz;
        out[ob+1usize] = F::cast_from(r10)*ux + F::cast_from(r11)*uy + F::cast_from(r12)*uz;
        out[ob+2usize] = F::cast_from(r20)*ux + F::cast_from(r21)*uy + F::cast_from(r22)*uz;
    }

    #[cube(launch)]
    pub fn normal_backward<F: Float>(
        transforms: &Tensor<F>, grad_out: &Tensor<F>, grad_transforms: &mut Tensor<F>, n: u32,
        cam_x: f32, cam_y: f32, cam_z: f32,
        r00: f32, r01: f32, r02: f32,
        r10: f32, r11: f32, r12: f32,
        r20: f32, r21: f32, r22: f32,
    ) {
        let i = CUBE_POS_X * WORKGROUP + UNIT_POS_X;
        if i >= n { terminate!(); }
        let base = (i * 10u32) as usize;
        let w = transforms[base + 3usize]; let x = transforms[base + 4usize];
        let y = transforms[base + 5usize]; let z = transforms[base + 6usize];
        let axis = shortest_axis::<F>(transforms, base);
        let (ax, ay, az) = axis_from_quat::<F>(w, x, y, z, axis);
        let raw_len = F::sqrt(ax*ax + ay*ay + az*az);
        let eps = F::cast_from(1.0e-12_f32);
        let len = F::max(raw_len, eps);
        let ux = ax/len; let uy = ay/len; let uz = az/len;
        let facing_dot = (F::cast_from(cam_x)-transforms[base])*ux
            + (F::cast_from(cam_y)-transforms[base+1usize])*uy
            + (F::cast_from(cam_z)-transforms[base+2usize])*uz;
        let zero = F::cast_from(0.0_f32); let one = F::cast_from(1.0_f32);
        let face = select(facing_dot < zero, -one, select(facing_dot > zero, one, zero));
        let ob = (i * 3u32) as usize;
        let gox = grad_out[ob]; let goy = grad_out[ob+1usize]; let goz = grad_out[ob+2usize];
        // VJP through camera rotation and the detached facing sign.
        let gux = face*(F::cast_from(r00)*gox + F::cast_from(r10)*goy + F::cast_from(r20)*goz);
        let guy = face*(F::cast_from(r01)*gox + F::cast_from(r11)*goy + F::cast_from(r21)*goz);
        let guz = face*(F::cast_from(r02)*gox + F::cast_from(r12)*goy + F::cast_from(r22)*goz);
        let radial = gux*ux + guy*uy + guz*uz;
        let inv = one/len;
        let gax = select(raw_len > eps, (gux-ux*radial)*inv, gux*inv);
        let gay = select(raw_len > eps, (guy-uy*radial)*inv, guy*inv);
        let gaz = select(raw_len > eps, (guz-uz*radial)*inv, guz*inv);
        let two = F::cast_from(2.0_f32);
        let (gw, gx, gy, gz) = if axis == 0u32 {
            (two*(w*gax + z*gay - y*gaz), two*(x*gax + y*gay + z*gaz),
             two*(-y*gax + x*gay - w*gaz), two*(-z*gax + w*gay + x*gaz))
        } else if axis == 1u32 {
            (two*(-z*gax + w*gay + x*gaz), two*(y*gax - x*gay + w*gaz),
             two*(x*gax + y*gay + z*gaz), two*(-w*gax - z*gay + y*gaz))
        } else {
            (two*(y*gax - x*gay + w*gaz), two*(z*gax - w*gay - x*gaz),
             two*(w*gax + z*gay - y*gaz), two*(x*gax + y*gay + z*gaz))
        };
        grad_transforms[base+3usize] = gw; grad_transforms[base+4usize] = gx;
        grad_transforms[base+5usize] = gy; grad_transforms[base+6usize] = gz;
    }
}

#[derive(Clone, Copy, Debug)]
struct CameraArgs { pos: Vec3, rot: Mat3 }

trait NormalOps<B: Backend> {
    fn forward(transforms: FloatTensor<B>, camera: CameraArgs) -> FloatTensor<B>;
    fn backward(transforms: FloatTensor<B>, grad: FloatTensor<B>, camera: CameraArgs) -> FloatTensor<B>;
}

fn alloc<R: CubeRuntime>(t: &CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    burn_cubecl::ops::numeric::zeros_client::<R>(t.client.clone(), t.device.clone(), shape, t.dtype)
}

fn launch_forward<R: CubeRuntime>(transforms: CubeTensor<R>, camera: CameraArgs) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
    let transforms = into_contiguous(transforms);
    let n = transforms.shape()[0];
    let out = alloc(&transforms, Shape::new([n, 3]));
    let client = transforms.client.clone(); let m = camera.rot;
    kernels::normal_forward::launch::<f32, R>(&client, CubeCount::Static((n as u32).div_ceil(kernels::WORKGROUP), 1, 1), CubeDim::new_1d(kernels::WORKGROUP),
        transforms.into_tensor_arg(), out.clone().into_tensor_arg(), n as u32, camera.pos.x, camera.pos.y, camera.pos.z,
        m.x_axis.x,m.y_axis.x,m.z_axis.x,m.x_axis.y,m.y_axis.y,m.z_axis.y,m.x_axis.z,m.y_axis.z,m.z_axis.z);
    out
}

fn launch_backward<R: CubeRuntime>(transforms: CubeTensor<R>, grad: CubeTensor<R>, camera: CameraArgs) -> CubeTensor<R> {
    use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
    let transforms = into_contiguous(transforms); let grad = into_contiguous(grad);
    let n = transforms.shape()[0];
    let out = alloc(&transforms, Shape::new([n, 10]));
    let client = transforms.client.clone(); let m = camera.rot;
    kernels::normal_backward::launch::<f32, R>(&client, CubeCount::Static((n as u32).div_ceil(kernels::WORKGROUP),1,1), CubeDim::new_1d(kernels::WORKGROUP),
        transforms.into_tensor_arg(), grad.into_tensor_arg(), out.clone().into_tensor_arg(), n as u32, camera.pos.x,camera.pos.y,camera.pos.z,
        m.x_axis.x,m.y_axis.x,m.z_axis.x,m.x_axis.y,m.y_axis.y,m.z_axis.y,m.x_axis.z,m.y_axis.z,m.z_axis.z);
    out
}

impl NormalOps<Self> for MainBackendBase {
    fn forward(t: FloatTensor<Self>, c: CameraArgs) -> FloatTensor<Self> { launch_forward(t,c) }
    fn backward(t: FloatTensor<Self>, g: FloatTensor<Self>, c: CameraArgs) -> FloatTensor<Self> { launch_backward(t,g,c) }
}

#[derive(Debug)] struct CustomOp { desc: CustomOpIr, camera: CameraArgs, backward: bool }
impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
    fn execute(&self, h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>) {
        if self.backward {
            let ([t,g],[out]) = self.desc.as_fixed();
            let v = MainBackendBase::backward(h.get_float_tensor::<MainBackendBase>(t),h.get_float_tensor::<MainBackendBase>(g),self.camera);
            h.register_float_tensor::<MainBackendBase>(&out.id,v);
        } else {
            let ([t],[out]) = self.desc.as_fixed();
            let v = MainBackendBase::forward(h.get_float_tensor::<MainBackendBase>(t),self.camera);
            h.register_float_tensor::<MainBackendBase>(&out.id,v);
        }
    }
}

impl NormalOps<Self> for Fusion<MainBackendBase> {
    fn forward(t: FloatTensor<Self>, camera: CameraArgs) -> FloatTensor<Self> {
        let n=t.shape().dims::<2>()[0]; let client=t.client.clone();
        let out=TensorIr::uninit(client.create_empty_handle(),Shape::new([n,3]),DType::F32);
        let desc=CustomOpIr::new("splat_normal_forward", &[t.into_ir()], &[out]);
        let [out]=client.register(StreamId::current(),OperationIr::Custom(desc.clone()),CustomOp{desc,camera,backward:false}).outputs(); out
    }
    fn backward(t: FloatTensor<Self>, g: FloatTensor<Self>, camera: CameraArgs) -> FloatTensor<Self> {
        let shape=t.shape(); let client=t.client.clone();
        let out=TensorIr::uninit(client.create_empty_handle(),shape,DType::F32);
        let desc=CustomOpIr::new("splat_normal_backward", &[t.into_ir(),g.into_ir()], &[out]);
        let [out]=client.register(StreamId::current(),OperationIr::Custom(desc.clone()),CustomOp{desc,camera,backward:true}).outputs(); out
    }
}

#[derive(Debug)] struct NormalBackward;
#[derive(Debug,Clone)] struct State<B:Backend>{ transforms:FloatTensor<B>, camera:CameraArgs }
impl<B:Backend+NormalOps<B>> Backward<B,1> for NormalBackward {
    type State=State<B>;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer){
        let g=grads.consume::<B>(&ops.node); let [parent]=ops.parents;
        if let Some(p)=parent { grads.register::<B>(p.id,B::backward(ops.state.transforms,g,ops.state.camera)); }
    }
}

/// Construct differentiable `[N,3]` camera-space pseudo-normals in one GPU launch.
pub fn splat_camera_normals(transforms: Tensor<2>, camera: &Camera) -> Tensor<2> {
    let camera=CameraArgs{pos:camera.position,rot:Mat3::from_quat(camera.rotation.inverse())};
    let ad=unwrap_ad_wgpu_float(transforms);
    let prep=NormalBackward.prepare::<NoCheckpointing>([ad.node.clone()]).compute_bound().stateful();
    let inner=ad.primitive; let out=<MainBackend as NormalOps<MainBackend>>::forward(inner.clone(),camera);
    let out=match prep { OpsKind::Tracked(p)=>p.finish(State{transforms:inner,camera},out), OpsKind::UnTracked(p)=>p.finish(out) };
    wrap_ad_wgpu_float(out)
}
