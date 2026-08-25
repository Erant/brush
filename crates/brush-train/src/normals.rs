//! Monocular normal-map supervision.
//!
//! The per-splat pseudo-normal (shortest-scale axis, oriented to face the
//! camera, rotated into camera space) is computed by a fused differentiable
//! GPU operator, then handed to the renderer as its generic `[N, 3]` per-splat
//! feature input (`brush_render::bwd::render_splats_with_features`). The
//! rasterizer alpha-composites it into three extra output channels in the
//! *same* pass as the color render — projection, sorting and tile mapping
//! are shared, so the marginal cost is just the extra blend lanes.
//! Gradients flow back through the compositing into the feature tensor
//! (and from there through the fused analytic backward into the shared
//! `transforms` param), and into position/rotation/scale/opacity via the
//! blend weights. See `docs/normal-supervision.md` for the full writeup.

use brush_render::{camera::Camera, gaussian_splats::Splats};
use burn::tensor::Tensor;
#[cfg(test)]
use burn::tensor::{Device, s};

/// Broadcast a small constant vector to `[n, K]`. Built on the inner
/// (non-autodiff) device and lifted via `Tensor::from_inner` — mirroring how
/// GT data is lifted in [`normal_loss`] below — so it can combine with
/// splat-derived tensors that may be on the autodiff graph without tripping
/// burn-dispatch's cross-backend assert (a plain `Tensor::from_floats(..,
/// &device)` does not automatically pick up the autodiff-ness of `device`).
#[cfg(test)]
fn broadcast_row<const K: usize>(vals: [f32; K], n: usize, device: &Device) -> Tensor<2> {
    let inner: Tensor<1> = Tensor::from_floats(vals, &device.clone().inner());
    let lifted: Tensor<1> = Tensor::from_inner(inner);
    lifted.reshape([1, K]).expand([n as i32, K as i32])
}

/// Axis-sign correction between brush's internal camera space (OpenCV-style:
/// +X right, +Y down, +Z forward into the scene — see
/// `brush_dataset::formats::opengl_c2w_to_pose`, which performs the same
/// Y/Z flip in the opposite direction to convert *into* this convention) and
/// the Sapiens2 normal-map encoding, which empirical inspection of
/// `~/Documents/circ_colmap/normals/` showed follows the opposite
/// (OpenGL-style: +Y up, +Z toward the viewer) convention: the blue channel
/// (Z) reads high at the center of a camera-facing surface, and red (X)
/// increases left-to-right, matching a shared X axis with Y and Z flipped.
///
/// Applied to the *GT* at the loss boundary (the sign flips are involutive,
/// so the same constant maps either direction) — the rendered feature stays
/// a plain brush-convention camera-space normal.
///
/// Calibrated with the `normal_calib` example
/// (`crates/brush-bench-test/examples/normal_calib.rs`): masked mean
/// cosine over all 8 sign combinations against a model trained with this
/// loss for 15k steps — `[+X, -Y, -Z]` won decisively at 0.980 vs 0.762
/// for the runner-up. Rerun that example to recalibrate for a different
/// normal predictor; if it disagrees, flip signs here — nowhere else.
/// Per-splat world-space unit pseudo-normal: the splat's local axis with the
/// smallest scale (standard proxy for vanilla anisotropic 3D Gaussians —
/// splats naturally flatten against the surfaces they represent as training
/// progresses, so this axis converges toward the true surface normal),
/// oriented to face the camera.
#[cfg(test)]
fn world_space_normals(splats: &Splats, camera: &Camera) -> Tensor<2> {
    let quats = splats.rotations(); // [N, 4] (w, x, y, z), unnormalized
    let means = splats.means(); // [N, 3]
    // Which local axis (x/y/z) has the smallest scale, per splat, from the
    // *raw* log-scales: `exp` is monotone, and the min-scale fold (when
    // set) maps every axis of a splat through `s -> sqrt(s^2 + f^2)` with
    // one shared `f` — also monotone — so the argmin is identical to the
    // folded `scales()` while skipping the whole fold op-chain. `argmin`
    // returns an Int tensor, which is inherently outside the float autodiff
    // graph — this index selection is a free stop-gradient, same idea as
    // the `Tensor::sign()` use below.
    let log_scales = splats.log_scales(); // [N, 3]
    let n = log_scales.dims()[0];
    let device = log_scales.device();
    let axis_idx = log_scales.argmin(1).squeeze_dim::<1>(1); // [N] Int
    let local_axis = axis_idx.float().one_hot::<2>(3);

    // Specialise q*v*q^-1 for the one-hot shortest axis. This selects a
    // rotation-matrix column directly, avoiding the generic quaternion/vector
    // graph (and all of its vector slicing and zero multiplies).
    let qw = quats.clone().slice(s![.., 0..1]);
    let qx = quats.clone().slice(s![.., 1..2]);
    let qy = quats.clone().slice(s![.., 2..3]);
    let qz = quats.slice(s![.., 3..4]);
    let ax = local_axis.clone().slice(s![.., 0..1]);
    let ay = local_axis.clone().slice(s![.., 1..2]);
    let az = local_axis.slice(s![.., 2..3]);
    let two = 2.0;
    let r00 = qw.clone() * qw.clone() + qx.clone() * qx.clone()
        - qy.clone() * qy.clone() - qz.clone() * qz.clone();
    let r01 = (qx.clone() * qy.clone() - qw.clone() * qz.clone()) * two;
    let r02 = (qx.clone() * qz.clone() + qw.clone() * qy.clone()) * two;
    let r10 = (qx.clone() * qy.clone() + qw.clone() * qz.clone()) * two;
    let r11 = qw.clone() * qw.clone() - qx.clone() * qx.clone()
        + qy.clone() * qy.clone() - qz.clone() * qz.clone();
    let r12 = (qy.clone() * qz.clone() - qw.clone() * qx.clone()) * two;
    let r20 = (qx.clone() * qz.clone() - qw.clone() * qy.clone()) * two;
    let r21 = (qy.clone() * qz.clone() + qw.clone() * qx.clone()) * two;
    let r22 = qw.clone() * qw - qx.clone() * qx - qy.clone() * qy + qz.clone() * qz;
    let world_axis = Tensor::cat(
        vec![
            r00 * ax.clone() + r01 * ay.clone() + r02 * az.clone(),
            r10 * ax.clone() + r11 * ay.clone() + r12 * az.clone(),
            r20 * ax + r21 * ay + r22 * az,
        ],
        1,
    );
    let axis_len = world_axis
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .clamp_min(1e-12);
    let world_axis = world_axis.div(axis_len);

    // Orient to face the camera: flip any normal pointing away from it.
    // `Tensor::sign()` has a zero backward gradient (confirmed against
    // burn's vendored autodiff ops), so this is a proper stop-gradient, not
    // a hand-rolled detach.
    let cam_pos = broadcast_row(
        [camera.position.x, camera.position.y, camera.position.z],
        n,
        &device,
    );
    let to_cam = cam_pos.sub(means);
    let sign = to_cam.mul(world_axis.clone()).sum_dim(1).sign(); // [N, 1]
    world_axis.mul(sign)
}

/// Rotate a `[N, 3]` world-space normal tensor into camera space using the
/// camera's fixed (non-learned) rotation.
#[cfg(test)]
fn rotate_to_camera_space(world_normal: Tensor<2>, camera: &Camera) -> Tensor<2> {
    // `camera.rotation` is the local(camera)-to-world rotation
    // (`Camera::local_to_world`); we need the inverse to go world -> camera.
    let world_to_cam = camera.rotation.inverse();
    let m = glam::Mat3::from_quat(world_to_cam);
    let n = world_normal.dims()[0];
    let x = world_normal.clone().slice([0..n, 0..1]);
    let y = world_normal.clone().slice([0..n, 1..2]);
    let z = world_normal.slice([0..n, 2..3]);
    Tensor::cat(
        vec![
            x.clone() * m.x_axis.x + y.clone() * m.y_axis.x + z.clone() * m.z_axis.x,
            x.clone() * m.x_axis.y + y.clone() * m.y_axis.y + z.clone() * m.z_axis.y,
            x * m.x_axis.z + y * m.y_axis.z + z * m.z_axis.z,
        ],
        1,
    )
}

/// Per-splat camera-space unit pseudo-normal in brush's camera convention,
/// `[N, 3]`, on the autodiff graph of the splats' params. This is the
/// feature tensor handed to `render_splats_with_features`.
pub fn splat_camera_normals(splats: &Splats, camera: &Camera) -> Tensor<2> {
    brush_render::bwd::splat_camera_normals(splats.transforms.val(), camera)
}

#[cfg(test)]
fn splat_camera_normals_reference(splats: &Splats, camera: &Camera) -> Tensor<2> {
    rotate_to_camera_space(world_space_normals(splats, camera), camera)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::gaussian_splats::SplatRenderMode;

    fn splats(device: &Device) -> Splats {
        Splats::from_raw(
            vec![0.2, -0.3, 1.0, -0.6, 0.4, 2.0, 0.1, 0.8, 1.4],
            vec![
                0.91, 0.12, -0.21, 0.31,
                0.72, -0.33, 0.18, 0.44,
                1.10, 0.08, 0.27, -0.19,
            ],
            vec![-2.0, -0.3, 0.2, 0.1, -1.7, 0.5, 0.3, 0.7, -1.4],
            vec![0.0; 9],
            vec![0.0; 3],
            SplatRenderMode::Default,
            device,
        )
    }

    async fn values(t: Tensor<2>) -> Vec<f32> {
        t.into_data_async().await.unwrap().to_vec().unwrap()
    }

    #[tokio::test]
    async fn fused_normals_match_tensor_reference_forward_and_backward() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let camera = Camera {
            position: glam::vec3(1.2, -0.7, -2.3),
            rotation: glam::Quat::from_euler(glam::EulerRot::XYZ, 0.31, -0.27, 0.18),
            ..Default::default()
        };
        let fused_splats = splats(&device);
        let reference_splats = splats(&device);
        let fused = splat_camera_normals(&fused_splats, &camera);
        let reference = splat_camera_normals_reference(&reference_splats, &camera);
        let fv = values(fused.clone()).await;
        let rv = values(reference.clone()).await;
        for (i, (a, b)) in fv.iter().zip(&rv).enumerate() {
            assert!((a - b).abs() < 2e-5, "forward lane {i}: fused={a}, reference={b}");
        }

        let weights = Tensor::<1>::from_floats(
            [0.7, -0.2, 0.4, -0.5, 0.9, 0.3, 0.2, 0.6, -0.8],
            &device,
        ).reshape([3, 3]);
        let fg = (fused * weights.clone()).sum().backward();
        let rg = (reference * weights).sum().backward();
        let fgrad = values(fused_splats.transforms.grad(&fg).unwrap()).await;
        let rgrad = values(reference_splats.transforms.grad(&rg).unwrap()).await;
        for (i, (a, b)) in fgrad.iter().zip(&rgrad).enumerate() {
            assert!((a - b).abs() < 3e-4, "gradient lane {i}: fused={a}, reference={b}");
        }
    }
}
