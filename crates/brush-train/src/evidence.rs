//! Per-splat multi-view *evidence*, and the view-dependent *confidence*
//! derived from it.
//!
//! The question this answers is "how much did the training views actually
//! constrain this Gaussian, and from where?" — decided once, in 3-D, per
//! splat, instead of per pixel per frame from rendered alpha. See
//! `docs/splat-confidence.md` for the full writeup and the downstream
//! contract.
//!
//! ## Gathering evidence without a new kernel
//!
//! The rasterizer's optional per-splat feature channel composites a
//! `[N, 3]` input with exactly the colour blend weights `vis = α·T`, and its
//! backward is *ungated*: `∂/∂feat_i[k] Σ_p F_p·V_p = Σ_p vis_ip·V_p[k]`
//! for any constant per-pixel weight map `V`. So rendering a *zero* feature
//! and backpropagating `Σ_p F_p·V_p` turns the existing autodiff backward
//! into a per-splat accumulator of whatever `V` encodes. With
//! `V_p = (m_p, m_p·res_p, 1)` — `m` the GT alpha (mask or matte), `res` the
//! photometric residual — each splat collects, per view:
//!
//! | lane | meaning |
//! |---|---|
//! | `w_in  = Σ vis·m`     | contribution mass landing inside GT foreground |
//! | `e     = Σ vis·m·res` | mass-weighted residual where the splat is used |
//! | `w_all = Σ vis`       | total contribution mass |
//!
//! Summed over views, plus a view count and the resultant of observation
//! directions, that is the whole [`SplatEvidence`].

use anyhow::Context;
use brush_dataset::scene::{Scene, SceneBatch};
use brush_dataset::scene_loader::load_view_batch;
use brush_loss::{ImageLossConfig, image_loss_eval, normal_loss_eval};
use brush_render::burn_glue::detach_autodiff;
use brush_render::bwd::burn_glue::lift_splats_to_autodiff;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
pub use brush_serde::{EVIDENCE_FIELDS, EVIDENCE_STRIDE};
use burn::tensor::{Device, Int, Tensor, TensorData, Transaction, s};
use clap::Args;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// In-mask contribution mass (in pixel-weight units at training resolution)
/// a view must give a splat for it to count as a *supporting view*. One
/// pixel's worth: below that the splat is a rounding error in this view.
pub const VIEW_MIN_MASS: f32 = 1.0;

/// Multi-view evidence for every splat of a model, indices aligned with the
/// splat tensors. CPU-side; `EVIDENCE_STRIDE` floats per splat when flattened
/// (see [`EVIDENCE_FIELDS`] for the order and the ply property names).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SplatEvidence {
    /// Σ over views of in-mask contribution mass.
    pub w_in: Vec<f32>,
    /// Σ over views of total contribution mass.
    pub w_all: Vec<f32>,
    /// Σ over views of mass-weighted residual (photometric, plus the
    /// optionally weighted normal residual).
    pub err: Vec<f32>,
    /// Number of views giving this splat at least [`VIEW_MIN_MASS`] in-mask mass.
    pub views: Vec<f32>,
    /// `Σ_views w_in · normalize(camera − mean)`: unnormalised resultant of
    /// the directions this splat was observed from.
    pub dir: Vec<[f32; 3]>,
}

impl SplatEvidence {
    pub fn len(&self) -> usize {
        self.w_in.len()
    }

    pub fn is_empty(&self) -> bool {
        self.w_in.is_empty()
    }

    /// Fraction of this splat's drawn mass that landed inside GT foreground.
    pub fn inmask(&self, i: usize) -> f32 {
        if self.w_all[i] > 0.0 {
            (self.w_in[i] / self.w_all[i]).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Flatten to `EVIDENCE_STRIDE` floats per splat, [`EVIDENCE_FIELDS`] order.
    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len() * EVIDENCE_STRIDE);
        for i in 0..self.len() {
            out.extend([
                self.w_in[i],
                self.w_all[i],
                self.err[i],
                self.views[i],
                self.dir[i][0],
                self.dir[i][1],
                self.dir[i][2],
            ]);
        }
        out
    }

    /// Inverse of [`Self::to_flat`].
    pub fn from_flat(flat: &[f32]) -> Self {
        assert!(
            flat.len().is_multiple_of(EVIDENCE_STRIDE),
            "evidence block length {} is not a multiple of {EVIDENCE_STRIDE}",
            flat.len()
        );
        let rows = flat.chunks_exact(EVIDENCE_STRIDE);
        Self {
            w_in: rows.clone().map(|r| r[0]).collect(),
            w_all: rows.clone().map(|r| r[1]).collect(),
            err: rows.clone().map(|r| r[2]).collect(),
            views: rows.clone().map(|r| r[3]).collect(),
            dir: rows.map(|r| [r[4], r[5], r[6]]).collect(),
        }
    }

    /// Keep the rows where `keep[i]` is set.
    pub fn select(&self, keep: &[bool]) -> Self {
        assert_eq!(
            keep.len(),
            self.len(),
            "keep mask must match the evidence length"
        );
        let pick = |v: &[f32]| -> Vec<f32> {
            v.iter()
                .zip(keep)
                .filter_map(|(x, k)| k.then_some(*x))
                .collect()
        };
        Self {
            w_in: pick(&self.w_in),
            w_all: pick(&self.w_all),
            err: pick(&self.err),
            views: pick(&self.views),
            dir: self
                .dir
                .iter()
                .zip(keep)
                .filter_map(|(x, k)| k.then_some(*x))
                .collect(),
        }
    }
}

/// Per-view evidence for every splat: a `[N, 3]` tensor of
/// `(w_in, e, w_all)` on the inner device, measured against one training
/// view. `splats` must live on the inner (non-autodiff) device.
///
/// `normal_weight > 0` with a batch that carries a normal map adds
/// `normal_weight · (L1 + 1 − cos)` of the rendered pseudo-normal against the
/// GT normal into the residual, at the cost of one extra forward render.
pub async fn view_evidence(
    splats: &Splats,
    batch: &SceneBatch,
    device: &Device,
    normal_weight: f32,
) -> Tensor<2> {
    let [h, w] = [batch.img_packed.shape[0], batch.img_packed.shape[1]];
    let img_size = glam::uvec2(w as u32, h as u32);
    let n = splats.num_splats() as usize;
    let gt_packed: Tensor<2, Int> = Tensor::from_data(batch.img_packed.clone(), device);

    // Zero feature, tracked: its gradient is the accumulator.
    let lifted = lift_splats_to_autodiff(splats.clone());
    let feats = Tensor::<2>::zeros([n, 3], &lifted.device()).require_grad();
    let out = brush_render::bwd::render_splats_with_features(
        lifted,
        &batch.camera,
        img_size,
        Vec3::ZERO,
        feats.clone(),
    )
    .await;

    // Per-pixel weight map, on the inner device. A 4-channel `pred` with a
    // zeroed alpha lane makes the loss map's alpha lane `|0 - gt.a| = gt.a`,
    // i.e. the mask, and lanes 0..3 the per-channel L1 residual — one fused
    // kernel for both. Rendered on black: in transparent mode the GT is
    // premultiplied at load, in masked mode the residual outside the mask is
    // zeroed by `m` anyway, so no bg compositing is needed in either.
    // `.inner()` strips the autodiff wrapper; `detach_autodiff` then clears
    // the residual checkpointing flag so the loss kernels see a plain tensor.
    let img = detach_autodiff(out.img.inner());
    let pred4 = Tensor::cat(
        vec![
            img.slice(s![.., .., 0..3]),
            Tensor::zeros([h, w, 1], device),
        ],
        2,
    );
    let map = image_loss_eval(
        pred4,
        gt_packed.clone(),
        ImageLossConfig {
            l1_weight: 1.0,
            ssim_weight: 0.0,
            composite_bg: None,
            mask: false,
        },
    );
    let m = map.clone().slice(s![.., .., 3..4]);
    // A `weights/` sidecar scales this view's say the same way it scales
    // its loss: `m` is the per-pixel "this view constrains here" weight,
    // so w_in and the residual both shrink where the trainer was told to
    // listen less. Without this a view whose loss was silenced over some
    // region would still count as full evidence — and full disagreement —
    // for whatever the other views put there.
    let m = match &batch.loss_weight {
        Some(data) => {
            let w: Tensor<2> = Tensor::from_data(data.clone(), device);
            m * w.unsqueeze_dim(2)
        }
        None => m,
    };
    let mut res = map.slice(s![.., .., 0..3]).mean_dim(2);

    if normal_weight > 0.0
        && let Some(normal_data) = &batch.normal_data
    {
        let lifted = lift_splats_to_autodiff(splats.clone());
        let normal_feats = crate::normals::splat_camera_normals(&lifted, &batch.camera);
        let normal_out = brush_render::bwd::render_splats_with_features(
            lifted,
            &batch.camera,
            img_size,
            Vec3::ZERO,
            normal_feats,
        )
        .await;
        let pred_normal = detach_autodiff(normal_out.features.expect("normal features").inner());
        let gt_normal: Tensor<2, Int> = Tensor::from_data(normal_data.clone(), device);
        let nmap = normal_loss_eval(pred_normal, gt_normal); // [2, H, W]
        let nres = nmap.slice(s![0..1, .., ..]).reshape([h, w, 1]);
        res = res + nres * normal_weight;
    }

    let v = Tensor::cat(vec![m.clone(), m * res, Tensor::ones([h, w, 1], device)], 2);
    let v: Tensor<3> = Tensor::from_inner(v);
    let loss = (out.features.expect("evidence features") * v).sum();
    let mut grads = loss.backward();
    let g = feats
        .grad_remove(&mut grads)
        .expect("the zero feature is tracked, so its gradient exists");
    detach_autodiff(g)
}

/// Accumulate [`SplatEvidence`] for `splats` over every view of `scene`.
/// `splats` must live on the inner (non-autodiff) `device`; nothing is
/// mutated, and indices of the result line up with the splat tensors.
pub async fn compute_evidence(
    splats: &Splats,
    scene: &Scene,
    device: &Device,
    normal_weight: f32,
) -> anyhow::Result<SplatEvidence> {
    let n = splats.num_splats() as usize;
    let means = splats.means();
    let mut sums: Tensor<2> = Tensor::zeros([n, 3], device);
    let mut views: Tensor<2> = Tensor::zeros([n, 1], device);
    let mut dir: Tensor<2> = Tensor::zeros([n, 3], device);

    let num_views = scene.views.len();
    for (vi, view) in scene.views.iter().enumerate() {
        let batch = load_view_batch(view)
            .await
            .with_context(|| format!("loading {}", view.image.path().display()))?;
        let g = view_evidence(splats, &batch, device, normal_weight).await;

        let w_in = g.clone().slice(s![.., 0..1]);
        let cam = batch.camera.position;
        let cam_t: Tensor<2> =
            Tensor::<1>::from_floats([cam.x, cam.y, cam.z], device).reshape([1, 3]);
        let d = cam_t.sub(means.clone());
        let d_len = d.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-9);
        dir = dir + d.div(d_len) * w_in.clone();
        views = views + w_in.greater_elem(VIEW_MIN_MASS).float();
        sums = sums + g;
        log::info!("evidence: view {}/{num_views}", vi + 1);
    }

    let data = Transaction::default()
        .register(sums)
        .register(views)
        .register(dir)
        .execute_async()
        .await
        .context("reading evidence back from the GPU")?;
    let mut it = data.into_iter();
    let mut take = || -> anyhow::Result<Vec<f32>> {
        it.next()
            .context("missing evidence tensor")?
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("evidence readback: {e:?}"))
    };
    let sums = take()?;
    let views = take()?;
    let dir = take()?;

    Ok(SplatEvidence {
        w_in: sums.chunks_exact(3).map(|r| r[0]).collect(),
        err: sums.chunks_exact(3).map(|r| r[1]).collect(),
        w_all: sums.chunks_exact(3).map(|r| r[2]).collect(),
        views,
        dir: dir.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect(),
    })
}

/// Drop splats whose in-mask fraction is below `min_inmask`, or that no
/// training view supported. Returns the pruned splats (with any 3D-filter
/// floor baked in, since rows are re-indexed) and the matching evidence.
pub fn prune_by_inmask(
    splats: Splats,
    evidence: &SplatEvidence,
    min_inmask: f32,
) -> (Splats, SplatEvidence) {
    let n = splats.num_splats() as usize;
    assert_eq!(evidence.len(), n, "evidence/splat count mismatch");
    let keep: Vec<bool> = (0..n)
        .map(|i| evidence.views[i] > 0.0 && evidence.inmask(i) >= min_inmask)
        .collect();
    let keep_idx: Vec<i32> = keep
        .iter()
        .enumerate()
        .filter_map(|(i, k)| k.then_some(i as i32))
        .collect();
    if keep_idx.len() == n {
        return (splats, evidence.clone());
    }

    // Bake first: selecting rows of the params alone would leave a stale
    // `[N_old]` min_scale behind.
    let mut splats = splats.bake_min_scale();
    let device = splats.device();
    let k = keep_idx.len();
    let idx: Tensor<1, Int> = Tensor::from_data(TensorData::new(keep_idx, [k]), &device);
    splats.transforms = splats.transforms.map(|t| t.select(0, idx.clone()));
    splats.sh_coeffs = splats.sh_coeffs.map(|c| c.select(0, idx.clone()));
    splats.raw_opacities = splats.raw_opacities.map(|o| o.select(0, idx.clone()));
    (splats, evidence.select(&keep))
}

/// Knobs turning [`SplatEvidence`] into a per-splat, per-view confidence in
/// `[0, 1]`. Defaults were tuned on the body2colmap helical re-render case;
/// see `docs/splat-confidence.md` for what each term catches.
#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ConfidenceParams {
    /// Residual scale of the agreement term `exp(-mean_residual / tau)`
    /// (mean L1 in 0..1 units where the splat is used).
    #[arg(
        long = "conf-tau",
        help_heading = "Confidence options",
        default_value = "0.08"
    )]
    pub tau: f32,
    /// Supporting views needed for full support (smoothstep from 0).
    #[arg(
        long = "conf-min-views",
        help_heading = "Confidence options",
        default_value = "4"
    )]
    pub min_views: f32,
    /// In-mask fraction below which a splat is fully distrusted.
    #[arg(
        long = "conf-inmask-lo",
        help_heading = "Confidence options",
        default_value = "0.3"
    )]
    pub inmask_lo: f32,
    /// In-mask fraction above which a splat is fully trusted.
    #[arg(
        long = "conf-inmask-hi",
        help_heading = "Confidence options",
        default_value = "0.8"
    )]
    pub inmask_hi: f32,
    /// Degrees beyond the observed direction cone a novel view may sit and
    /// still be fully covered.
    #[arg(
        long = "conf-angle-margin",
        help_heading = "Confidence options",
        default_value = "30"
    )]
    pub angle_margin_deg: f32,
    /// Degrees over which coverage fades from 1 to 0 past the margin.
    #[arg(
        long = "conf-angle-soft",
        help_heading = "Confidence options",
        default_value = "15"
    )]
    pub angle_soft_deg: f32,
    /// Also distrust disc-like splats seen at grazing or back-facing angles.
    #[arg(
        long = "conf-facing",
        help_heading = "Confidence options",
        default_value = "false"
    )]
    pub facing: bool,
    /// Angle (degrees, from the splat normal) at which the facing term
    /// reaches zero.
    #[arg(
        long = "conf-graze-deg",
        help_heading = "Confidence options",
        default_value = "80"
    )]
    pub graze_deg: f32,
}

impl Default for ConfidenceParams {
    fn default() -> Self {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrap {
            #[command(flatten)]
            p: ConfidenceParams,
        }
        Wrap::parse_from([""]).p
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Elementwise smoothstep with per-element edges (`e1 >= e0 + eps`).
fn smoothstep_t(e0: Tensor<1>, e1: Tensor<1>, x: Tensor<1>) -> Tensor<1> {
    let t = ((x - e0.clone()) / (e1 - e0).clamp_min(1e-6)).clamp(0.0, 1.0);
    t.clone() * t.clone() * (t * -2.0 + 3.0)
}

/// Splat-local shortest axis in world space, unit length, and whether the
/// splat is disc-like enough for it to mean anything (`min/mid scale < 0.5`).
fn shortest_axis(transform: &[f32]) -> ([f32; 3], bool) {
    let q =
        glam::Quat::from_xyzw(transform[4], transform[5], transform[6], transform[3]).normalize();
    let scales = [transform[7], transform[8], transform[9]];
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| scales[*a].total_cmp(&scales[*b]));
    let axis = match order[0] {
        0 => q * Vec3::X,
        1 => q * Vec3::Y,
        _ => q * Vec3::Z,
    };
    // Log-scales: ratio of true scales = exp(difference).
    let disc = (scales[order[0]] - scales[order[1]]).exp() < 0.5;
    (axis.to_array(), disc)
}

/// Per-splat confidence, split into the view-independent part (computed
/// once, on the CPU) and the view-dependent part (a handful of `[N]` tensor
/// ops per camera).
pub struct ConfidenceModel {
    means: Tensor<2>,
    static_conf: Tensor<1>,
    mu: Tensor<2>,
    cos_in: Tensor<1>,
    cos_out: Tensor<1>,
    facing: Option<(Tensor<2>, Tensor<1>)>,
    cos_graze: f32,
}

impl ConfidenceModel {
    pub async fn new(
        evidence: &SplatEvidence,
        splats: &Splats,
        params: &ConfidenceParams,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let n = splats.num_splats() as usize;
        anyhow::ensure!(
            evidence.len() == n,
            "evidence has {} rows for {n} splats",
            evidence.len()
        );
        let eps = 1e-6f32;
        let m0 = params.angle_margin_deg.to_radians();
        let m1 = (params.angle_margin_deg + params.angle_soft_deg).to_radians();

        let mut static_conf = Vec::with_capacity(n);
        let mut mu = Vec::with_capacity(n * 3);
        let mut cos_in = Vec::with_capacity(n);
        let mut cos_out = Vec::with_capacity(n);
        for i in 0..n {
            let w_in = evidence.w_in[i];
            let inmask = evidence.inmask(i);
            let agree = if w_in > 0.0 {
                (-(evidence.err[i] / w_in.max(eps)) / params.tau.max(eps)).exp()
            } else {
                0.0
            };
            let support = smoothstep(0.0, params.min_views.max(eps), evidence.views[i]);
            let d = Vec3::from(evidence.dir[i]);
            let d_len = d.length();
            let mut conf = smoothstep(params.inmask_lo, params.inmask_hi, inmask) * agree * support;
            if d_len < eps || w_in <= 0.0 {
                conf = 0.0;
            }
            static_conf.push(conf);
            let dir = if d_len > eps { d / d_len } else { Vec3::ZERO };
            mu.extend(dir.to_array());
            // Resultant length -> half-angle of the observed cone (exact
            // for a uniform spherical cap; the margin absorbs the rest).
            let kappa = (d_len / w_in.max(eps)).clamp(0.0, 1.0);
            let phi = (2.0 * kappa - 1.0).clamp(-1.0, 1.0).acos();
            let c_in = (phi + m0).min(std::f32::consts::PI).cos();
            let c_out = (phi + m1).min(std::f32::consts::PI).cos().min(c_in - 1e-3);
            cos_in.push(c_in);
            cos_out.push(c_out);
        }

        let facing = if params.facing {
            let transforms: Vec<f32> = splats
                .transforms
                .val()
                .into_data_async()
                .await
                .context("reading transforms")?
                .into_vec()
                .map_err(|e| anyhow::anyhow!("transforms readback: {e:?}"))?;
            let mut normals = Vec::with_capacity(n * 3);
            let mut gate = Vec::with_capacity(n);
            for i in 0..n {
                let (axis, disc) = shortest_axis(&transforms[i * 10..(i + 1) * 10]);
                let axis = Vec3::from(axis);
                let m = Vec3::new(mu[i * 3], mu[i * 3 + 1], mu[i * 3 + 2]);
                // Orient toward the observed hemisphere.
                let axis = if axis.dot(m) < 0.0 { -axis } else { axis };
                normals.extend(axis.to_array());
                gate.push(if disc { 1.0 } else { 0.0 });
            }
            Some((
                Tensor::from_data(TensorData::new(normals, [n, 3]), device),
                Tensor::from_data(TensorData::new(gate, [n]), device),
            ))
        } else {
            None
        };

        Ok(Self {
            means: splats.means(),
            static_conf: Tensor::from_data(TensorData::new(static_conf, [n]), device),
            mu: Tensor::from_data(TensorData::new(mu, [n, 3]), device),
            cos_in: Tensor::from_data(TensorData::new(cos_in, [n]), device),
            cos_out: Tensor::from_data(TensorData::new(cos_out, [n]), device),
            facing,
            cos_graze: params.graze_deg.to_radians().cos(),
        })
    }

    /// Per-splat confidence in `[0, 1]` for a view from `camera`, `[N]`.
    pub fn for_camera(&self, camera: &Camera) -> Tensor<1> {
        let device = self.means.device();
        let cam = camera.position;
        let cam_t: Tensor<2> =
            Tensor::<1>::from_floats([cam.x, cam.y, cam.z], &device).reshape([1, 3]);
        let v = cam_t.sub(self.means.clone());
        let v_len = v.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-9);
        let v = v.div(v_len);

        let cos_theta = (v.clone() * self.mu.clone()).sum_dim(1).squeeze_dim::<1>(1);
        let coverage = smoothstep_t(self.cos_out.clone(), self.cos_in.clone(), cos_theta);
        let mut conf = self.static_conf.clone() * coverage;

        if let Some((normals, gate)) = &self.facing {
            let dot = (v * normals.clone()).sum_dim(1).squeeze_dim::<1>(1);
            let n = dot.dims()[0];
            let f = smoothstep_t(
                Tensor::zeros([n], &device),
                Tensor::full([n], self.cos_graze, &device),
                dot,
            );
            // Non-disc splats (gate 0) keep facing = 1.
            conf = conf * (gate.clone() * (f.neg() + 1.0)).neg().add_scalar(1.0);
        }
        conf
    }

    /// `[N, 3]` feature tensor `[confidence, 0, 0]` for the rasterizer.
    pub fn feature_for_camera(&self, camera: &Camera) -> Tensor<2> {
        let c = self.for_camera(camera);
        let n = c.dims()[0];
        let zeros = Tensor::zeros([n, 2], &c.device());
        Tensor::cat(vec![c.reshape([n, 1]), zeros], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_round_trip_and_select() {
        let ev = SplatEvidence {
            w_in: vec![1.0, 2.0, 3.0],
            w_all: vec![2.0, 2.0, 6.0],
            err: vec![0.1, 0.2, 0.3],
            views: vec![1.0, 0.0, 5.0],
            dir: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let flat = ev.to_flat();
        assert_eq!(flat.len(), 3 * EVIDENCE_STRIDE);
        assert_eq!(SplatEvidence::from_flat(&flat), ev);
        let sel = ev.select(&[true, false, true]);
        assert_eq!(sel.w_in, vec![1.0, 3.0]);
        assert_eq!(sel.dir, vec![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!((ev.inmask(0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn shortest_axis_picks_smallest_scale() {
        // Identity rotation, z is the shortest (log) scale by a wide margin.
        let t = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0];
        let (axis, disc) = shortest_axis(&t);
        assert!(disc);
        assert!((axis[2].abs() - 1.0).abs() < 1e-6);
        // Isotropic splat: not a disc.
        let t = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!shortest_axis(&t).1);
    }
}
