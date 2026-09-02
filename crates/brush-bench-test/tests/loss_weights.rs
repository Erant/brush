//! A `weights/` sidecar is a per-pixel multiplier on a view's loss map, on
//! top of whatever its alpha mode does. It is the one channel a transparent
//! view otherwise lacks — its alpha is a target, not a weight — and the way
//! to tell the trainer "listen to this view less over there" without
//! touching what it says about the silhouette. These pin the arithmetic:
//! the multiply is exact, it reaches every term, and a weighted stream
//! trains alongside an unweighted one.

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode,
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
};
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::tensor::{Device, TensorData};
use glam::{Quat, Vec3};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const W: usize = 64;
const H: usize = 64;

fn test_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, -4.0),
        Quat::IDENTITY,
        45f64.to_radians(),
        45f64.to_radians(),
        glam::vec2(0.5, 0.5),
        Pinhole,
    )
}

fn two_splats(device: &Device) -> Splats {
    let means = vec![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0];
    let rotations = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let s = 0.3f32.ln();
    let log_scales = vec![s, s, s, s, s, s];
    let sh = vec![0.6, 0.2, 0.2, 0.2, 0.6, 0.2];
    let opac = vec![3.0, 3.0];
    Splats::from_raw(
        means,
        rotations,
        log_scales,
        sh,
        opac,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

/// Mid-grey GT, opaque on the left half and empty on the right, so a
/// transparent view has both an RGB residual and an alpha-match residual to
/// weight. `weight` is `Some((left, right))` for a sidecar that is one value
/// over each half, `None` for no sidecar.
fn batch(alpha_mode: AlphaMode, weight: Option<(f32, f32)>) -> SceneBatch {
    let packed: Vec<i32> = (0..(W * H))
        .map(|i| {
            let a: u32 = if i % W < W / 2 { 255 } else { 0 };
            (128 | 128 << 8 | 128 << 16 | a << 24) as i32
        })
        .collect();
    let loss_weight = weight.map(|(left, right)| {
        let w: Vec<f32> = (0..(W * H))
            .map(|i| if i % W < W / 2 { left } else { right })
            .collect();
        TensorData::new(w, [H, W])
    });
    SceneBatch {
        img_packed: TensorData::new(packed, [H, W]),
        has_alpha: true,
        alpha_mode,
        alpha_coverage: (alpha_mode == AlphaMode::Masked).then_some(0.5),
        camera: test_camera(),
        normal_data: None,
        loss_weight,
    }
}

fn deterministic_config() -> TrainConfig {
    TrainConfig {
        background_noise_strength: 0.0,
        ..TrainConfig::default()
    }
}

async fn first_step_loss(config: &TrainConfig, batch: SceneBatch) -> f32 {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let splats = two_splats(&device);
    let mut trainer = SplatTrainer::new(
        config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );
    let (_splats, stats) = trainer.step(batch, splats).await;
    stats
        .loss
        .into_scalar_async::<f32>()
        .await
        .expect("loss readback")
}

fn assert_close(got: f32, want: f32, what: &str) {
    assert!(
        (got - want).abs() < 1e-4 * want.abs().max(1.0),
        "{what}: expected {want}, got {got}"
    );
}

/// The multiply is exact and reaches the alpha-match lane: a uniform weight
/// of one half halves a transparent view's whole loss, RGB and alpha alike.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn a_uniform_weight_scales_a_transparent_view_s_loss_exactly() {
    let config = deterministic_config();
    let plain = first_step_loss(&config, batch(AlphaMode::Transparent, None)).await;
    let half = first_step_loss(&config, batch(AlphaMode::Transparent, Some((0.5, 0.5)))).await;
    let ones = first_step_loss(&config, batch(AlphaMode::Transparent, Some((1.0, 1.0)))).await;

    assert!(plain > 0.0, "expected a non-trivial loss, got {plain}");
    assert_close(ones, plain, "a weight of 1 everywhere is no weight at all");
    assert_close(half, plain * 0.5, "a weight of 0.5 everywhere");
}

/// A weight of zero silences a region outright: what is left is exactly the
/// other half's contribution, and silencing both halves leaves nothing.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn a_zero_weight_silences_a_region() {
    let config = deterministic_config();
    let plain = first_step_loss(&config, batch(AlphaMode::Transparent, None)).await;
    let left_only = first_step_loss(&config, batch(AlphaMode::Transparent, Some((1.0, 0.0)))).await;
    let right_only =
        first_step_loss(&config, batch(AlphaMode::Transparent, Some((0.0, 1.0)))).await;
    let nothing = first_step_loss(&config, batch(AlphaMode::Transparent, Some((0.0, 0.0)))).await;

    assert_close(
        left_only + right_only,
        plain,
        "the two halves partition the loss",
    );
    assert!(
        left_only > 0.0 && right_only > 0.0,
        "{left_only} / {right_only}"
    );
    assert_close(nothing, 0.0, "a weight of 0 everywhere");
}

/// On a masked view the sidecar stacks with the mask: the mask already zeroes
/// the right half, so weighting the left half by a half halves the loss.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn a_weight_stacks_with_a_mask() {
    let config = deterministic_config();
    let plain = first_step_loss(&config, batch(AlphaMode::Masked, None)).await;
    let half_left = first_step_loss(&config, batch(AlphaMode::Masked, Some((0.5, 1.0)))).await;

    assert!(plain > 0.0, "expected a non-trivial loss, got {plain}");
    assert_close(
        half_left,
        plain * 0.5,
        "half weight over the masked-in half",
    );
}

/// Weighted and unweighted views arrive interleaved in one run.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn weighted_and_unweighted_views_train_together() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut splats = two_splats(&device);
    let config = deterministic_config();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );

    for i in 0..6 {
        let b = match i % 3 {
            0 => batch(AlphaMode::Transparent, None),
            1 => batch(AlphaMode::Transparent, Some((0.1, 1.0))),
            _ => batch(AlphaMode::Masked, Some((0.1, 1.0))),
        };
        let (new_splats, stats) = trainer.step(b, splats).await;
        splats = new_splats;
        let loss = stats
            .loss
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");
        assert!(loss.is_finite(), "step {i} produced a non-finite loss");
    }
    assert!(splats.num_splats() > 0);
}
