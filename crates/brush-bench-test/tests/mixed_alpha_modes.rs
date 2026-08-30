//! A dataset may mix alpha modes: on some frames the alpha channel means
//! "ignore this region" (masked — the background is real, we just don't want
//! to fit it), on others it means "nothing is here" (transparent — the model
//! should learn `alpha = 0` there). The mode is per view all the way down, so
//! a single run can carry both. These tests pin the two things that only
//! matter once they're mixed: the trainer must accept an interleaved stream,
//! and `normalize_masked_loss` must put the two on a comparable scale.

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

/// Mid-grey GT. `alpha_left` applies to the left half of the frame and
/// `alpha_right` to the right, so coverage is exactly their average.
fn batch(alpha_mode: AlphaMode, alpha_left: u32, alpha_right: u32) -> SceneBatch {
    let packed: Vec<i32> = (0..(W * H))
        .map(|i| {
            let a = if i % W < W / 2 {
                alpha_left
            } else {
                alpha_right
            };
            (128 | 128 << 8 | 128 << 16 | a << 24) as i32
        })
        .collect();
    let coverage = (alpha_left + alpha_right) as f32 / (2.0 * 255.0);
    SceneBatch {
        img_packed: TensorData::new(packed, [H, W]),
        has_alpha: true,
        alpha_mode,
        alpha_coverage: (alpha_mode == AlphaMode::Masked).then_some(coverage),
        camera: test_camera(),
        normal_data: None,
    }
}

/// Background noise is random per step, and in masked mode it still reaches
/// the loss through the render's `(1 - alpha) * bg` term. Zero it so a loss
/// comparison across two runs is exact.
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

/// The trainer takes one view per step, so a mixed dataset reaches it as an
/// interleaved stream of modes. Both kernel variants (`mask` is a comptime
/// flag) get compiled in the same run.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn alternating_alpha_modes_train_without_crashing() {
    let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    let mut splats = two_splats(&device);
    let config = deterministic_config();
    let mut trainer = SplatTrainer::new(
        &config,
        &device,
        BoundingBox::from_min_max(Vec3::splat(-2.0), Vec3::splat(2.0)),
    );

    for i in 0..6 {
        let b = if i % 2 == 0 {
            batch(AlphaMode::Masked, 255, 0)
        } else {
            batch(AlphaMode::Transparent, 255, 0)
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

/// The loss kernel weights each pixel by `gt.a`, but the trainer averages over
/// the whole frame — so a half-covered mask yields half the loss of the same
/// residual over a full frame. `normalize_masked_loss` divides that back out,
/// which is what puts a masked view on the same footing as a transparent one.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn normalize_masked_loss_divides_by_mask_coverage() {
    let half_masked = batch(AlphaMode::Masked, 255, 0);
    assert_eq!(half_masked.alpha_coverage, Some(0.5));

    let plain = first_step_loss(&deterministic_config(), half_masked.clone()).await;
    let normalized = first_step_loss(
        &TrainConfig {
            normalize_masked_loss: true,
            ..deterministic_config()
        },
        half_masked,
    )
    .await;

    assert!(plain > 0.0, "expected a non-trivial loss, got {plain}");
    assert!(
        (normalized - plain * 2.0).abs() < 1e-4 * plain.max(1.0),
        "coverage 0.5 should double the loss: {plain} -> {normalized}"
    );
}

/// Transparent views are untouched by the flag: their alpha is supervised
/// directly rather than used as a don't-care weight, so there is no coverage
/// to correct for.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn normalize_masked_loss_leaves_transparent_views_alone() {
    let transparent = batch(AlphaMode::Transparent, 255, 0);
    assert_eq!(transparent.alpha_coverage, None);

    let plain = first_step_loss(&deterministic_config(), transparent.clone()).await;
    let normalized = first_step_loss(
        &TrainConfig {
            normalize_masked_loss: true,
            ..deterministic_config()
        },
        transparent,
    )
    .await;

    assert!(
        (normalized - plain).abs() < 1e-4 * plain.max(1.0),
        "transparent loss should be unchanged: {plain} -> {normalized}"
    );
}
