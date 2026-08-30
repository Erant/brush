//! Pins the feature-gradient identity behind `brush_train::evidence`: with a
//! zero per-splat feature and a per-pixel weight map `V`, the feature
//! gradient is `Σ_p vis_ip · V_p` — so the three evidence lanes are exact
//! contribution-mass sums, not approximations.

#![allow(clippy::missing_assert_message)]

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, TextureMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel::Pinhole,
    render_splats,
};
use brush_train::evidence::view_evidence;
use burn::tensor::{Device, TensorData, s};
use glam::{Quat, Vec3};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const W: u32 = 64;
const H: u32 = 64;

/// Two opaque-ish splats in front of a camera at the origin looking down +Z:
/// one projecting into the left half of the image, one into the right.
fn two_splats(device: &Device) -> Splats {
    let means = vec![-1.2, 0.0, 5.0, 1.2, 0.0, 5.0];
    let rotations = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let s = 0.25f32.ln();
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

/// GT: mask (alpha 255) on the left half only, mid-grey colour everywhere.
fn half_masked_batch(camera: Camera) -> SceneBatch {
    let packed: Vec<i32> = (0..(W * H))
        .map(|i| {
            let x = i % W;
            let a: u32 = if x < W / 2 { 255 } else { 0 };
            (128 | 128 << 8 | 128 << 16 | a << 24) as i32
        })
        .collect();
    SceneBatch {
        img_packed: TensorData::new(packed, [H as usize, W as usize]),
        has_alpha: true,
        alpha_mode: AlphaMode::Masked,
        camera,
        normal_data: None,
    }
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn evidence_lanes_are_exact_mass_sums() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();
    let camera = Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        45f64.to_radians(),
        45f64.to_radians(),
        glam::vec2(0.5, 0.5),
        Pinhole,
    );
    let splats = two_splats(&device);
    let batch = half_masked_batch(camera);

    let g: Vec<f32> = view_evidence(&splats, &batch, &device, 0.0)
        .await
        .into_data_async()
        .await
        .unwrap()
        .into_vec()
        .unwrap();
    let (left, right) = (&g[0..3], &g[3..6]);

    // Mass conservation: Σ_i w_all_i == Σ_p alpha_p of the same render.
    let (img, _) = render_splats(
        splats,
        &camera,
        glam::uvec2(W, H),
        Vec3::ZERO,
        None,
        TextureMode::Float,
    )
    .await;
    let alpha_sum: f32 = img
        .slice(s![.., .., 3..4])
        .sum()
        .into_scalar_async()
        .await
        .unwrap();
    let w_all = left[2] + right[2];
    assert!(
        alpha_sum > 10.0,
        "scene should cover a good number of pixels"
    );
    assert!(
        (w_all - alpha_sum).abs() < 1e-3 * alpha_sum,
        "Σ w_all = {w_all}, Σ alpha = {alpha_sum}"
    );

    // The left splat draws only inside the mask, the right one only outside.
    assert!(
        (left[0] - left[2]).abs() < 1e-3 * left[2],
        "left: w_in {} vs w_all {}",
        left[0],
        left[2]
    );
    assert!(
        right[0] < 1e-3 * right[2],
        "right: w_in {} vs w_all {}",
        right[0],
        right[2]
    );

    // Residual lane: L1 against mid-grey is in (0, 1] per pixel, so
    // 0 < e <= w_in for the left splat and e == 0 for the right one.
    assert!(left[1] > 0.0 && left[1] <= left[0] * (1.0 + 1e-4));
    assert!(right[1] < 1e-3 * right[2]);
}
