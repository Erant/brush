#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::{mean_alpha, sample_to_packed_data, view_to_sample_image};
use brush_loss::{ImageLossConfig, image_loss_eval};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, RenderAux, TextureMode, render_splats};
use burn::tensor::{Device, Int, Tensor, s};
use glam::Vec3;
use image::DynamicImage;

pub struct EvalSample {
    pub gt_img: DynamicImage,
    pub rendered: Tensor<3>,
    pub psnr: Tensor<1>,
    pub ssim: Tensor<1>,
    pub render_aux: RenderAux,
}

/// `normalize_masked` mirrors the trainer's `normalize_masked_loss`: on a
/// [`AlphaMode::Masked`] view it scores only the masked region (weighting the
/// residual by `gt.a` and dividing by mask coverage) instead of the whole
/// frame. Without it, a masked view's PSNR is diluted by background the model
/// was never asked to fit, which makes metrics incomparable across views in a
/// run that mixes alpha modes. Off by default so existing numbers don't shift.
pub async fn eval_stats(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    normalize_masked: bool,
    device: &Device,
) -> Result<EvalSample> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let sample = view_to_sample_image(gt_img.clone(), alpha_mode);
    // Exact for binary masks: the mse path squares the already-`a`-weighted
    // residual, and `a^2 == a` only when `a` is 0 or 1. Soft masks are
    // approximated (they weight by `a^2` while dividing by mean `a`).
    let masked = normalize_masked && alpha_mode == AlphaMode::Masked && sample.color().has_alpha();
    let coverage = if masked {
        mean_alpha(&sample).max(crate::train::MIN_MASK_COVERAGE)
    } else {
        1.0
    };
    let (gt_packed_data, _has_alpha) = sample_to_packed_data(sample);
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

    // Render on reference black background.
    let (img, render_aux) =
        render_splats(splats, gt_cam, res, Vec3::ZERO, None, TextureMode::Float).await;
    let render_rgb = img.slice(s![.., .., 0..3]);

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (render_rgb * 255.0).round() / 255.0;

    let cfg = |l1, ssim| ImageLossConfig {
        l1_weight: l1,
        ssim_weight: ssim,
        composite_bg: None,
        mask: masked,
    };
    // MSE = mean(L1^2) since |a - b|^2 == (a - b)^2. Dividing by coverage turns
    // the whole-frame mean into a mean over the masked region.
    let mse = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0))
        .powi_scalar(2)
        .mean()
        / coverage;
    let psnr = mse.recip().log() * 10.0 / std::f32::consts::LN_10;
    let ssim = image_loss_eval(render_rgb.clone(), gt_packed, cfg(0.0, 1.0)).mean() / coverage;

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        rendered: render_rgb,
        render_aux,
    })
}

impl EvalSample {
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = self.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = img.clone().into_data_async().await?.into_vec::<f32>()?;
        let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
            .expect("Failed to create image from tensor")
            .into();
        let img: image::DynamicImage = img.into_rgb8().into();
        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
        Ok(())
    }
}
