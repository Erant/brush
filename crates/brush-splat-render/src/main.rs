//! `brush-splat-render` — render a trained `.ply` against an explicit list of
//! cameras, without a training session. Replaces gsplat in the body2colmap
//! pipeline's `RenderSplatStep`; see `brush-render-utility.md` at the repo
//! root for the full design rationale, and `docs/splat-confidence.md` for the
//! `--confidence` output.
//!
//! ```text
//! brush-splat-render \
//!     --splat scene.ply \
//!     --cameras cameras.json \
//!     --output-dir out/ \
//!     [--background 1.0,1.0,1.0] \
//!     [--output-format png] \
//!     [--confidence [--dataset <colmap dir>] [--cull-color 0.5,0.5,0.5] \
//!         [--gate-lo 0.45 --gate-hi 0.65] [--confidence-sidecar]]
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use brush_dataset::config::LoadDatasetConfig;
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::gaussian_splats::{SplatRenderMode, Splats, render_splats_with_features};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{TextureMode, render_splats};
use brush_train::evidence::{ConfidenceModel, ConfidenceParams, SplatEvidence, compute_evidence};
use brush_vfs::BrushVfs;
use clap::Parser;
use glam::{Mat3, Quat, UVec2, Vec2, Vec3};
use image::{GrayImage, Rgba32FImage, RgbaImage};
use serde::Deserialize;

#[derive(Parser)]
#[command(about = "Render a trained splat .ply against an explicit camera list")]
struct Args {
    /// Trained gaussian splat scene, as .ply.
    #[arg(long)]
    splat: PathBuf,

    /// Camera list, in body2colmap.Camera pixel-space terms (see cameras.json schema).
    #[arg(long)]
    cameras: PathBuf,

    /// Directory to write one RGBA image per camera into. Created if missing.
    #[arg(long, default_value = "out")]
    output_dir: PathBuf,

    /// Background color composited under the splat's accumulated alpha, as "r,g,b" in 0..1.
    /// Ignored with `--confidence`, which composites under `--cull-color` instead.
    #[arg(long, default_value = "1.0,1.0,1.0", value_parser = parse_vec3)]
    background: Vec3,

    /// Image encoder used to write output files (extension appended to each camera's name stem).
    #[arg(long, default_value = "png")]
    output_format: String,

    /// Gate every pixel by the per-splat multi-view confidence instead of leaving the
    /// decision to a downstream alpha threshold. Rejected pixels resolve to `--cull-color`
    /// and the written alpha becomes the gate. Needs evidence: `ev_*` properties in the
    /// ply (trained with `--export-evidence`) or `--dataset` to measure it here.
    #[arg(long, help_heading = "Confidence options", default_value = "false")]
    confidence: bool,

    /// Colour that culled (low-confidence) pixels resolve to, as "r,g,b" in 0..1. Also
    /// the compositing background under the splat in confidence mode.
    #[arg(long, help_heading = "Confidence options", default_value = "0.5,0.5,0.5", value_parser = parse_vec3)]
    cull_color: Vec3,

    /// Per-pixel confidence at or below which a pixel is fully culled.
    #[arg(long, help_heading = "Confidence options", default_value = "0.45")]
    gate_lo: f32,

    /// Per-pixel confidence at or above which a pixel is fully kept. Set equal to
    /// `--gate-lo` for a hard cut.
    #[arg(long, help_heading = "Confidence options", default_value = "0.65")]
    gate_hi: f32,

    /// Also write the raw per-pixel confidence as `<stem>.conf.<format>` (8-bit grey).
    #[arg(long, help_heading = "Confidence options", default_value = "false")]
    confidence_sidecar: bool,

    /// Training dataset directory (COLMAP / nerfstudio / `RealityCapture`) to measure
    /// evidence against when the ply carries none. Loaded with the dataset options below,
    /// which should match the training run (notably `--alpha-mode`).
    #[arg(long, help_heading = "Confidence options")]
    dataset: Option<PathBuf>,

    /// Weight of the normal-map residual in the evidence residual when the dataset has
    /// `normals/`. 0 skips the extra normal render.
    #[arg(long, help_heading = "Confidence options", default_value = "0.0")]
    evidence_normal_weight: f32,

    /// After measuring evidence from `--dataset`, also write the splat back out to this
    /// .ply with the `ev_*` properties, so later renders can skip the dataset.
    #[arg(long, help_heading = "Confidence options")]
    write_evidence: Option<PathBuf>,

    #[command(flatten)]
    conf: ConfidenceParams,

    #[command(flatten)]
    load: LoadDatasetConfig,
}

fn parse_vec3(s: &str) -> Result<Vec3, String> {
    let parts: Vec<&str> = s.split(',').collect();
    let [r, g, b] = parts.as_slice() else {
        return Err(format!("expected \"r,g,b\", got \"{s}\""));
    };
    let parse = |p: &str| p.trim().parse::<f32>().map_err(|e| e.to_string());
    Ok(Vec3::new(parse(r)?, parse(g)?, parse(b)?))
}

/// One entry in `cameras.json`, in `body2colmap.Camera`'s pixel-space terms:
/// intrinsics as (fx, fy, cx, cy) in pixels, extrinsics as an OpenGL-convention
/// (Y-up, camera looks down -Z) camera-to-world pose.
#[derive(Deserialize)]
struct CameraEntry {
    name: String,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    position: [f32; 3],
    /// Camera-to-world rotation matrix, row-major (`rotation[row][col]`).
    /// Columns are the camera's local axes (X right, Y up, Z backward)
    /// expressed in world coordinates — `body2colmap.Camera.rotation`
    /// serialized directly.
    rotation: [[f32; 3]; 3],
}

#[derive(Deserialize)]
struct CamerasFile {
    width: u32,
    height: u32,
    cameras: Vec<CameraEntry>,
}

/// Convert a body2colmap camera entry (OpenGL c2w: Y-up, -Z-forward local
/// axes) to brush's `Camera` (`OpenCV` c2w: Y-down, +Z-forward local axes,
/// same world frame — see `brush-render-utility.md`, "The camera convention
/// is the risky part").
///
/// Derivation: body2colmap's own gsplat path left-multiplies its w2c by
/// `diag(1,-1,-1,1)` to go OpenGL -> `OpenCV` (`splat_renderer.py`). Inverting
/// that relation gives the c2w rotation directly: `R_cv = R_gl @ diag(1,-1,-1)`,
/// i.e. negate the Y and Z *columns* of the input matrix (columns are local
/// axes in world coords) and leave X untouched. Position is unchanged: both
/// sides export to the same COLMAP world frame with no world-level transform
/// (`body2colmap.coordinates.world_to_colmap_camera` only flips the
/// per-camera local axes, never the world axes).
fn to_brush_camera(entry: &CameraEntry, width: u32, height: u32) -> Camera {
    let r = entry.rotation;
    let col_x = Vec3::new(r[0][0], r[1][0], r[2][0]);
    let col_y = Vec3::new(-r[0][1], -r[1][1], -r[2][1]);
    let col_z = Vec3::new(-r[0][2], -r[1][2], -r[2][2]);
    let rotation = Quat::from_mat3(&Mat3::from_cols(col_x, col_y, col_z));
    let position = Vec3::from(entry.position);

    let fov_x = focal_to_fov(entry.fx, width, &CameraModel::Pinhole);
    let fov_y = focal_to_fov(entry.fy, height, &CameraModel::Pinhole);
    let center_uv = Vec2::new(
        entry.cx as f32 / width as f32,
        entry.cy as f32 / height as f32,
    );

    Camera::new(
        position,
        rotation,
        fov_x,
        fov_y,
        center_uv,
        CameraModel::Pinhole,
    )
}

/// Plain render: RGB composited over `background`, alpha = accumulated
/// splat opacity.
async fn render_plain(
    splats: Splats,
    camera: &Camera,
    img_size: UVec2,
    background: Vec3,
    output_path: &Path,
) -> Result<()> {
    let (img, _aux) = render_splats(
        splats,
        camera,
        img_size,
        background,
        None,
        TextureMode::Float,
    )
    .await;

    let [h, w, c] = [img.dims()[0], img.dims()[1], img.dims()[2]];
    anyhow::ensure!(c == 4, "expected an RGBA render, got {c} channels");
    let data = img.into_data_async().await?.into_vec::<f32>()?;

    let image = Rgba32FImage::from_raw(w as u32, h as u32, data)
        .context("render output size didn't match its own tensor dims")?;
    let image = image::DynamicImage::from(image).into_rgba8();
    image.save(output_path)?;
    Ok(())
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    if hi <= lo {
        return if x >= lo { 1.0 } else { 0.0 };
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Confidence-gated render: composite over `cull`, blend toward `cull` by
/// the gate, write the gate as alpha; optionally the raw confidence beside it.
#[allow(clippy::too_many_arguments)]
async fn render_gated(
    splats: Splats,
    model: Option<&ConfidenceModel>,
    camera: &Camera,
    img_size: UVec2,
    cull: Vec3,
    gate: (f32, f32),
    output_path: &Path,
    sidecar_path: Option<&Path>,
) -> Result<()> {
    let n = splats.num_splats() as usize;
    let feats = if let Some(m) = model {
        m.feature_for_camera(camera)
    } else {
        // No evidence: every splat fully trusted, so the confidence channel
        // degenerates to accumulated alpha.
        let device = splats.device();
        burn::tensor::Tensor::cat(
            vec![
                burn::tensor::Tensor::ones([n, 1], &device),
                burn::tensor::Tensor::zeros([n, 2], &device),
            ],
            1,
        )
    };
    let (img, _aux) = render_splats_with_features(
        splats,
        camera,
        img_size,
        cull,
        None,
        TextureMode::Float,
        Some(feats),
    )
    .await;

    let [h, w, c] = [img.dims()[0], img.dims()[1], img.dims()[2]];
    anyhow::ensure!(c == 7, "expected an RGBA+feature render, got {c} channels");
    let data = img.into_data_async().await?.into_vec::<f32>()?;

    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgba = Vec::with_capacity(w * h * 4);
    let mut conf = Vec::with_capacity(w * h);
    for px in data.chunks_exact(7) {
        // px[0..3] is already composited over `cull` (the render background).
        let c = px[4].clamp(0.0, 1.0);
        let g = smoothstep(gate.0, gate.1, c);
        rgba.extend([
            to_u8(px[0] * g + cull.x * (1.0 - g)),
            to_u8(px[1] * g + cull.y * (1.0 - g)),
            to_u8(px[2] * g + cull.z * (1.0 - g)),
            to_u8(g),
        ]);
        conf.push(to_u8(c));
    }
    RgbaImage::from_raw(w as u32, h as u32, rgba)
        .context("render output size didn't match its own tensor dims")?
        .save(output_path)?;
    if let Some(sidecar) = sidecar_path {
        GrayImage::from_raw(w as u32, h as u32, conf)
            .context("confidence size mismatch")?
            .save(sidecar)?;
    }
    Ok(())
}

/// Evidence for the confidence model, in order of preference: carried by the
/// ply, measured against `--dataset`, or absent (warn; confidence then
/// degenerates to alpha).
async fn resolve_evidence(
    args: &Args,
    splats: &Splats,
    ply_evidence: Option<Vec<f32>>,
    device: &burn::tensor::Device,
) -> Result<Option<SplatEvidence>> {
    if let Some(flat) = ply_evidence {
        let ev = SplatEvidence::from_flat(&flat);
        anyhow::ensure!(
            ev.len() == splats.num_splats() as usize,
            "ply evidence has {} rows for {} splats",
            ev.len(),
            splats.num_splats()
        );
        log::info!(
            "using the evidence block carried by {}",
            args.splat.display()
        );
        return Ok(Some(ev));
    }
    let Some(dataset_dir) = &args.dataset else {
        log::warn!(
            "--confidence without evidence: {} carries no ev_* properties and no --dataset was given; every splat is treated as fully trusted (confidence = alpha)",
            args.splat.display()
        );
        return Ok(None);
    };
    let vfs = Arc::new(
        BrushVfs::from_path(dataset_dir)
            .await
            .with_context(|| format!("opening dataset {}", dataset_dir.display()))?,
    );
    let loaded = brush_dataset::load_dataset(vfs, &args.load)
        .await
        .with_context(|| format!("loading dataset {}", dataset_dir.display()))?;
    for warning in loaded.warnings {
        log::warn!("{warning}");
    }
    let scene = &loaded.dataset.train;
    log::info!(
        "measuring evidence against {} training views from {}",
        scene.views.len(),
        dataset_dir.display()
    );
    let ev = compute_evidence(splats, scene, device, args.evidence_normal_weight).await?;
    if let Some(path) = &args.write_evidence {
        let bytes =
            brush_serde::splat_to_ply_with_evidence(splats.clone(), None, Some(&ev.to_flat()))
                .await
                .context("serialising splat with evidence")?;
        tokio::fs::write(path, bytes)
            .await
            .with_context(|| format!("writing {}", path.display()))?;
        log::info!("wrote evidence-carrying ply to {}", path.display());
    }
    Ok(Some(ev))
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let cameras_json = std::fs::read_to_string(&args.cameras)
        .with_context(|| format!("reading {}", args.cameras.display()))?;
    let cameras_file: CamerasFile =
        serde_json::from_str(&cameras_json).context("parsing cameras.json")?;
    let img_size = UVec2::new(cameras_file.width, cameras_file.height);

    let wgpu_device = brush_process::burn_init_setup().await;
    let device: burn::tensor::Device = wgpu_device.into();

    let ply_file = tokio::fs::File::open(&args.splat)
        .await
        .with_context(|| format!("opening {}", args.splat.display()))?;
    let msg = brush_serde::load_splat_from_ply(tokio::io::BufReader::new(ply_file), None)
        .await
        .with_context(|| format!("loading {}", args.splat.display()))?;
    let render_mode = msg.meta.render_mode.unwrap_or(SplatRenderMode::Default);
    let ply_evidence = msg.data.evidence.clone();
    let splats = msg.data.into_splats(&device, render_mode);
    log::info!(
        "loaded {} splats from {} ({render_mode:?} render mode)",
        splats.num_splats(),
        args.splat.display()
    );

    let model = if args.confidence {
        match resolve_evidence(&args, &splats, ply_evidence, &device).await? {
            Some(ev) => Some(ConfidenceModel::new(&ev, &splats, &args.conf, &device).await?),
            None => None,
        }
    } else {
        None
    };

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    for entry in &cameras_file.cameras {
        let camera = to_brush_camera(entry, img_size.x, img_size.y);
        anyhow::ensure!(
            camera.is_valid(),
            "camera '{}' is invalid (nan/inf)",
            entry.name
        );

        let stem = Path::new(&entry.name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&entry.name);
        let output_path = args
            .output_dir
            .join(format!("{stem}.{}", args.output_format));

        if args.confidence {
            let sidecar = args.confidence_sidecar.then(|| {
                args.output_dir
                    .join(format!("{stem}.conf.{}", args.output_format))
            });
            render_gated(
                splats.clone(),
                model.as_ref(),
                &camera,
                img_size,
                args.cull_color,
                (args.gate_lo, args.gate_hi),
                &output_path,
                sidecar.as_deref(),
            )
            .await?;
        } else {
            render_plain(
                splats.clone(),
                &camera,
                img_size,
                args.background,
                &output_path,
            )
            .await?;
        }
        log::info!("rendered '{}' -> {}", entry.name, output_path.display());
    }

    Ok(())
}
