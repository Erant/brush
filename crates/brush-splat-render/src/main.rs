//! `brush-splat-render` — render a trained `.ply` against an explicit list of
//! cameras, without a training session. Replaces gsplat in the body2colmap
//! pipeline's `RenderSplatStep`; see `brush-render-utility.md` at the repo
//! root for the full design rationale.
//!
//! ```text
//! brush-splat-render \
//!     --splat scene.ply \
//!     --cameras cameras.json \
//!     --output-dir out/ \
//!     [--background 1.0,1.0,1.0] \
//!     [--output-format png]
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{TextureMode, render_splats};
use clap::Parser;
use glam::{Mat3, Quat, UVec2, Vec2, Vec3};
use image::Rgba32FImage;
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
    #[arg(long, default_value = "1.0,1.0,1.0", value_parser = parse_vec3)]
    background: Vec3,

    /// Image encoder used to write output files (extension appended to each camera's name stem).
    #[arg(long, default_value = "png")]
    output_format: String,
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

async fn render_one(
    splats: brush_render::Splats,
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
    let splats = msg.data.into_splats(&device, SplatRenderMode::Default);
    log::info!(
        "loaded {} splats from {}",
        splats.num_splats(),
        args.splat.display()
    );

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

        render_one(
            splats.clone(),
            &camera,
            img_size,
            args.background,
            &output_path,
        )
        .await?;
        log::info!("rendered '{}' -> {}", entry.name, output_path.display());
    }

    Ok(())
}
