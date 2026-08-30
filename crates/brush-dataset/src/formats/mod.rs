use crate::{Dataset, config::LoadDatasetConfig, scene::SceneView};
use brush_render::AlphaMode;
use brush_serde::{DeserializeError, SplatMessage, load_splat_from_ply};

use brush_vfs::BrushVfs;
use image::ImageError;
use itertools::{Either, Itertools};
use std::{path::Path, sync::Arc};

pub mod colmap;
pub mod nerfstudio;
pub mod realitycapture;

use thiserror::Error;

pub struct DatasetLoadResult {
    pub init_splat: Option<SplatMessage>,
    pub dataset: Dataset,
    pub warnings: Vec<String>,
}

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("I/O error while loading dataset: {0}")]
    Io(#[from] std::io::Error),

    #[error("Error decoding JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Error decoding camera parameters: {0}")]
    InvalidCamera(String),

    #[error("Error when decoding format: {0}")]
    InvalidFormat(String),

    #[error("Error loading splat data: {0}")]
    PlyError(#[from] DeserializeError),

    #[error("Error loading image in data: {0}")]
    ImageError(#[from] ImageError),
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error(transparent)]
    FormatError(#[from] FormatError),

    #[error("Failed to load initial point cloud: {0}")]
    InitialPointCloudError(#[from] DeserializeError),

    #[error(
        "Format not recognized: only colmap, nerfstudio json and RealityCapture csv are supported"
    )]
    FormatNotSupported,
}

pub async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Result<DatasetLoadResult, DatasetError> {
    let mut dataset = colmap::load_dataset(vfs.clone(), load_args).await;

    if dataset.is_none() {
        dataset = nerfstudio::read_dataset(vfs.clone(), load_args).await;
    }

    if dataset.is_none() {
        dataset = realitycapture::read_dataset(vfs.clone(), load_args).await;
    }

    let Some(dataset) = dataset else {
        return Err(DatasetError::FormatNotSupported);
    };

    let mut result = dataset?;

    // A dataset that parsed but has no usable training views (e.g. every image
    // was missing or filtered out) would otherwise "load" and then crash on the
    // first training batch. Reject it here with a typed error instead.
    if result.dataset.train.views.is_empty() {
        return Err(FormatError::InvalidFormat(
            "dataset contains no usable training views (all images missing or filtered out)"
                .to_owned(),
        )
        .into());
    }

    report_alpha_modes(&result.dataset, load_args, &mut result.warnings);

    // If there's an initial ply file, override the init stream with that.
    let mut ply_paths: Vec<_> = vfs.files_with_extension("ply").collect();
    ply_paths.sort();

    let main_ply = ply_paths
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n == "init.ply"))
        .or_else(|| ply_paths.last());

    let init_splat = if let Some(main_ply) = main_ply {
        log::info!("Using ply {main_ply:?} as initial point cloud.");
        let reader = vfs
            .reader_at_path(main_ply)
            .await
            .map_err(DeserializeError)?;
        Some(load_splat_from_ply(reader, load_args.subsample_points).await?)
    } else {
        result.init_splat
    };

    Ok(DatasetLoadResult {
        init_splat,
        dataset: result.dataset,
        warnings: result.warnings,
    })
}

/// Log how the dataset's views resolved to alpha modes, and warn when an
/// explicit `--alpha-mode` is flattening a mix the `masks/` sidecar layout
/// would otherwise have produced.
///
/// A run can legitimately mix modes — some frames' alpha means "ignore this
/// region" (masked), others' means "nothing is here" (transparent) — and the
/// trainer handles that per view. The failure mode this guards against is
/// silent: passing `--alpha-mode` forces every view, so a carefully prepared
/// mix collapses to one interpretation with nothing in the log to say so.
fn report_alpha_modes(
    dataset: &Dataset,
    load_args: &LoadDatasetConfig,
    warnings: &mut Vec<String>,
) {
    let views = || {
        dataset
            .train
            .views
            .iter()
            .chain(dataset.eval.iter().flat_map(|s| s.views.iter()))
    };

    let (mut masked, mut transparent) = (0usize, 0usize);
    for view in views() {
        match view.image.alpha_mode() {
            AlphaMode::Masked => masked += 1,
            AlphaMode::Transparent => transparent += 1,
        }
    }
    log::info!("Dataset alpha modes: {masked} masked, {transparent} transparent view(s)");

    if let Some(forced) = load_args.alpha_mode {
        let with_sidecar = views().filter(|v| v.image.has_mask_sidecar()).count();
        let total = masked + transparent;
        // A mix is only *implied* when the sidecars disagree with each other:
        // some views have one and some don't.
        if with_sidecar > 0 && with_sidecar < total {
            warnings.push(format!(
                "--alpha-mode forced all {total} views to {forced:?}, but {with_sidecar} of them \
                 have a masks/ sidecar and {} do not — without the flag those would have loaded \
                 as masked and transparent respectively. Drop --alpha-mode to train on the mix.",
                total - with_sidecar,
            ));
        }
    }
}

/// Resolve a bare image name (as stored by colmap / `RealityCapture`, which only
/// record a filename) to a path in the VFS by brute-force suffix search. Masks
/// are skipped so an image never resolves to its own mask.
fn find_image_by_name<'a>(vfs: &'a BrushVfs, name: &str) -> Option<&'a Path> {
    vfs.files_ending_in(name)
        .filter(|p| !p.iter().any(|f| f == "masks"))
        .min()
}

/// Convert an OpenGL/Blender camera-to-world matrix (the nerfstudio
/// `transform_matrix` convention: +X right, +Y up, +Z back) into brush's
/// camera pose (+X right, +Y down, +Z forward).
fn opengl_c2w_to_pose(mut c2w: glam::Mat4) -> (glam::Vec3, glam::Quat) {
    c2w.y_axis *= -1.0;
    c2w.z_axis *= -1.0;
    let (_, rotation, translation) = c2w.to_scale_rotation_translation();
    (translation, rotation)
}

/// Split views into (train, eval) by selecting every `eval_split_every`-th view
/// for eval. With `None`, every view is a train view.
fn split_eval_every(
    views: Vec<SceneView>,
    eval_split_every: Option<usize>,
) -> (Vec<SceneView>, Vec<SceneView>) {
    views.into_iter().enumerate().partition_map(|(i, v)| {
        if let Some(split) = eval_split_every
            && i % split == 0
        {
            Either::Right(v)
        } else {
            Either::Left(v)
        }
    })
}

/// Find a sibling monocular normal-map image for `path`, by looking for a
/// `normals/` directory whose subpath (relative to `path`'s own directory)
/// matches, mirroring [`find_mask_path`]'s convention but for `normals`
/// instead of `masks`, and without the `.mask` stem-suffix variant (not used
/// by the normal-map dataset convention this targets).
fn find_normal_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");

    vfs.iter_files().find(|candidate| {
        let Some(stem) = candidate.file_stem() else {
            return false;
        };

        if stem.eq_ignore_ascii_case(search_name) || stem.eq_ignore_ascii_case(search_stem) {
            let normals_idx = candidate
                .components()
                .position(|c| c.as_os_str().eq_ignore_ascii_case("normals"));

            normals_idx.is_some_and(|idx| {
                let candidate_components: Vec<_> = candidate.components().collect();
                let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
                let normal_dir_subpath =
                    &candidate_components[idx + 1..candidate_components.len() - 1];
                path_dir_components.ends_with(normal_dir_subpath)
            })
        } else {
            false
        }
    })
}

fn find_mask_path<'a>(vfs: &'a BrushVfs, path: &'a Path) -> Option<&'a Path> {
    let search_name = path.file_name().expect("File must have a name");
    let search_stem = path.file_stem().expect("File must have a name");
    let mut search_mask = search_stem.to_owned();
    search_mask.push(".mask");
    let search_mask = &search_mask;

    vfs.iter_files().find(|candidate| {
        // For the target, we don't care about its actual extension. Lets see if either the name or stem matches.
        let Some(stem) = candidate.file_stem() else {
            return false;
        };

        // We have the name of the file a la img.png, and the stem a la img.
        // We now want to accept any of img.png.*, img.*, img.mask.*.
        if stem.eq_ignore_ascii_case(search_name)
            || stem.eq_ignore_ascii_case(search_stem)
            || stem.eq_ignore_ascii_case(search_mask)
        {
            // Find "masks" directory in candidate path
            let masks_idx = candidate
                .components()
                .position(|c| c.as_os_str().eq_ignore_ascii_case("masks"));

            // Check if the image directory path ends with the directory subpath after "masks/"
            // e.g., masks/foo/bar/bla.png should match images/foo/bar/bla.jpeg
            masks_idx.is_some_and(|idx| {
                let candidate_components: Vec<_> = candidate.components().collect();

                // Get directory components only (excluding filename)
                let path_dir_components: Vec<_> = path.parent().unwrap().components().collect();
                let mask_dir_subpath =
                    &candidate_components[idx + 1..candidate_components.len() - 1];
                path_dir_components.ends_with(mask_dir_subpath)
            })
        } else {
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use wasm_bindgen_test::wasm_bindgen_test;

    /// A dataset may legitimately mix alpha modes: alpha means "ignore this
    /// region" on frames that carry a `masks/` sidecar, and "nothing is here"
    /// on frames whose alpha is embedded in the image itself. The sidecar
    /// layout alone decides, per view, which is which.
    #[wasm_bindgen_test(unsupported = test)]
    fn mixed_mask_layout_resolves_per_view_alpha_modes() {
        let vfs = Arc::new(BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/masked.png"),
            PathBuf::from("images/transparent.png"),
            PathBuf::from("masks/masked.png"),
        ]));

        let mode = |name: &str| {
            let path = PathBuf::from(format!("images/{name}.png"));
            let mask = find_mask_path(&vfs, &path).map(Path::to_path_buf);
            crate::scene::LoadImage::new(vfs.clone(), path, mask, 1920, None).alpha_mode()
        };

        assert_eq!(mode("masked"), AlphaMode::Masked);
        assert_eq!(mode("transparent"), AlphaMode::Transparent);
    }

    /// `--alpha-mode` is a global force, not a per-view default: it overrides
    /// the sidecar signal for every view. The b2crunner pipeline depends on
    /// this (it writes alpha into `masks/` then passes `--alpha-mode
    /// transparent`), which is why `report_alpha_modes` warns instead of
    /// changing the precedence.
    #[wasm_bindgen_test(unsupported = test)]
    fn explicit_alpha_mode_overrides_the_sidecar_signal() {
        let vfs = Arc::new(BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/masked.png"),
            PathBuf::from("masks/masked.png"),
        ]));
        let path = PathBuf::from("images/masked.png");
        let mask = find_mask_path(&vfs, &path).map(Path::to_path_buf);

        for forced in [AlphaMode::Masked, AlphaMode::Transparent] {
            let image = crate::scene::LoadImage::new(
                vfs.clone(),
                path.clone(),
                mask.clone(),
                1920,
                Some(forced),
            );
            assert_eq!(image.alpha_mode(), forced);
            assert!(image.has_mask_sidecar());
        }
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_mask() {
        // Basic matching with same extension
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.png"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/img.png")),
            Some(Path::new("masks/img.png"))
        );
        // Different extensions are ok.
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/img.jpeg"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/img.jpeg")),
            Some(Path::new("masks/img.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_mask_formats() {
        // Test img.png.mask format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo.png"),
            PathBuf::from("masks/foo.png.mask"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/foo.png")),
            Some(Path::new("masks/foo.png.mask"))
        );

        // Test img.mask.png format
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/bar.jpeg"),
            PathBuf::from("masks/bar.mask.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/bar.jpeg")),
            Some(Path::new("masks/bar.mask.png"))
        );
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_nested_dirs() {
        // Nested directories must match
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/foo/bar/img.png"),
            PathBuf::from("masks/foo/bar/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/foo/bar/img.png")),
            Some(Path::new("masks/foo/bar/img.png"))
        );
        // Should not match wrong subpath
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/baz/img.png"),
            PathBuf::from("masks/foo/img.png"),
        ]);
        assert_eq!(find_mask_path(&vfs, Path::new("images/baz/img.png")), None);
    }

    #[wasm_bindgen_test(unsupported = test)]
    fn test_find_case_insensitive() {
        let vfs = BrushVfs::create_test_vfs(vec![
            PathBuf::from("images/IMG.PNG"),
            PathBuf::from("masks/img.png"),
        ]);
        assert_eq!(
            find_mask_path(&vfs, Path::new("images/IMG.PNG")),
            Some(Path::new("masks/img.png"))
        );
    }
}
