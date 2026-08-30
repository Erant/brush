use clap::{Args, Parser};
use serde::{Deserialize, Serialize};

#[derive(Clone, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ProcessConfig {
    /// Random seed.
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,
    /// Iteration to resume from
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,
    /// Eval every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "1000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub eval_every: u32,
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    pub eval_save_to_disk: bool,
    /// Export every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "5000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub export_every: u32,
    /// Location to put exported files. Supports {dataset} interpolation for the dataset
    /// folder name, and {timestamp} interpolation for the process start time (as Unix
    /// seconds) — opt into the latter (e.g. "./{dataset}_exports/{timestamp}/") to give
    /// repeated runs their own directory so they never overwrite a previous run's
    /// exports. Path is relative to the dataset's parent directory (or CWD if
    /// unavailable). Use "./{dataset}/" to export inside the dataset folder.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./{dataset}_exports/"
    )]
    pub export_path: String,
    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "export_{iter}.ply"
    )]
    pub export_name: String,
    /// At the end of training, measure per-splat multi-view evidence against every
    /// training view and write it into the final ply as `ev_*` vertex properties, so
    /// `brush-splat-render --confidence` can gate novel views without the dataset.
    /// See docs/splat-confidence.md.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    pub export_evidence: bool,
    /// Before the final export, drop splats whose in-mask contribution fraction
    /// (evidence `w_in / w_all`) is below this value, or that no training view
    /// supported at all. Implies computing evidence. Off when unset.
    #[arg(long, help_heading = "Process options")]
    pub evidence_prune_inmask: Option<f32>,
    /// Weight of the normal-map residual folded into the evidence residual when the
    /// dataset has `normals/`. 0 skips the extra normal render.
    #[arg(long, help_heading = "Process options", default_value = "0.0")]
    pub evidence_normal_weight: f32,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainStreamConfig {
    #[clap(flatten)]
    #[serde(flatten)]
    pub train_config: brush_train::config::TrainConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub model_config: brush_dataset::config::ModelConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub load_config: brush_dataset::config::LoadDatasetConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub rerun_config: brush_rerun::RerunConfig,
}

impl Default for TrainStreamConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
