#![recursion_limit = "256"]
#![cfg(not(target_family = "wasm"))]

use brush_async::Actor;
use brush_process::DataSource;
use brush_process::RunningProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::create_process;
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;

use clap::{Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::trace_span;

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

/// How often the CLI prints its one-line training diagnostic.
const DIAGNOSTIC_EVERY: Duration = Duration::from_secs(10);

/// Print a line above the progress bars. When stdout isn't a terminal indicatif
/// draws nothing at all, so fall back to a plain println there — the diagnostic
/// is the whole point of a piped or logged run.
fn report(sp: &MultiProgress, line: &str) {
    if sp.is_hidden() {
        println!("{line}");
    } else {
        let _ = sp.println(line);
    }
}

/// Splat counts get big; keep them short in the diagnostic line.
fn format_count(count: u32) -> String {
    if count >= 1_000_000 {
        format!("{:.2}M", f64::from(count) / 1e6)
    } else if count >= 10_000 {
        format!("{:.1}k", f64::from(count) / 1e3)
    } else {
        count.to_string()
    }
}

/// Rolling training state, summarized into one line every [`DIAGNOSTIC_EVERY`].
struct Diagnostics {
    total_iters: u64,
    iter: u32,
    train_elapsed: Duration,
    train_loss: Option<f32>,
    splats: u32,
    eval_views: u32,
    lod_progress: Option<(u32, u32)>,
    last_eval: Option<(u32, f32, f32)>,
    /// Iteration and wall clock at the previous report, for the interval rate.
    last_report: Option<(u32, Instant)>,
}

impl Diagnostics {
    fn new(total_iters: u64) -> Self {
        Self {
            total_iters,
            iter: 0,
            train_elapsed: Duration::from_secs(0),
            train_loss: None,
            splats: 0,
            eval_views: 0,
            lod_progress: None,
            last_eval: None,
            last_report: None,
        }
    }

    /// The diagnostic line, or `None` before the first training step (there's
    /// nothing to say yet, and loading already has its own spinner).
    fn line(&mut self, now: Instant) -> Option<String> {
        if self.iter == 0 {
            return None;
        }

        let mut parts = vec![format!("iter {}/{}", self.iter, self.total_iters)];

        if let Some((lod, total_lods)) = self.lod_progress {
            parts.push(format!("LOD {lod}/{total_lods}"));
        }

        // Rate since the last report (what you'd watch to spot a slowdown),
        // plus the average over the trainer's own accumulated step time.
        let rate = self.last_report.and_then(|(iter, at)| {
            let secs = now.duration_since(at).as_secs_f64();
            (secs > 0.0 && self.iter > iter).then(|| f64::from(self.iter - iter) / secs)
        });
        let avg_secs = self.train_elapsed.as_secs_f64();
        let avg = (avg_secs > 0.0).then(|| f64::from(self.iter) / avg_secs);
        match (rate, avg) {
            (Some(rate), Some(avg)) => parts.push(format!("{rate:.1} it/s (avg {avg:.1})")),
            (Some(rate), None) => parts.push(format!("{rate:.1} it/s")),
            (None, Some(avg)) => parts.push(format!("avg {avg:.1} it/s")),
            (None, None) => {}
        }

        if let Some(loss) = self.train_loss {
            parts.push(format!("loss {loss:.5}"));
        }

        if self.splats > 0 {
            parts.push(format!("{} splats", format_count(self.splats)));
        }

        match self.last_eval {
            Some((iter, psnr, ssim)) => {
                parts.push(format!("eval@{iter} {psnr:.2} PSNR / {ssim:.3} SSIM"));
            }
            None if self.eval_views > 0 => parts.push("eval pending".to_owned()),
            None => {}
        }

        parts.push(format!(
            "{} elapsed",
            humantime::format_duration(Duration::from_secs(self.train_elapsed.as_secs()))
        ));

        // ETA off the interval rate rather than the average, so it tracks the
        // speed the run is actually going at now.
        if let Some(rate) = rate
            && self.total_iters > u64::from(self.iter)
        {
            let left = (self.total_iters - u64::from(self.iter)) as f64 / rate;
            parts.push(format!(
                "~{} left",
                humantime::format_duration(Duration::from_secs(left as u64))
            ));
        }

        self.last_report = Some((self.iter, now));
        Some(format!("📊 {}", parts.join(" · ")))
    }
}

/// Build the training process described by `args`, or `None` if no source was
/// given. Shared by the standalone CLI binary and brush-app's headless path.
pub fn build_process(args: &Cli) -> Option<RunningProcess> {
    let source = args.source.clone()?;
    let cli_config = args.train_stream.clone();
    Some(create_process(source, async move |init| {
        Some(brush_process::args_file::merge_configs(&init, &cli_config))
    }))
}

/// Initialize the backend, then drive `process` to completion on the CLI UI.
pub async fn run_headless(
    process: RunningProcess,
    train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    brush_process::burn_init_setup().await;
    run_cli_ui(process, train_stream_config).await
}

/// Run the CLI: pin the trainer stream to a dedicated [`Actor`] thread,
/// drive the indicatif UI on the main task.
pub async fn run_cli_ui(
    mut process: RunningProcess,
    #[allow(unused)] train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    // Pump the trainer stream from a dedicated Actor thread; the
    // indicatif UI loop below consumes its output on the main task.
    let (tx, mut messages) = mpsc::unbounded_channel();
    let trainer = Actor::new("cli-trainer");
    trainer
        .run(move || async move {
            while let Some(msg) = process.stream.next().await {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        })
        .detach();

    // Hold the actor for the lifetime of the UI loop; dropping it
    // would kill the pump.
    let _trainer = trainer;

    // Initialize the logger with indicatif integration to prevent
    // progress bars from clobbering log output.
    let sp = {
        let mut builder = env_logger::builder();
        builder.target(env_logger::Target::Stdout);
        let logger = builder.build();
        let level = logger.filter();
        let multi = MultiProgress::new();

        LogWrapper::new(multi.clone(), logger)
            .try_init()
            .expect("Failed to initialize logger");
        log::set_max_level(level);

        multi
    };

    let main_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indacitif config")
            .tick_strings(&[
                "🖌️      ",
                "█🖌️     ",
                "▓█🖌️    ",
                "░▓█🖌️   ",
                "•░▓█🖌️  ",
                "·•░▓█🖌️ ",
                " ·•░▓🖌️ ",
                "  ·•░🖌️ ",
                "   ·•🖌️ ",
                "    ·🖌️ ",
                "     🖌️ ",
                "    🖌️ █",
                "   🖌️ █▓",
                "  🖌️ █▓░",
                " 🖌️ █▓░•",
                "🖌️ █▓░•·",
                "🖌️ ▓░•· ",
                "🖌️ ░•·  ",
                "🖌️ •·   ",
                "🖌️ ·    ",
                "🖌️      ",
            ]),
    );

    let stats_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["ℹ️", "ℹ️"]),
    );

    let train_progress = {
        let tc = &train_stream_config.train_config;
        let bar = ProgressBar::new(tc.total_iters() as u64)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({per_sec}, {eta} remaining)",
            )
            .expect("Invalid indicatif config").progress_chars("◍○○"),
        )
        .with_message("Steps");
        sp.add(bar)
    };

    let main_spinner = sp.add(main_spinner);
    main_spinner.enable_steady_tick(Duration::from_millis(120));

    let eval_spinner = sp.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .expect("Invalid indicatif config")
                .tick_strings(&["✅", "✅"]),
        ),
    );

    eval_spinner.set_message("waiting for dataset...");

    let stats_spinner = sp.add(stats_spinner);
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        report(
            &sp,
            "ℹ️  running in debug mode, compile with --release for best performance",
        );
    }

    let mut diag = Diagnostics::new(train_stream_config.train_config.total_iters() as u64);

    // Fires the periodic diagnostic even while the trainer is quiet (loading,
    // a long refine), so a stalled run is visible too.
    let mut diag_tick = tokio::time::interval(DIAGNOSTIC_EVERY);
    diag_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    diag_tick.tick().await; // The first tick resolves immediately.

    loop {
        let msg = tokio::select! {
            msg = messages.recv() => match msg {
                Some(msg) => msg,
                None => break,
            },
            _ = diag_tick.tick() => {
                if let Some(line) = diag.line(Instant::now()) {
                    report(&sp, &line);
                }
                continue;
            }
        };

        let _span = trace_span!("CLI UI").entered();

        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                report(&sp, "❌ Encountered an error");
                return Err(error);
            }
        };

        match msg {
            ProcessMessage::NewProcess => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { name, training, .. } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    report(
                        &sp,
                        "❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)",
                    );
                    break;
                }
                main_spinner.set_message(format!("Loading {name}..."));
            }
            ProcessMessage::SplatsUpdated { num_splats, .. } => {
                diag.splats = num_splats;
            }
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainConfig { .. } => {}
                TrainMessage::Dataset { dataset } => {
                    let train_views = dataset.train.views.len();
                    let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                    log::info!(
                        "Loaded dataset with {train_views} training, {eval_views} eval views",
                    );
                    main_spinner.set_message(format!(
                        "Loading dataset with {train_views} training, {eval_views} eval views",
                    ));
                    diag.eval_views = eval_views as u32;
                    if eval_views > 0 {
                        eval_spinner.set_message(format!(
                            "evaluating {} views every {} steps",
                            eval_views, train_stream_config.process_config.eval_every,
                        ));
                    } else {
                        eval_spinner.finish_and_clear();
                    }
                }
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    lod_progress,
                    train_loss,
                } => {
                    if let Some((lod, total_lods)) = lod_progress {
                        main_spinner.set_message(format!("LOD {lod}/{total_lods}"));
                    } else {
                        main_spinner.set_message("Training");
                    }
                    train_progress.set_position(iter as u64);
                    diag.iter = iter;
                    diag.train_elapsed = total_elapsed;
                    diag.lod_progress = lod_progress;
                    diag.train_loss = train_loss;
                }
                TrainMessage::RefineStep {
                    cur_splat_count,
                    iter,
                } => {
                    stats_spinner.set_message(format!("Current splat count {cur_splat_count}"));
                    log::info!("Refine iter {iter}, {cur_splat_count} splats.");
                    diag.splats = cur_splat_count;
                }
                TrainMessage::EvalResult {
                    iter,
                    avg_psnr,
                    avg_ssim,
                } => {
                    log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");

                    eval_spinner.set_message(format!(
                        "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                    ));
                    diag.last_eval = Some((iter, avg_psnr, avg_ssim));
                }
                TrainMessage::DoneTraining => {}
            },
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
                main_spinner.set_message("Completed loading");
                stats_spinner.set_message("Completed loading");
            }
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
                report(&sp, &format!("⚠️: {error}"));
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    // A final diagnostic, so a finished run leaves its numbers behind even if
    // it ended between ticks.
    if let Some(line) = diag.line(Instant::now()) {
        report(&sp, &line);
    }

    let duration_secs = Duration::from_secs(diag.train_elapsed.as_secs());
    report(
        &sp,
        &format!(
            "Training took {}",
            humantime::format_duration(duration_secs)
        ),
    );

    log::info!(
        "Done training! Took {:?}.",
        humantime::format_duration(duration_secs)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn formats_counts() {
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(9_999), "9999");
        assert_eq!(format_count(45_678), "45.7k");
        assert_eq!(format_count(2_500_000), "2.50M");
    }

    #[test]
    fn diagnostics_stay_quiet_before_the_first_step() {
        let mut diag = Diagnostics::new(1000);
        assert!(diag.line(Instant::now()).is_none());
    }

    #[test]
    fn diagnostics_report_rate_psnr_and_eta() {
        let mut diag = Diagnostics::new(1000);
        diag.eval_views = 5;
        diag.iter = 100;
        diag.train_elapsed = Duration::from_secs(2);
        diag.splats = 45_678;

        // No previous report yet, so only the average rate is known.
        let start = Instant::now();
        let first = diag.line(start).expect("a step has landed");
        assert!(first.contains("iter 100/1000"), "{first}");
        assert!(first.contains("avg 50.0 it/s"), "{first}");
        assert!(first.contains("45.7k splats"), "{first}");
        assert!(first.contains("eval pending"), "{first}");
        // Nothing to extrapolate an ETA from on the first line.
        assert!(!first.contains("left"), "{first}");

        diag.iter = 300;
        diag.train_elapsed = Duration::from_secs(6);
        diag.train_loss = Some(0.0421);
        diag.last_eval = Some((200, 24.5, 0.8123));

        let second = diag
            .line(start + Duration::from_secs(10))
            .expect("still training");
        // 200 iters in 10s of wall clock, 300 iters in 6s of trainer time.
        assert!(second.contains("20.0 it/s (avg 50.0)"), "{second}");
        assert!(second.contains("loss 0.04210"), "{second}");
        assert!(
            second.contains("eval@200 24.50 PSNR / 0.812 SSIM"),
            "{second}"
        );
        // 700 iters left at the interval rate.
        assert!(second.contains("~35s left"), "{second}");
    }

    #[test]
    fn parses_source_and_overrides() {
        let cli = Cli::try_parse_from([
            "brush-cli",
            "some/dataset/path",
            "--total-train-iters",
            "50",
            "--eval-split-every",
            "2",
            "--max-resolution",
            "512",
            "--sh-degree",
            "2",
            "--seed",
            "7",
        ])
        .unwrap();

        assert!(matches!(
            &cli.source,
            Some(DataSource::Path(p)) if p == "some/dataset/path"
        ));
        // Passing a source flips the viewer default off.
        assert!(!cli.with_viewer);

        let ts = &cli.train_stream;
        assert_eq!(ts.train_config.total_train_iters, 50);
        assert_eq!(ts.train_config.total_iters(), 50); // No LOD levels by default.
        assert_eq!(ts.load_config.eval_split_every, Some(2));
        assert_eq!(ts.load_config.max_resolution, 512);
        assert_eq!(ts.model_config.sh_degree, 2);
        assert_eq!(ts.process_config.seed, 7);

        // A source without a viewer is a valid combination.
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn parses_url_source() {
        let cli = Cli::try_parse_from(["brush-cli", "https://example.com/data.zip"]).unwrap();
        assert!(matches!(
            &cli.source,
            Some(DataSource::Url(u)) if u == "https://example.com/data.zip"
        ));
    }

    #[test]
    fn defaults_to_viewer_without_source() {
        let cli = Cli::try_parse_from(["brush-cli"]).unwrap();
        assert!(cli.source.is_none());
        assert!(cli.with_viewer);
        // Viewer without a source is valid (brush-app's default mode).
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn viewer_flag_with_source() {
        let cli = Cli::try_parse_from(["brush-cli", "some/path", "--with-viewer"]).unwrap();
        assert!(cli.with_viewer);
        assert!(cli.source.is_some());
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn validate_rejects_headless_without_source() {
        let mut cli = Cli::try_parse_from(["brush-cli"]).unwrap();
        cli.with_viewer = false;
        let Err(err) = cli.validate() else {
            panic!("expected validation error")
        };
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn rejects_unknown_flag() {
        assert!(Cli::try_parse_from(["brush-cli", "--not-a-real-flag"]).is_err());
    }
}
