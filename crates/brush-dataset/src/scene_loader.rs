use std::sync::Arc;

use brush_async::Actor;
use brush_render::AlphaMode;
use rand::{SeedableRng, seq::SliceRandom};
use tokio::sync::{Mutex, mpsc};

use crate::{
    config::LoadDatasetConfig,
    scene::{
        Scene, SceneBatch, SceneView, mean_alpha, normal_sample_to_data, sample_to_packed_data,
        view_to_sample_image,
    },
};

/// Shared cache of GPU-ready scene batches. Each slot holds at most one
/// batch; once the running total passes `budget_bytes`, new batches bypass
/// the cache and just get re-decoded + re-packed on every visit.
///
/// Caching the packed batch (instead of the decoded `DynamicImage`) skips
/// the per-hit decode → premultiply → repack work: a cache hit is now a
/// single copy of the already-packed `[H, W]` u32 buffer.
struct BatchCache {
    slots: Vec<Option<Arc<SceneBatch>>>,
    used_bytes: u64,
    budget_bytes: u64,
}

impl BatchCache {
    fn new(n_views: usize, budget_bytes: u64) -> Self {
        Self {
            slots: vec![None; n_views],
            used_bytes: 0,
            budget_bytes,
        }
    }

    fn get(&self, index: usize) -> Option<Arc<SceneBatch>> {
        self.slots[index].clone()
    }

    fn insert(&mut self, index: usize, batch: Arc<SceneBatch>) {
        if self.slots[index].is_some() {
            return;
        }
        // Track exact bytes: rounding to whole MB let sub-MB images slip in
        // for free and bypass the budget entirely.
        let normal_bytes = batch.normal_data.as_ref().map_or(0, |d| d.as_bytes().len());
        let size_bytes: u64 = (batch.img_packed.as_bytes().len() + normal_bytes)
            .try_into()
            .expect("shouldn't exceed ~18 Exabytes...");
        if self.used_bytes + size_bytes < self.budget_bytes {
            self.slots[index] = Some(batch);
            self.used_bytes += size_bytes;
        }
    }
}

pub struct SceneLoader {
    rx: mpsc::Receiver<SceneBatch>,
    // Owns the loader actor threads. Dropping cancels them; their
    // senders then drop, the channel closes, and `next_batch` returns.
    _actors: Vec<Actor>,
}

impl SceneLoader {
    pub fn new(scene: &Scene, seed: u64, config: &LoadDatasetConfig) -> Self {
        // Prefetch buffer: at most 4 batches ahead of the trainer.
        // Two tasks per actor share this buffer so one task's I/O can
        // overlap with the other's decode + GPU upload.
        let (tx, rx) = mpsc::channel(4);

        // Fan out only as many loaders as we have real parallelism.
        // Wasm shares one JS event loop, so extra actors just add
        // contention without overlapping I/O.
        let n_actors = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism().map_or(8, |p| p.get())
        };
        const TASKS_PER_ACTOR: usize = 2;

        let views = scene.views.clone();
        let cache = Arc::new(Mutex::new(BatchCache::new(
            views.len(),
            config.max_scene_batch_cache_size,
        )));

        let mut task_idx: u64 = 0;
        let actors: Vec<Actor> = (0..n_actors)
            .map(|i| {
                let actor = Actor::new(&format!("dataloader-{i}"));
                for _ in 0..TASKS_PER_ACTOR {
                    let views = views.clone();
                    let cache = cache.clone();
                    let tx = tx.clone();
                    let task_seed = seed.wrapping_add(task_idx);
                    task_idx += 1;
                    actor
                        .run(move || run_loader(views, cache, tx, task_seed))
                        .detach();
                }
                actor
            })
            .collect();

        Self {
            rx,
            _actors: actors,
        }
    }

    pub async fn next_batch(&mut self) -> SceneBatch {
        self.rx
            .recv()
            .await
            .expect("Scene loader channel closed unexpectedly")
    }
}

/// Load one view's ground truth into a GPU-ready [`SceneBatch`]: decode and
/// resize the frame, premultiply it in transparent alpha mode, pack RGBA into
/// one `u32` per pixel, and pick up the sibling normal map when there is one.
/// Shared by the training loader and by any pass that walks the dataset
/// view-by-view outside training (e.g. the end-of-training evidence pass).
pub async fn load_view_batch(view: &SceneView) -> image::ImageResult<SceneBatch> {
    let raw = view.image.load().await?;
    let alpha_mode = view.image.alpha_mode();
    let sample = view_to_sample_image(raw, alpha_mode);
    let (target_w, target_h) = (sample.width(), sample.height());
    // Measure coverage before `sample_to_packed_data` consumes the sample.
    // Only masked views need it; a transparent view's alpha is supervised
    // directly, and an opaque one has nothing to correct for.
    let alpha_coverage = (alpha_mode == AlphaMode::Masked && sample.color().has_alpha())
        .then(|| mean_alpha(&sample));
    let (img_packed, has_alpha) = sample_to_packed_data(sample);
    let normal_data = match view.image.load_normal(target_w, target_h).await {
        Some(Ok(img)) => Some(normal_sample_to_data(img)),
        Some(Err(e)) => {
            // A corrupt normal map degrades that view to photometric-only
            // supervision rather than killing the loader task (and with it
            // the whole training run).
            log::warn!(
                "Failed to load normal map for {}: {e}",
                view.image.path().display()
            );
            None
        }
        None => None,
    };
    Ok(SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode,
        alpha_coverage,
        camera: view.camera,
        normal_data,
    })
}

async fn run_loader(
    views: Arc<Vec<crate::scene::SceneView>>,
    cache: Arc<Mutex<BatchCache>>,
    tx: mpsc::Sender<SceneBatch>,
    seed: u64,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut shuffled: Vec<usize> = Vec::new();

    loop {
        if shuffled.is_empty() {
            shuffled = (0..views.len()).collect();
            shuffled.shuffle(&mut rng);
        }
        let index = shuffled.pop().expect("Need at least one view in dataset");
        let view = &views[index];

        let batch = if let Some(batch) = cache.lock().await.get(index) {
            batch
        } else {
            let batch = Arc::new(
                load_view_batch(view)
                    .await
                    .expect("Scene loader failed to load an image"),
            );
            cache.lock().await.insert(index, batch.clone());
            batch
        };

        // The channel takes an owned batch; clone the packed buffer out of
        // the shared cache entry.
        if tx.send(batch.as_ref().clone()).await.is_err() {
            break;
        }
        brush_async::yield_now().await;
    }
}
