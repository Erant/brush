# Integrating splat confidence into b2crunner

Audience: whoever (or whatever) is editing `~/Projects/b2crunner` to consume
the confidence-gated render that `brush-splat-render --confidence` now
produces. It replaces the pipeline's `mask_splat` stage. Read
`docs/splat-confidence.md` in this repo for *what* is measured; this page is
only about the seams in b2crunner.

## 1. What changed in brush (this fork, branch `normal-map-supervision`)

### `brush` (training CLI) — three new `Process options`

| flag | default | effect |
|---|---|---|
| `--export-evidence` | off | after the last step, measure per-splat multi-view evidence against every training view (~2 s for 100k splats × 81 views) and write it into the final ply as seven extra vertex properties `ev_w_in, ev_w_all, ev_err, ev_views, ev_dir_0..2` |
| `--evidence-prune-inmask <f>` | off | before that export, drop splats whose in-mask fraction is below `f` or that no view supported; implies evidence. `0.1`–`0.3` are sane; verify visually before shipping a default |
| `--evidence-normal-weight <w>` | 0 | fold `w · normal residual` into the evidence residual when the dataset has `normals/` (one extra render per view); off by default, untuned |

The `ev_*` properties are ignored by every other ply reader. Only the LOD-0
final export gets them (`--lod-levels 0`, which is the pipeline's case).

### `brush-splat-render` — `--confidence` mode

```
brush-splat-render --splat scene.ply --cameras cameras.json --output-dir out/ \
    --confidence \
    [--cull-color 0.5,0.5,0.5] [--gate-lo 0.45] [--gate-hi 0.65] \
    [--confidence-sidecar] \
    [--dataset <training dataset dir> [--alpha-mode transparent] [--write-evidence with_ev.ply]] \
    [--conf-tau 0.08 --conf-min-views 4 --conf-inmask-lo 0.3 --conf-inmask-hi 0.8 \
     --conf-angle-margin 30 --conf-angle-soft 15 --conf-facing --conf-graze-deg 80]
```

Evidence source, in order: the ply's `ev_*` block → `--dataset` (measure it
now; the dataset options must match training, notably `--alpha-mode`) →
none (warns, every splat fully trusted, confidence degenerates to alpha).
`--alpha-mode` is a global force, so this pipeline's `masks`-as-alpha layout
keeps working unchanged — but a dataset trained on a *mix* of alpha modes (see
[mixed-alpha-modes.md](mixed-alpha-modes.md)) must omit the flag here too, or
the evidence pass sees different ground truth than training did.
Without `--confidence` the binary behaves exactly as before.

### The output contract in confidence mode — **this is the breaking part**

| | before (and still, without `--confidence`) | with `--confidence` |
|---|---|---|
| RGB | `colour·α + background·(1−α)` — premultiplied over `--background` | `(colour·α + cull·(1−α))·g + cull·(1−g)`: composited over the **cull colour**, then blended toward it by the gate |
| alpha | accumulated splat opacity `α` | the **gate** `g = smoothstep(gate_lo, gate_hi, C)`, where `C` is the per-pixel confidence |
| `--background` | used | **ignored**; `--cull-color` is the background |
| sidecar | – | `--confidence-sidecar` writes `<stem>.conf.png`, raw `C` as 8-bit grey |

Consequences:

- Rejected pixels are the cull colour (default 0.5 grey), **not black**.
- A fully transparent pixel is *not* black any more, so anything that asserts
  "premultiplied over black" (`_check_premultiplied` in
  `steps/anchor_stub.py`, used by `composite_splat_views`) must never see a
  confidence render. Keep confidence mode **off** for the face-view renders
  that feed `composite_splat_views` (`fast_helical_shell.yaml` has three
  `render_splat` steps; only the helical re-render wants confidence).
- The alpha is already the decision. Thresholding it at 239/255, dilating,
  and bilateral-filtering — `mask_splat` — is now wrong, not merely
  redundant: it would re-composite the grey frames over black and smear the
  gate's soft edge.

## 2. Changes to make in b2crunner

### `pipeline/steps/brush.py`

Add params and pass them through in `cmd`:

```python
Param("export_evidence", bool, True,
      "Write per-splat multi-view evidence (ev_* vertex properties) into the "
      "final ply so render_splat --confidence needs no dataset"),
Param("evidence_prune_inmask", float, None,
      "Drop splats whose in-mask contribution fraction is below this before the "
      "final export; empty disables", advanced=True),
Param("evidence_normal_weight", float, 0.0,
      "Weight of the normal-map residual in the evidence residual", advanced=True),
...
if params["export_evidence"]:
    cmd.append("--export-evidence")
if params["evidence_prune_inmask"] is not None:
    cmd.extend(["--evidence-prune-inmask", str(params["evidence_prune_inmask"])])
if params["evidence_normal_weight"] > 0:
    cmd.extend(["--evidence-normal-weight", str(params["evidence_normal_weight"])])
```

`export_evidence` defaulting to on is fine for both trainings (`train_splat`
and `train_final_splat`): the cost is seconds and the ply stays readable
everywhere. Leave pruning off by default until it has been looked at on a
real run.

### `pipeline/steps/splat.py` (`render_splat`)

Add params:

```python
Param("confidence", bool, False,
      "Gate the render by per-splat multi-view confidence (replaces mask_splat). "
      "Needs a ply trained with export_evidence, or evidence_dataset"),
Param("cull_color", list, [0.5, 0.5, 0.5],
      "Colour rejected pixels resolve to in confidence mode; also the "
      "compositing background"),
Param("gate_lo", float, 0.45, "Confidence at/below which a pixel is culled"),
Param("gate_hi", float, 0.65, "Confidence at/above which a pixel is kept"),
Param("confidence_sidecar", bool, False,
      "Also keep <stem>.conf.png (raw per-pixel confidence) for tuning", advanced=True),
Param("evidence_dataset", str, None,
      "Training dataset dir to measure evidence against when the ply carries "
      "none (e.g. the export_colmap_intermediate output)", advanced=True),
Param("conf_args", list, [],
      "Extra --conf-* flags passed verbatim to brush-splat-render", advanced=True),
```

In `_rasterize`, when `confidence` is on, replace `--background` with:

```python
cmd += ["--confidence",
        "--cull-color", ",".join(f"{c:.6f}" for c in cull_color),
        "--gate-lo", str(gate_lo), "--gate-hi", str(gate_hi)]
if confidence_sidecar:
    cmd.append("--confidence-sidecar")
if evidence_dataset:
    cmd += ["--dataset", str(evidence_dataset), "--alpha-mode", "transparent"]
cmd += conf_args
```

and when reading the frames back, keep doing what it does (RGB → `images`,
alpha → `masks`): the alpha is now the gate, foreground = 1, which is still
"the splat's per-pixel mask" as far as the field is concerned. Ignore
`*.conf.png` files when collecting outputs (or copy them to the crashlog /
debug dir); they are not frames.

The long "BLACK, not white" comment above `bg_color` describes the old
contract. In confidence mode `bg_color` is unused; say so next to it rather
than deleting the history, since the non-confidence path (face views) still
depends on it.

### `pipeline/steps/mask_splat.py`

Give it a pass-through mode rather than deleting it, so the step ordering
documented in `steps/anchor_stub.py` ("`inject_anchor` must run AFTER
`mask_splat`") keeps holding and A/B against the recorded run stays
possible:

```python
Param("mode", str, "threshold", choices=("threshold", "passthrough"),
      "threshold: the recorded ComfyUI subgraph (alpha cut, dilate, bilateral). "
      "passthrough: frames untouched (already confidence-gated by render_splat), "
      "masks still replaced by the all-1.0 VACE batch"),
```

`passthrough` must still emit `masks = all 1.0`: that batch is the per-frame
VACE flag `denoise_pass2` reads ("synthetic, regenerate"), and `inject_anchor`
downstream writes its 0.0 into it. The images go through unchanged — no
compositing over black, no filter.

Update `tests/test_mask_splat.py` for the new param (the threshold path must
still reproduce `cyber_6f/masked_splatted` byte-for-byte-ish).

### Workflows (`fast_helical_native.yaml`, `fast_helical_full.yaml`, `fast_helical_shell.yaml`)

Stage 2/3 in each:

```yaml
  - id: train_splat
    step: brush
    params:
      export_evidence: true          # (default; spelled out)

  - id: rerender_splat
    step: render_splat
    params:
      confidence: true
      cull_color: [0.5, 0.5, 0.5]
      # bg_color is ignored in confidence mode

  - id: mask_splat_fringes
    step: mask_splat
    params:
      mode: passthrough
```

Do **not** turn `confidence` on for the face-view `render_splat` steps that
feed `composite_splat_views` in `fast_helical_shell.yaml`; they need the
old premultiplied-over-black output. `tests/test_workflows.py` mirrors the
native and full files — keep the edits identical in both.

If a run's ply predates `--export-evidence`, wire
`evidence_dataset: ${globals.output_root}/colmap_intermediate` and turn
`export_colmap_intermediate` on for that run; that dataset is exactly what
`train_splat` saw.

### Denoise pass 2 expectations

`denoise_pass2` now receives a subject cut out of **0.5 grey** instead of
black, with a soft (≈ two-value) edge from the smoothstep gate instead of a
bilateral-filtered hard cut, and no halo (the render is composited over the
same grey it is culled to, so partial coverage fades toward grey, never
toward a contrasting colour). The per-frame VACE mask is unchanged (all 1.0
except the anchor). If the prompt or any negative prompt mentions a black
background, revisit it. The anchor frame (`frame_00038_` in the recorded
run) is still `anchor.png` verbatim at alpha 0 — `inject_anchor` is
untouched.

### Docker image

Both binaries come from this fork's workspace: `cargo build --release -p
brush-cli -p brush-splat-render` (the `brush-builder` stage). A
`brush-splat-render` from an older build has no `--confidence`; a `brush`
from an older build writes no `ev_*`, which the new renderer reports as a
warning and falls back to alpha — so mismatched images degrade loudly, not
silently.

## 3. Verifying the integration

1. `render_splat` output frames: grey outside the subject, alpha 0 there and
   255 inside, RGB never black-premultiplied. `frame_00020_` of a
   cyber2_6f-style run must not show the dark wedge between the legs that
   `cyber2_6f/masked_splatted/frame_00020_` keeps.
2. `mask_splat` in passthrough: `images` byte-identical to its input,
   `masks` all 1.0.
3. `inject_anchor` still after `mask_splat`; the anchor frame at alpha 0.
4. `denoise_pass2` output has no black rims and no grey bleed into the
   subject.
5. Tuning, if needed, via `confidence_sidecar: true` and `conf_args`:
   raise `--conf-tau` if genuine surface is being culled, lower it if
   disagreeing splats survive; `--conf-angle-margin` (default 30°) is the
   lever for how much of the helix's elevation extremes is trusted (15°
   chewed patches out of the jacket at the top of the helix, 30° keeps the
   body and trims only the grazing fringe).

## 4. Known limits

- Evidence view counts use "≥ 1 pixel of in-mask mass at training
  resolution", so they scale with `--max-resolution`; defaults were tuned at
  720–1080 px short side.
- Measured on `circ_colmap` along the cyber2_6f helical path, the gated
  region is slightly less frame-to-frame stable than a raw alpha cut
  (kept-region IoU 0.81 vs 0.85) because it culls more silhouette; the
  decision itself no longer wanders per frame.
- `--conf-facing` and `--evidence-normal-weight` exist but are untuned; leave
  them off in the workflows until someone looks at them.
