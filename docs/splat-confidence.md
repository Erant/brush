# Splat confidence: per-splat multi-view evidence, rendered as a channel

`brush-splat-render --confidence` replaces the body2colmap pipeline's
`mask_splat` stage (alpha ≥ 239/255, dilate 2 px, bilateral filter — see
`spatial-reinforcement.md`) with a decision made once, in 3-D, per Gaussian:
*how much did the training views actually constrain this splat, and from
where?* That per-splat **evidence** is measured against the training set,
turned into a per-view **confidence** in `[0, 1]`, alpha-composited through
the rasterizer's feature channel into a per-pixel confidence map, and gated
once. No threshold on rendered alpha, no dilation, no filter.

It works in both alpha modes. In `transparent` mode rendered alpha is at
least correlated with confidence; in `masked` mode it is meaningless (nothing
ever penalises a splat drawing over background), which is exactly why the
old stage could not be made to work there and this one can.

## Evidence

For every training view, each splat collects three numbers, where
`vis_ip = α_ip · T_ip` is the splat's compositing weight at pixel `p`,
`m_p` the GT alpha (mask or matte), and `res_p` the mean-RGB L1 residual
between the render (on black) and the GT:

| lane | meaning |
|---|---|
| `w_in  = Σ_p vis·m`     | contribution mass landing inside GT foreground |
| `e     = Σ_p vis·m·res` | mass-weighted residual where the splat is used |
| `w_all = Σ_p vis`       | total contribution mass |

Summed over all views, plus:

- `views` — number of views giving the splat ≥ 1 pixel of in-mask mass
  (`VIEW_MIN_MASS`, in pixel-weight units at training resolution);
- `dir` — `Σ_views w_in · normalize(camera − mean)`, the unnormalised
  resultant of the directions the splat was observed from.

Seven floats per splat (`brush_train::evidence::SplatEvidence`).

### How it is gathered — no new kernel

The rasterizer's optional `[N, 3]` per-splat feature channel is composited
with exactly the colour blend weights, and its backward is ungated:
`∂/∂feat_i[k] Σ_p F_p·V_p = Σ_p vis_ip · V_p[k]` for any constant per-pixel
weight map `V`. Rendering a *zero* feature and backpropagating
`Σ_p F_p · V_p` with `V_p = (m_p, m_p·res_p, 1)` therefore hands back the
three lanes above as the feature gradient — exactly, not approximately
(`crates/brush-bench-test/tests/evidence.rs` pins `Σ_i w_all_i = Σ_p α_p`).
`m` and `res` come from one `image_loss_eval` call with a 4-channel prediction
whose alpha lane is zeroed, so the loss map's alpha lane is `|0 − gt.a| =
gt.a`. `--evidence-normal-weight w` adds `w · (L1 + 1 − cos)` of the rendered
pseudo-normal against a `normals/` GT map into `res` (one extra forward
render per view).

Cost: one render + backward per training view, a few seconds for 81 views.

### Where it lives

- **At the end of training**: `brush ... --export-evidence` measures evidence
  against every training view after the last step (refine never runs on the
  last step, so indices line up with the export) and writes it into the final
  ply as extra vertex properties
  `ev_w_in, ev_w_all, ev_err, ev_views, ev_dir_0, ev_dir_1, ev_dir_2`
  (`brush_serde::EVIDENCE_FIELDS`). Ordinary ply readers ignore them; brush's
  importer picks them up as `SplatData::evidence`. Only for the LOD-0 final
  export (`lod_levels == 0`).
- **`--evidence-prune-inmask f`** (off by default): before that export, drop
  splats whose in-mask fraction `w_in / w_all` is below `f`, or that no view
  supported at all. This removes fringe Gaussians at the source, so even the
  plain alpha render is cleaner. Conservative values (`0.1`–`0.3`) are the
  place to start; check visually.
- **In `brush-splat-render`**: `--confidence` uses the ply's `ev_*` block if
  present, else measures it against `--dataset <dir>` (the training dataset,
  loaded with the same dataset options — notably `--alpha-mode`; add
  `--write-evidence out.ply` to save the block for later runs), else warns and
  treats every splat as fully trusted (confidence degenerates to alpha).
  "The same options" includes *omitting* `--alpha-mode` when training did: it
  forces every view, so passing it against a dataset trained on a mix of alpha
  modes (see [mixed-alpha-modes.md](mixed-alpha-modes.md)) re-reads the masked
  views as transparent and measures evidence against differently premultiplied
  ground truth than training saw.

## Confidence

For a novel camera at `c`, per splat with mean `μ_i`, `v = normalize(c − μ_i)`,
`μ = dir/|dir|`, `κ = |dir| / w_in`:

| term | formula | catches |
|---|---|---|
| `inmask`   | `smoothstep(lo, hi, w_in / w_all)` | splats drawing over GT background: silhouette fringe, masked-mode junk |
| `agree`    | `exp(−(e / w_in) / τ)` | splats the views disagree on (the dark wedges between limbs) |
| `support`  | `smoothstep(0, n_min, views)` | one-/two-view floaters |
| `coverage` | `1 − smoothstep(φ + m₀, φ + m₁, ∠(v, μ))`, `φ = acos(2κ − 1)` | novel views outside the cone the splat was seen from (a helical orbit's elevations vs. a flat training orbit) |
| `facing`   | optional: `smoothstep(0, cos g, n_i · v)` for disc-like splats only, `n_i` the shortest axis oriented toward `μ` | disc splats seen grazing or from behind |

`c_i(v) = inmask · agree · support · coverage · facing`. Everything but
`coverage` and `facing` is view-independent and computed once
(`ConfidenceModel::new`); per camera it is a handful of `[N]` tensor ops.
`φ` is the half-angle of a spherical cap with resultant length `κ` — exact for
a cap, slightly under for a pure arc, which the margin absorbs.

| flag | default | |
|---|---|---|
| `--conf-tau` | 0.08 | residual scale of `agree` (mean L1 in 0..1) |
| `--conf-min-views` | 4 | views for full `support` |
| `--conf-inmask-lo/hi` | 0.3 / 0.8 | `inmask` fade |
| `--conf-angle-margin` | 30° | `m₀`: fully covered this far past the observed cone |
| `--conf-angle-soft` | 15° | `m₁ − m₀`: fade width |
| `--conf-facing` | off | enable the facing term |
| `--conf-graze-deg` | 80° | angle from the normal at which `facing` reaches 0 |

Because the per-splat confidence varies smoothly with view direction, the
per-pixel map has no per-frame decision to wander: a splat that is trusted
in one frame is trusted in the next unless the view direction moved it out
of its observed cone. The kept *region* still changes with the camera —
it culls more silhouette than an alpha threshold does, so it has more
boundary to move. Measured on `circ_colmap` along the cyber2_6f helical
path (81 frames, defaults): frame-to-frame IoU of the kept region 0.81
(min 0.72) versus 0.85 (min 0.79) for a plain `alpha ≥ 239/255` cut on the
same renders, with 6.7% of the alpha silhouette culled on average (15% at
the elevation extremes).

## Rendering and the output contract

`[c_i, 0, 0]` is the per-splat feature; the composited channel
`C_p = Σ_i T·α·c_i` is the **absolute confidence**: zero where nothing
trusted is drawn, low where only fringe or junk is drawn *even if alpha ≈ 1*.

The gate is `g = smoothstep(gate_lo, gate_hi, C_p)` (`--gate-lo 0.45
--gate-hi 0.65`; equal values give a hard cut). Because fringe splats are
already suppressed inside `C_p`, the gate sits near 0.5 instead of 0.94 and
needs neither dilation nor a filter. The written image is

```
RGB = (splat over cull) · g + cull · (1 − g)
A   = g
```

with `cull = --cull-color` (default `0.5,0.5,0.5`): rejected pixels resolve to
a selectable colour rather than black, and the render is composited over
that same colour so partially covered pixels fade toward it too. **This
deliberately replaces the old "premultiplied over black, alpha = coverage"
contract** — the alpha channel is now the gate. `--confidence-sidecar`
additionally writes `<stem>.conf.<format>`, the raw `C_p` as 8-bit grey, for
tuning or for downstream steps that want the soft value.

## Pipeline integration (b2crunner, separate repo)

- `steps/brush.py`: pass `--export-evidence` (optionally
  `--evidence-prune-inmask`).
- `steps/splat.py` (`render_splat`): pass `--confidence --cull-color ...
  --gate-lo/--gate-hi`; drop `bg_color`'s load-bearing black.
- `mask_splat` becomes a pass-through (keep it wired for A/B).
- `composite_splat_views`'s premultiplied-over-black assertion no longer
  applies to confidence output.

## Tuning notes

`VIEW_MIN_MASS` (1 pixel of mass) makes `views` resolution-dependent; the
defaults above assume training near 720–1080 px on the short side. Start
tuning with `--confidence-sidecar` on a frame where the old stage struggled
(e.g. `cyber2_6f/splatted/frame_00020_`, the wedge between the legs) and
raise `--conf-tau` if genuine surface is being culled, lower it if
disagreeing splats survive. The coverage margin is the other lever: on a
flat circular training orbit re-rendered along a ±30° helix, 15° culled
15–25% of the silhouette at the elevation extremes (patches out of the
jacket and face), 30° keeps the body intact and trims only the grazing
hair/silhouette fringe — hence the default.
