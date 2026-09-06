# Spatial reinforcement — the black render + `mask_splat` pair

Between training a splat and handing its re-render to the second denoise
pass, the pipeline throws away every pixel the splat is not confident
about and paints what is left onto pure black. Two steps do it:
`render_splat` (with `bg_color: [0, 0, 0]`) and `mask_splat`. Neither is
clever, and the decision they make together is a per-pixel, per-frame,
2-D one about a 3-D object — which is why this is written down. The
right home for it is almost certainly the splatting/rasterisation side,
where per-Gaussian opacity and coverage are still available; this
document is what has to be reproduced before that move.

## Where it sits

```
denoise_pass1 -> colmap_export -> brush (train)
              -> rerender_splat   (render_splat, helical, 81 frames, BLACK bg)
              -> mask_splat_fringes   (filter_size 6, dilation 2)
              -> reinject_anchor
              -> denoise_pass2   (strength 0.8)
```

Identical in all three workflow files (`fast_helical_native.yaml`,
`fast_helical_full.yaml`, `fast_helical_shell.yaml`); the native and full
files' copies are kept in sync by `tests/test_workflows.py`.

The stage is a native port of the ComfyUI `mask_splat.json` subgraph
(eight generic nodes — `ToBinaryMask`, `InvertMask`, `ImpactDilateMask`,
`AILab_ImageCombiner`, `BilateralFilterImage`, save), collapsed into one
step in `pipeline/steps/mask_splat.py`.

## What actually happens

### 1. Render on black

`render_splat` shells out to `brush-splat-render`, which writes

```
rgb = colour*alpha + bg*(1 - alpha)
```

With `bg = 0` the RGB it writes is **premultiplied by alpha**, and the
alpha channel comes back as the frame's per-pixel mask
(`dataset.masks`, foreground = 1). The step's `bg_color` default is
already `[0.0, 0.0, 0.0]`.

Black is not cosmetic. A splat render has soft, low-alpha fringes
wherever the Gaussians are uncertain — thin hair, silhouette edges,
anything under-observed. On a white or grey background those fringes are
*bright*, and the bilateral filter in step 2 smears them across the
silhouette before anything blacks them out, so a halo survives into the
frames `denoise_pass2` sees. On black, fringe and background are the same
colour and there is nothing to bleed. The recorded ComfyUI run rendered
this stage on 127 grey with alpha 0 in the background
(`cyber_6f/splatted`) and its masked output is black RGB at alpha 255
everywhere (`cyber_6f/masked_splatted`) — black matches where that
pipeline *ends up*, which is what matters; matching its intermediate grey
would reintroduce the same halo, just dimmer.

(The same premultiplication assumption is relied on, and enforced, by
`composite_splat_views` in `steps/anchor_stub.py`, which refuses a render
that is more than 8/255 bright where it is fully transparent.)

### 2. `mask_splat`

Per frame, on the render's own alpha:

| | |
|---|---|
| **threshold** | `keep = alpha >= 1 - threshold/255`, i.e. `>= 239/255` at the default `threshold: 16`. Strictly greater-than, no rounding to 0-255 first. |
| **dilate** | grow `keep` with a plain `dilation x dilation` kernel of ones (2x2 at `dilation: 2`) — not `2*dilation+1`, not an ellipse. `dilation: 0` is valid and skipped. |
| **composite** | `rgb * keep` — everything outside the kept region becomes exactly 0. |
| **bilateral filter** | `cv2.bilateralFilter(d=filter_size, sigmaColor=0.5*255, sigmaSpace=100)` over the composited frame. |
| **masks out** | replaced with an all-1.0 batch. |

The threshold and dilation semantics were fitted against recorded output,
not read off the node source, and neither is guessable: rounding the mask
to 0-255 before comparing pushed the max error from 15 to 140, and a
`2*dilation+1` kernel pushed it to 200.

`filter_size`/`dilation` is 6/2 in all shipping workflows (`fast
helical`); the older `helical` and `tiered` pipelines used 12/4 and 4/0.

### 3. The output mask means something else entirely

`dataset.masks` goes into `mask_splat` as the splat's **per-pixel alpha**
and comes out as an all-1.0 batch, which downstream is the **per-frame**
VACE flag: 1.0 = "synthetic, denoise this frame", 0.0 = "a real
photograph, keep it". Two different kinds of mask share the field, and
this step is where the meaning changes. Consequences:

- The blacked-out region is *not* protected. `denoise_pass2` is told to
  regenerate every one of those frames in full, background included.
- `reinject_anchor` must run **after** `mask_splat`, never before. Before
  it, it would overwrite `dataset.masks` — the splat alpha — with its own
  all-1.0 batch, and every step above would silently become a no-op. The
  anchor frame itself is not this step's business: in the recorded run
  `frame_00038_` is the real photo verbatim at alpha 0, neither
  composited nor filtered.

## The end result

Measured on `cyber_6f/splatted/frame_00020_`:

- 27.2% of the frame carries non-zero splat alpha. Nothing in the frame
  ever reaches alpha 1.0 — the maximum is 0.996, so the 239/255 cutoff is
  close to the top of the actual range, and small changes to `threshold`
  move a lot of pixels.
- 23.5% survives the threshold; dilation grows that to 23.8%.
- So **~13.6% of everything the splat drew is discarded as fringe** (3.7%
  of the frame), and the rest of the frame — about three quarters — is
  hard black.
- The recorded ComfyUI frame is 24.9% non-black, the difference being the
  bilateral filter lifting boundary pixels a value or two off zero.

The port is verified against that recorded stage
(`tests/test_mask_splat.py`): the surviving pixel set agrees on ~99.85%
of pixels with every disagreement a near-black boundary value of 1-3, and
the filtered values differ by a mean of ~0.25/255 with a max of 15,
consistent with a difference in the bilateral filter's border handling.
Visually identical, not bit-exact.

What `denoise_pass2` therefore receives is a hard-edged subject cut out
of black, with no soft matte and no halo, plus a per-frame instruction to
regenerate the whole thing — so it re-invents the background from the
prompt rather than inheriting the splat's guesses about it. Those frames
then go through the upscale and into `train_final_splat`.

## Why it is crude

Worth naming, since fixing it is the point of writing this down:

- **The decision is 2-D and per-frame.** A Gaussian that is well
  observed from one direction and grazing from another is judged
  independently in each frame, so the cut wanders between neighbouring
  views of a smooth 81-frame orbit. Nothing enforces temporal or 3-D
  consistency.
- **Confidence is proxied by rendered alpha.** Accumulated opacity along
  a ray is not the same quantity as "the training views constrained
  this", and the rasteriser is the only thing that ever sees the real
  per-Gaussian evidence.
- **The threshold is brittle by construction.** Alpha tops out at 0.996;
  a cutoff at 0.937 sits inside the noise band rather than safely below
  it.
- **Dilation is a fudge for the threshold being too aggressive** — grow
  back by two pixels what a hard cut just removed, with no relation to
  where the error actually is.
- **The bilateral filter runs on already-composited black**, so it drags
  edge pixels toward the background it was meant to protect them from,
  and is the entire source of the port's residual disagreement with the
  recorded run.
- **The per-pixel alpha is destroyed on the way out**, overwritten by the
  per-frame VACE flag, so nothing downstream can reconsider the decision
  or use a soft version of it.

A fix in the splatting step would decide coverage once, in 3-D, from
per-Gaussian opacity and observation counts, and hand out a soft
confidence channel the denoiser could actually be conditioned on —
replacing the threshold, the dilation and the filter at the same time.

That fix now exists: `brush-splat-render --confidence`, backed by per-splat
evidence measured against the training views (`brush --export-evidence`).
See `docs/splat-confidence.md` for what it measures, the flags, and the new
output contract (culled pixels resolve to a selectable colour, alpha is the
gate).
