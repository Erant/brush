# Mixing masked and transparent views in one run

An alpha channel on a training frame can mean two different things:

- **Masked** — *"ignore this region."* The background is real, we just don't
  want to fit it. Pixels outside the mask contribute nothing to the loss, and
  nothing pushes the splats' alpha to zero there.
- **Transparent** — *"nothing is here."* The model should actively learn
  `alpha = 0` outside the subject, so these views carve the silhouette.

Brush resolves this **per view**, so a single training run can carry both. The
mode rides all the way through the pipeline: `LoadImage::alpha_mode` →
`SceneBatch::alpha_mode` → the loss config for that step. Batches are one view
per step, so nothing collapses the mix.

Mixing is useful when part of a capture has a clean matte (an object on a
turntable, a green-screen pass) and part doesn't (handheld footage where only a
rough subject mask is available). The transparent views carve the silhouette;
the masked views abstain there while still contributing interior detail.

## Dataset layout

The `masks/` sidecar convention decides the mode:

| view | resolved mode |
|---|---|
| has a `masks/<name>.*` sidecar | `Masked` |
| alpha embedded in the image, no sidecar | `Transparent` |
| plain RGB, no sidecar | irrelevant — there is no alpha to interpret |

```
dataset/
  cameras.txt
  images.txt
  points3D.txt
  images/
    handheld_00001.jpg     # masked: alpha comes from the sidecar
    turntable_00001.png    # transparent: RGBA, alpha embedded
  masks/
    handheld_00001.png     # only the masked frames get one
```

Detection is automatic and works for the COLMAP, Nerfstudio and RealityCapture
loaders — all three resolve sidecars through the same helper. Mask filenames
are flexible (`masks/img.png`, `masks/img.jpg.png`, `masks/img.mask.png`, and
nested subpaths mirroring the image directory).

On load, brush logs the census:

```
Dataset alpha modes: 42 masked, 17 transparent view(s)
```

## `--alpha-mode` forces every view

`--alpha-mode masked|transparent` is a **global override**, not a per-view
default. Passing it flattens the mix — every view is forced to that mode
regardless of its sidecar. This is deliberate: some pipelines deliver a matte
through `masks/` and then ask for it to be read as transparency.

**To train on a mix, omit the flag.** If you pass it on a dataset where only
some views have sidecars, brush emits a warning naming the counts, so a
flattened mix is never silent.

## The sharp edge

An RGBA image whose alpha is *really* a mask, with no `masks/` sidecar, is
treated as transparency — and transparent-mode GT is **premultiplied at load**,
which destroys the RGB under the mask. If a frame's alpha means "ignore this",
move it into a `masks/` sidecar. There is no way to detect the intent from the
pixels alone.

## `--normalize-masked-loss`

Off by default. The loss kernel multiplies each pixel by `gt.a`, but the trainer
averages over the *whole* frame — so a view whose mask covers 20% of the frame
contributes roughly 0.2x the gradient of a transparent view of the same subject.

In a mask-only run that is a uniform rescale and harmless. In a mixed run it is
a systematic per-view weighting: the masked views quietly count for less. This
flag divides a masked view's loss by its mask coverage, turning the whole-frame
mean into a mean over the masked region:

```
brush-cli <dataset> --normalize-masked-loss
```

It matters most when masks cover a small fraction of the frame. Coverage is
measured once per view on the CPU at load time and cached with the batch, so it
costs one linear scan per view for the whole run — no per-step GPU reduction.
Coverage is floored at 1% when dividing, capping amplification at 100x; a view
masked out entirely produces an all-zero loss map anyway, so the floor only
guards the divide.

The same flag applies mask-weighting to the eval PSNR/SSIM. Without it, a masked
eval view is scored on background the model was never asked to fit, which makes
metrics incomparable across views in a mixed run. This is exact for binary masks
(`a` is 0 or 255, so the `a` weighting survives the mse's squaring) and
approximate for soft ones.

## Notes

- **LPIPS is skipped on masked views.** The LPIPS path has no mask support, so
  the term would supervise exactly the background the photometric loss excludes.
  A per-pixel mask isn't meaningful for VGG anyway — its receptive field spans
  the mask boundary. `--lpips-loss-weight` defaults to `0`, so this is inert
  unless you turn it on.
- **Background compositing is skipped on masked views.** Masked GT is not
  premultiplied, so folding `gt + (1 - a) * bg` into it would compare a
  straight-alpha GT against a premultiplied-over-bg render at partial-mask
  edges. Transparent views still composite as before.
- **Expect a one-off stall per mode.** The `mask` flag in the loss kernel is a
  compile-time constant, so a mixed run JIT-compiles both variants — once each,
  at the first view of each mode.
