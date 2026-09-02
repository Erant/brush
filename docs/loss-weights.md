# Per-pixel loss weights: the `weights/` sidecar

A view's alpha channel already weights its loss, but only in one of the two
alpha modes, and only as a side effect of what the alpha *means*:

- **Masked** — the loss map is multiplied by `gt.a`, so a soft mask is also
  a soft weight. But the mask is the view's statement of *where the subject
  is*, and lowering it somewhere also stops that region carving anything.
- **Transparent** — the alpha is a *target*: the model learns `alpha = 0`
  outside the subject. There is no weight channel at all; every pixel of the
  frame counts in full.

A `weights/` sidecar is the missing channel: a greyscale image, `0..255`
read as `[0, 1]`, that multiplies the view's loss map pixel by pixel on top
of whatever its alpha mode does. It never changes what the view *says* —
the silhouette a transparent view carves, the region a masked view fits —
only how loudly it says it there.

## Layout

```
dataset/
  images/
    frame_00001.png       # RGBA, transparent: still carves the silhouette
    support_00001.png     # RGB + masks/ sidecar: masked
  masks/
    support_00001.png
  weights/
    frame_00001.png       # greyscale; 255 = full weight, 0 = silent
```

Matching follows the `normals/` convention: `weights/<name>.*` or
`weights/<stem>.*`, with the directory subpath after `weights/` mirroring
the image's own. A view without one trains at weight 1. The map is resized
to the training resolution with a triangle filter (it is a smooth scalar
field, so blending is right); a colour file is reduced to luma and its own
alpha is ignored. The loader logs a census: `Dataset loss weights: N
view(s) carry a weights/ sidecar`.

## What it reaches

Everything the view contributes, so that "listen less here" means the same
thing to every term:

| term | how the weight enters |
|---|---|
| L1 + SSIM (and the alpha-match lane on transparent views) | the `[H, W, C]` loss map is multiplied before the frame mean |
| normal supervision | a weighted masked mean: both the residual plane and the count plane are multiplied, so the unweighted region keeps its scale |
| `--normalize-masked-loss` | unchanged — coverage is still the mask's mean, not the weight's |
| the evidence pass (`--export-evidence`) | the per-pixel mask lane `m` becomes `m · w`, so `ev_w_in` and `ev_err` shrink where the trainer was told to listen less |

That last row matters for a confidence-gated render: a view whose loss was
silenced over some region must not still count as full evidence — and full
disagreement — for what the *other* views put there.

LPIPS (off by default) is not weighted: its receptive field spans any
weight boundary, as it spans the mask boundary, and it is already skipped
on masked views for the same reason.

## The use it was built for

Several sources describing one surface at different levels of trust: a
photograph-derived render of a face handed to the training as masked
supporting views, and the frames a diffusion model produced of the same
face, which have to keep carving the silhouette but should not argue with
the photograph. The frames get a weight map that fades to `1 − strength`
where the face render covers them, and the face render wins.
