# `brush-splat-render`: rendering a trained ply outside a training session

`brush-splat-render` is a standalone binary (`crates/brush-splat-render`)
that loads a trained Gaussian splat `.ply` and renders an explicit list of
cameras to RGBA PNGs. It exists to replace
[gsplat](https://github.com/nerfstudio-project/gsplat) in body2colmap's
`RenderSplatStep`: gsplat publishes no wheel past torch 2.4 / cu124, so on a
modern stack it JIT-compiles its CUDA kernels on first use and needs `nvcc`
at runtime — the reason the body2colmap pipeline image had to ship a CUDA
*devel* base. brush already renders splats, is already in that image, and is
wgpu/Vulkan, so routing this through it drops the CUDA toolchain entirely and
leaves one graphics API instead of two.

It is a **separate crate we maintain**, not a `brush` subcommand: a `render`
CLI is a fair bit of surface area for a niche use case that, unlike features
such as [normal-map supervision](normal-supervision.md), is unlikely to be
wanted upstream.

## Usage

```
brush-splat-render \
    --splat scene.ply \
    --cameras cameras.json \
    --output-dir out/ \
    [--background 1.0,1.0,1.0] \
    [--output-format png]
```

- `--splat`: a trained Gaussian splat `.ply`. Loaded through the same
  `brush_serde::load_splat_from_ply` path the viewer uses, so the standard
  3DGS ply conventions (log-space scales, logit opacities, `(N, K, 3)` SH
  coefficients) are handled automatically — there's no separate conversion
  step to keep in sync with brush's own loader.
- `--cameras`: a camera list, in `body2colmap.Camera`'s pixel-space terms
  (schema below).
- `--output-dir`: created if missing. One RGBA PNG per camera, named after
  that camera's JSON entry's `name` field (extension replaced by
  `--output-format`).
- `--background`: composited under the splat's accumulated alpha as
  `rgb·alpha + bg·(1−alpha)`. This composite, and the accumulated alpha
  itself, come straight from `brush_render::render_splats`'s
  `TextureMode::Float` output — no separate compositing step.
- `--output-format`: image encoder (default `png`).

**Alpha is the accumulated splat opacity, not a constant.** This matters for
any downstream step (e.g. body2colmap's `mask_splat`) that thresholds alpha
to drop the low-confidence fringes of a render.

With `--confidence` the contract changes: the render is gated by a per-splat
multi-view confidence, rejected pixels resolve to `--cull-color` (default
0.5 grey) and the written alpha is that gate rather than opacity. See
[splat-confidence.md](splat-confidence.md) — it is the in-renderer
replacement for `mask_splat`.

## `cameras.json` schema

One entry per view, in the pixel-space terms `body2colmap.Camera` already
holds, so the conversion to brush's FOV-and-quaternion form happens here in
Rust rather than on the Python side:

```json
{
  "width": 720,
  "height": 1280,
  "cameras": [
    {
      "name": "frame_00001_.png",
      "fx": 1213.917, "fy": 1213.917,
      "cx": 360.0, "cy": 640.0,
      "position": [x, y, z],
      "rotation": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    }
  ]
}
```

`width`/`height` are shared across all cameras in the file. `rotation` is
row-major (`rotation[row][col]`) and is `body2colmap.Camera.rotation`
serialized directly: a camera-to-world matrix whose **columns** are the
camera's local axes (X right, Y up, Z backward) expressed in world
coordinates — i.e. OpenGL convention, camera looks down −Z. Only pinhole
intrinsics are supported (`body2colmap.Camera` doesn't model distortion).

## The camera convention

This is the part worth getting right deliberately rather than by trial and
error, because a wrong-but-plausible convention produces an image that looks
almost correct — vertically mirrored, or subtly offset — rather than
obviously broken.

`body2colmap.Camera` stores an OpenGL-convention camera-to-world pose
(Y-up, camera looks down −Z). Brush's `Camera` (`crates/brush-render/src/camera.rs`)
stores an OpenCV-convention camera-to-world pose (Y-down, camera looks down
+Z) — confirmed by how `crates/brush-dataset/src/formats/colmap.rs` builds
one: it inverts COLMAP's world-to-camera quaternion/translation with no axis
flip, and COLMAP's own convention is OpenCV.

body2colmap's gsplat path (`splat_renderer.py`) goes from its OpenGL world-to-camera
matrix to the OpenCV one gsplat wants by left-multiplying with
`diag(1,−1,−1,1)`:

```python
viewmat_cv = opengl_to_opencv @ camera.get_w2c()
```

(Its docstring claims the opposite — "no conversion needed" — directly above
the code that converts. The code is correct; the comment is stale.)

Inverting that relation gives the camera-to-world rotation `brush-splat-render`
needs directly: `R_cv = R_gl · diag(1,−1,−1)`, i.e. **negate the Y and Z
columns** of the input rotation matrix and leave X untouched
(`to_brush_camera` in `crates/brush-splat-render/src/main.rs`). Position is
unchanged — both conventions share the same world frame; only the
per-camera local axes differ (`body2colmap.coordinates.world_to_colmap_camera`
flips axes, never the world itself).

**This was verified against a real gsplat oracle**, not just derived and
shipped: the same trained splat and the same 8 cameras (spanning an orbit),
rendered once through gsplat directly (COLMAP's w2c fed to gsplat as-is,
since COLMAP is already OpenCV) and once through `brush-splat-render` (the
same pose converted to OpenGL form and back, exercising the conversion above
end-to-end). Result: mean absolute error 0.0008–0.0015 on RGB and
0.0006–0.0010 on alpha (comfortably under the ~1/255 ≈ 0.0039 bar), with
differences confined to silhouette edges — the two rasterizers' expected
subpixel disagreement — and no structured mirroring, flip, or offset.

## What's deliberately out of scope here

- **Wiring this into `pipeline/steps/splat.py`.** Nothing on the body2colmap
  side has changed yet; `RenderSplatStep` still calls gsplat via
  `SplatRenderer`. Swapping that call for this binary (either by having
  `SplatRenderer` shell out, or by bypassing it from `splat.py` the way
  `steps/brush.py` already shells out to the `brush` binary) is a separate,
  later change.
- **Non-pinhole camera models.** `body2colmap.Camera` only carries `(fx, fy,
  cx, cy)`, so `cameras.json` never asks for more, even though brush's
  renderer supports fisheye/distortion models.
- **A regression-test fixture for this oracle comparison.** The comparison
  above was run manually against local data (`~/Documents/circ_colmap` +
  its trained `.ply`), not against images checked into this repo.
