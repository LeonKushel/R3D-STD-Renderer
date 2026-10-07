# render/metal: std on Metal (macOS, Apple Silicon)

`R3D_STD_METAL=1` on a Mac draws the render with Metal instead of OpenGL. The
scene code is the same; only the renderer underneath changes (`renderer.py`
stands in for `render/gl.py`'s `SpriteRenderer`, `slider_body.py` for the slider
body renderer, `ssaa.py` for the results-screen downscale). Frames leave it as
finished yuv420p, converted on the GPU with the arithmetic `render/gl.py`
holds equal to ffmpeg's own.

Default OFF. Off, nothing in this directory is imported.

## What it does not do

* **Bloom**: not on the Metal path. A render that asks for bloom is drawn with
  OpenGL (decided before the first frame, in the same process).
* **Merged renders** (`merge.py`): they build their own OpenGL renderer and are
  not touched.
* It needs the GPU colour conversion (`R3D_STD_GPU_YUV`, on by default on a Mac
  where the local ffmpeg agrees with it); without it the render is drawn with
  OpenGL.

## If it fails

Setting it up is guarded: no Metal device, a library that does not load, bloom,
no GPU colour conversion -> the render uses OpenGL, with a line on stderr saying
why. A failure after that (an exception or a non-zero exit while drawing with
Metal) makes `__main__` run the whole job again on OpenGL in a fresh process.
A cancelled job is not re-run.

## How close it is to OpenGL

Sprites, slider bodies, the downscale and the colour conversion are equal to
the OpenGL renderer's sample for sample (`tests/test_metal.py`). Whole frames
differ by at most 2 levels on a few HUD texels: OpenGL's and Metal's mipmap
generators part by one level from mip level 3 down.

## The library

`libr3dcore.dylib` is arm64 and is built from `src/*.swift` by `build.sh`,
nothing else. The build is repeatable: on an M1 Max with Apple Swift 6.4
(swiftlang-6.4.0.34.1, target arm64-apple-macosx26.0) it gives the same file
every time,

    md5    b580f2ecc9f24a0027d16bafab5e5165
    sha256 3e24fb0c7027c1a11cddd88776109cea558f8737a65356e22085cce5272ad655

To check the committed binary, run `build.sh` on a Mac with that toolchain and
compare.
