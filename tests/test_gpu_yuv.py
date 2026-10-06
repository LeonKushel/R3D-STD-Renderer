"""R3D_STD_GPU_YUV must hand the encoder the bytes ffmpeg itself would have
produced from the same RGB frame. Two links, both tested here:

  ffmpeg  ==  rgb_to_yuv420p (numpy)      skipped where ffmpeg is not installed
  rgb_to_yuv420p  ==  the shader pair     skipped where there is no GL context

Noise is the test image on purpose: a smooth picture hides the chroma filter
(a 2x2 box average agrees with the real eight-row filter on a gradient and on
only 7% of noise samples)."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

from osu_std_renderer.render import gl


def _ffmpeg(rgb: np.ndarray) -> "np.ndarray | None":
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    h, w = rgb.shape[:2]
    p = subprocess.run(
        [ff, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
         f"{w}x{h}", "-i", "pipe:0", "-pix_fmt", "yuv420p", "-f", "rawvideo",
         "pipe:1"], input=rgb.tobytes(), capture_output=True, timeout=60)
    if p.returncode or len(p.stdout) != w * h * 3 // 2:
        return None
    return np.frombuffer(p.stdout, np.uint8)


def _images():
    rng = np.random.default_rng(20261006)
    for w, h in ((1280, 720), (854, 480), (640, 360), (64, 16), (2, 12)):
        yield f"noise {w}x{h}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    # one-pixel black/white rows: the hardest case for the vertical filter,
    # and it drives the sums past both clip limits
    stripes = np.zeros((96, 160, 3), np.uint8)
    stripes[1::2] = 255
    yield "stripes", stripes
    sat = np.zeros((48, 64, 3), np.uint8)
    sat[:24, :, 0] = 255
    sat[24:, :, 2] = 255
    yield "saturated red over blue", sat


def test_twin_equals_ffmpeg_on_every_sample():
    ran = 0
    for name, rgb in _images():
        ref = _ffmpeg(rgb)
        if ref is None:
            print("SKIP (no usable ffmpeg)")
            return
        got = gl.rgb_to_yuv420p(rgb)
        bad = int((got != ref).sum())
        assert bad == 0, f"{name}: {bad} of {ref.size} samples differ from ffmpeg"
        ran += 1
    assert ran == 7


def test_under_twelve_rows_is_refused_not_approximated():
    # swscale shortens its chroma filter there; the twin clamps, so they part
    rgb = np.random.default_rng(1).integers(0, 256, (8, 16, 3), dtype=np.uint8)
    ref = _ffmpeg(rgb)
    if ref is None:
        print("SKIP (no usable ffmpeg)")
        return
    assert (gl.rgb_to_yuv420p(rgb) != ref).any()          # documents WHY the guard exists
    assert "12 rows" in open(gl.__file__).read()


def test_chroma_weights_are_the_full_precision_ones():
    # the 8-bit rounding of these (-4 -11 31 112 ...) is the near miss
    assert sum(gl._CHROMA_TAPS) == 8192
    assert gl._CHROMA_TAPS == tuple(reversed(gl._CHROMA_TAPS))


def test_shader_equals_twin():
    try:
        spr = gl.SpriteRenderer(320, 180)
    except Exception:  # noqa: BLE001 -- no GL device on this box
        print("SKIP (no GL context)")
        return
    rng = np.random.default_rng(7)
    for trial in range(3):
        # the scene texture's rows are stored bottom-up; the conversion is the
        # same read in either direction (symmetric weights and edge clamp), so
        # the array is compared in the texture's own row order
        rgba = rng.integers(0, 256, (180, 320, 4), dtype=np.uint8)
        if trial == 2:                      # stripes through the clip limits
            rgba[:] = 0
            rgba[1::2] = 255
        spr.color_tex.write(rgba.tobytes())
        spr._ensure_yuv()
        spr._run_yuv_passes(spr.color_tex)
        y = np.frombuffer(spr._fbo_y.read(components=1, alignment=1), np.uint8)
        u = np.frombuffer(spr._fbo_uv.read(components=1, alignment=1,
                                           attachment=0), np.uint8)
        v = np.frombuffer(spr._fbo_uv.read(components=1, alignment=1,
                                           attachment=1), np.uint8)
        want = gl.rgb_to_yuv420p(np.ascontiguousarray(rgba[..., :3]))
        got = np.concatenate([y, u, v])
        bad = int((got != want).sum())
        assert bad == 0, f"trial {trial}: {bad} of {want.size} samples differ"
