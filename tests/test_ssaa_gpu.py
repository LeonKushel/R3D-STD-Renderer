"""render/ssaa_gpu.py: the integer LANCZOS downscale must equal Pillow's,
byte for byte. The arithmetic is tested through its numpy twin (pure CPU, no
GL); the shader pair is the same arithmetic and is checked against Pillow on
the live GL context at render time (GpuLanczos.self_check)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from osu_std_renderer.render.scene import ssaa_internal_size
from osu_std_renderer.render.ssaa_gpu import (PRECISION_BITS, coeffs,
                                              resize_exact_cpu)


def _pil(img, ow, oh):
    return np.asarray(Image.fromarray(img).resize((ow, oh), Image.LANCZOS),
                      dtype=np.uint8)


def test_coeffs_shape_matches_pillows_kernel_size():
    # 1080 -> 720: scale 1.5, support 4.5, ksize ceil(4.5)*2+1
    bounds, kk = coeffs(1080, 720)
    assert kk.shape == (720, 11) and bounds.shape == (720, 2)
    assert bounds[:, 0].min() == 0
    assert (bounds[:, 0] + bounds[:, 1]).max() == 1080
    # every row's fixed-point coefficients sum to ~1.0
    one = 1 << PRECISION_BITS
    assert np.abs(kk.sum(axis=1) - one).max() <= kk.shape[1]


def test_twin_equals_pillow_at_every_output_size_the_outro_uses():
    rng = np.random.default_rng(11)
    for ow, oh in ((1280, 720), (854, 480), (960, 540), (1600, 900),
                   (640, 360), (1024, 768)):
        iw, ih = ssaa_internal_size(ow, oh)
        # full width, a strip of rows: enough to cover the vertical kernel
        img = rng.integers(0, 256, (ih, iw, 3), dtype=np.uint8)[:, : iw // 4]
        w4 = img.shape[1]
        ow4 = max(2, round(ow * w4 / iw))
        assert np.array_equal(resize_exact_cpu(img, ow4, oh), _pil(img, ow4, oh))
        # and the horizontal coefficients at the real widths, on a few rows
        row = rng.integers(0, 256, (4, iw, 3), dtype=np.uint8)
        assert np.array_equal(resize_exact_cpu(row, ow, 4), _pil(row, ow, 4))


def test_twin_equals_pillow_at_the_clip_limits():
    # 0/255 noise drives the sums past both ends of the 8-bit range
    rng = np.random.default_rng(12)
    img = (rng.integers(0, 2, (270, 480, 3), dtype=np.uint8) * 255)
    assert np.array_equal(resize_exact_cpu(img, 320, 180), _pil(img, 320, 180))
    flat = np.full((90, 160, 3), 255, np.uint8)
    assert np.array_equal(resize_exact_cpu(flat, 100, 60), _pil(flat, 100, 60))


def test_identity_size_is_untouched():
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert resize_exact_cpu(img, 6, 4) is img
