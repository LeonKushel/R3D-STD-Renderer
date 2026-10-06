"""R3D_STD_MULTITEX: several textures per draw call must draw what one
texture per draw call draws. The plan (which sprite samples which unit, where
a call ends) is pure and tested here without a GPU; the drawn bytes are
compared flag off vs flag on in two child processes, because the flag picks
the shader pair and the vertex layout when render/gl.py is imported."""
from __future__ import annotations

import os
import subprocess
import sys

from osu_std_renderer.render.gl import Sprite, plan_texture_batches


def _sprites(keys, additive=()):
    return [Sprite(0.0, 0.0, 1.0, 1.0, k, additive=(i in additive))
            for i, k in enumerate(keys)]


def _check(ordered, n_norm, max_units):
    units, batches = plan_texture_batches(ordered, n_norm, max_units)
    assert len(units) == len(ordered)
    covered = 0
    for first, end, keys in batches:
        assert first == covered and end > first      # contiguous, in order
        assert 1 <= len(keys) <= max_units
        assert len(set(keys)) == len(keys)           # one unit per texture
        for i in range(first, end):                  # each sprite: its own texture
            assert keys[units[i]] == ordered[i].texture_key
        covered = end
    assert covered == len(ordered)
    return units, batches


def test_one_batch_when_the_textures_fit():
    keys = ["a", "b", "c"] * 40                      # 120 sprites, 3 textures
    units, batches = _check(_sprites(keys), len(keys), 15)
    assert batches == [(0, 120, ["a", "b", "c"])]
    assert units[:6] == [0, 1, 2, 0, 1, 2]


def test_a_batch_ends_only_when_it_needs_one_unit_too_many():
    keys = [f"t{i}" for i in range(16)] + ["t15", "t0", "t15"]
    units, batches = _check(_sprites(keys), len(keys), 15)
    # the 16th distinct texture opens the second batch; t0 is new to it
    assert [(f, e) for f, e, _ in batches] == [(0, 15), (15, 19)]
    assert batches[1][2] == ["t15", "t0"]
    assert units[15:] == [0, 0, 1, 0]


def test_the_blend_passes_never_share_a_call():
    keys = ["a", "b", "a", "b", "a"]
    units, batches = _check(_sprites(keys), 3, 15)   # 3 normal, 2 additive
    assert [(f, e) for f, e, _ in batches] == [(0, 3), (3, 5)]
    assert batches[1][2] == ["b", "a"]               # units restart per batch
    assert units == [0, 1, 0, 0, 1]


def test_all_additive_and_no_additive():
    keys = ["a", None, "a", None]
    _, only_add = _check(_sprites(keys), 0, 15)
    assert only_add == [(0, 4, ["a", None])]
    _, no_add = _check(_sprites(keys), 4, 15)
    assert no_add == [(0, 4, ["a", None])]


def test_solid_quads_take_a_unit_like_any_texture():
    keys = [None, "a", None, "", "a"]                # None and "" both mean solid
    units, batches = _check(_sprites(keys), len(keys), 15)
    assert batches[0][2] == [None, "a", ""]
    assert units == [0, 1, 0, 2, 1]


def test_many_textures_many_batches():
    keys = [f"g{i % 47}" for i in range(500)]
    _, batches = _check(_sprites(keys), 350, 15)
    assert sum(e - f for f, e, _ in batches) == 500
    assert any(f == 350 for f, _, _ in batches)


# One frame that leans on everything the fragment stage touches: 40 textures of
# different sizes, half with mip chains (minified, so the mip level matters),
# rotations, sub-rects, both blend passes, solid quads, three draw() calls.
_CHILD = r"""
import hashlib, random, sys
import numpy as np
from osu_std_renderer.render import gl
spr = gl.SpriteRenderer(640, 360)
rng = np.random.default_rng(7)
for i in range(40):
    h, w = int(rng.integers(3, 150)), int(rng.integers(3, 150))
    spr.upload_texture(f"t{i}", rng.integers(0, 256, (h, w, 4), dtype=np.uint8),
                       clamp=bool(i % 3 == 0), mipmaps=bool(i % 2))
random.seed(5)
def batch(n):
    out = []
    for j in range(n):
        key = None if j % 9 == 0 else f"t{random.randrange(40)}"
        out.append(gl.Sprite(
            random.uniform(0, 640), random.uniform(0, 360),
            random.uniform(2, 260), random.uniform(2, 260), key,
            (random.random(), random.random(), random.random(), random.random()),
            rotation=random.uniform(-3, 3) if j % 4 == 0 else 0.0,
            additive=(j % 5 == 0),
            uv_off=(0.0, random.uniform(0, 0.5)) if j % 7 == 0 else (0.0, 0.0),
            uv_scale=(1.0, 0.5) if j % 7 == 0 else (1.0, 1.0)))
    return out
spr.begin((0.1, 0.2, 0.3))
for n in (300, 1, 120):
    spr.draw(batch(n))
frame = spr.read_rgb()
print("MULTITEX", int(gl._MULTITEX), hashlib.md5(frame.tobytes()).hexdigest(),
      int(frame.max()) - int(frame.min()))
"""


def _child(flag: str):
    env = {k: v for k, v in os.environ.items() if not k.startswith("R3D_")}
    env["R3D_STD_MULTITEX"] = flag
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.run([sys.executable, "-c", _CHILD], cwd=root, env=env,
                       capture_output=True, text=True, timeout=120)
    for line in p.stdout.splitlines():
        if line.startswith("MULTITEX "):
            return line.split()[1:]
    return None


def test_drawn_bytes_equal_flag_off_and_on():
    off = _child("0")
    if off is None:
        print("SKIP (no GL context)")
        return
    on = _child("1")
    assert on is not None, "the multi-texture path failed where the default drew"
    assert off[0] == "0" and on[0] == "1"            # each child ran its own path
    assert int(off[2]) > 100                         # a real picture, not a blank
    assert off[1] == on[1], f"frame differs: {off[1]} vs {on[1]}"
