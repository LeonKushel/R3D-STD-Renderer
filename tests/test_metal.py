"""render/metal: the Metal renderer against the GL renderer it must stand in
for, and its colour conversion against ffmpeg. Every test skips where there is
no Metal library to load (anything but a Mac) or no GL context to compare with.

What "equal" means here is deliberate. Sprites, slider bodies, the downscale
and the colour conversion are compared for EXACT equality: they are. Whole HUD
frames are not asserted equal, because GL's and Metal's mip generators part by
one level on a few texels from mip level 3 down; that is covered by the
whole-render comparison (see the Metal ledger), not by a unit test."""
from __future__ import annotations

import math
import random
import sys

import numpy as np


def _pair(w: int, h: int):
    if sys.platform != "darwin":
        return None
    try:
        from osu_std_renderer.render import gl
        from osu_std_renderer.render.metal.renderer import MetalSpriteRenderer
        return gl.SpriteRenderer(w, h), MetalSpriteRenderer(w, h)
    except Exception as e:  # noqa: BLE001 - no GL context or no Metal device
        print(f"SKIP ({e})")
        return None


def test_sprites_draw_what_gl_draws():
    pair = _pair(1280, 720)
    if pair is None:
        return
    from osu_std_renderer.render import gl
    rng = np.random.default_rng(7)
    tex = [(f"t{i}", rng.integers(0, 256, (int(rng.integers(3, 150)), int(rng.integers(3, 150)), 4),
                                  dtype=np.uint8), bool(i % 3 == 0), bool(i % 2)) for i in range(40)]
    frames = []
    for spr in pair:
        for key, px, clamp, mips in tex:
            spr.upload_texture(key, px, clamp=clamp, mipmaps=mips)
        spr.begin((0.1, 0.2, 0.3))
        for n, seed in ((300, 5), (1, 6), (120, 7)):
            random.seed(seed)
            batch = []
            for j in range(n):
                key = None if j % 9 == 0 else f"t{random.randrange(40)}"
                batch.append(gl.Sprite(
                    random.uniform(0, 1280), random.uniform(0, 720),
                    random.uniform(2, 260), random.uniform(2, 260), key,
                    (random.random(), random.random(), random.random(), random.random()),
                    rotation=random.uniform(-3, 3) if j % 4 == 0 else 0.0,
                    additive=(j % 5 == 0),
                    uv_off=(0.0, random.uniform(0, 0.5)) if j % 7 == 0 else (0.0, 0.0),
                    uv_scale=(1.0, 0.5) if j % 7 == 0 else (1.0, 1.0)))
            spr.draw(batch)
        frames.append(np.asarray(spr.read_rgb()).astype(int))
    d = np.abs(frames[0] - frames[1])
    assert frames[0].max() - frames[0].min() > 100            # a real picture
    assert d.max() == 0, f"{int((d != 0).sum())} values differ, largest {d.max()}"


def test_slider_bodies_draw_what_gl_draws():
    pair = _pair(1920, 1080)
    if pair is None:
        return
    from osu_std_renderer.render import gl
    from osu_std_renderer.render.metal.slider_body import MetalSliderBodyRenderer
    from osu_std_renderer.render.slider_body import BodyStyle, SliderBodyRenderer
    g, m = pair
    w, h = 1920, 1080
    random.seed(11)
    paths = []
    for _ in range(3):
        cx, cy, r = random.uniform(0.25, 0.75) * w, random.uniform(0.3, 0.7) * h, random.uniform(0.1, 0.25) * h
        a0 = random.uniform(0, 6.28)
        paths.append([(cx + r * math.cos(a0 + t * 0.09), cy + r * math.sin(a0 + t * 0.07) * 0.8)
                      for t in range(48)])
    paths.append([(0.1 * w, 0.1 * h), (0.5 * w, 0.15 * h), (0.9 * w, 0.6 * h)])
    out = []
    for spr, bodies in ((g, SliderBodyRenderer(g.ctx, w, h)), (m, MetalSliderBodyRenderer(m, w, h))):
        spr.begin((0.02, 0.02, 0.04))
        for i, p in enumerate(paths):
            style = BodyStyle(body_color=(0.2 + 0.2 * i, 0.5, 0.9 - 0.2 * i), alpha=0.6 + 0.1 * i)
            body = bodies.build_body(p, radius_px=h * 0.045, style=style, snake=(0.0, 1.0 - 0.1 * i))
            bodies.draw_body(body, spr.fbo, alpha=0.9)
            spr.draw([gl.Sprite(p[0][0], p[0][1], h * 0.09, h * 0.09, None, (1, 1, 1, 0.5))])
        out.append(np.asarray(spr.read_rgb()).astype(int))
    d = np.abs(out[0] - out[1])
    assert d.max() == 0, f"{int((d != 0).sum())} values differ, largest {d.max()}"


def test_frame_conversion_equals_the_reference_routine():
    pair = _pair(640, 360)
    if pair is None:
        return
    from osu_std_renderer.render import gl
    _, m = pair
    rng = np.random.default_rng(3)
    m.upload_texture("n", rng.integers(0, 256, (360, 640, 4), dtype=np.uint8), mipmaps=False)
    for _ in range(2):
        m.begin((0.0, 0.0, 0.0))
        m.draw([gl.Sprite(320.0, 180.0, 640.0, 360.0, "n", (1.0, 1.0, 1.0, 1.0))])
        rgb = m.read_rgb()                                   # top-down picture
    m.begin((0.0, 0.0, 0.0))
    m.draw([gl.Sprite(320.0, 180.0, 640.0, 360.0, "n", (1.0, 1.0, 1.0, 1.0))])
    assert m.read_yuv_async() is None                        # the pipeline is filling
    yuv = np.array(m.read_yuv_drain()[-1])
    # the writer's contract: planes bottom-up, i.e. the conversion of the flipped picture
    want = gl.rgb_to_yuv420p(np.ascontiguousarray(rgb[::-1]))
    assert int((yuv != want).sum()) == 0


def test_results_downscale_equals_pillow():
    if sys.platform != "darwin":
        return
    try:
        from osu_std_renderer.render.metal.renderer import MetalSpriteRenderer
        from osu_std_renderer.render.metal.ssaa import MetalLanczos
        from osu_std_renderer.render.scene import ssaa_internal_size
    except Exception as e:  # noqa: BLE001
        print(f"SKIP ({e})")
        return
    for ow, oh in ((1280, 720), (854, 480), (960, 540)):
        iw, ih = ssaa_internal_size(ow, oh)
        spr = MetalSpriteRenderer(ow, oh)
        hi = MetalSpriteRenderer(iw, ih, core=spr.core)
        assert MetalLanczos(spr, hi).self_check(), f"{iw}x{ih} -> {ow}x{oh} differs from Pillow"
        spr.release()


def test_a_reupload_never_rewrites_the_texture_in_place():
    # frames already committed may not have run yet; an in-place rewrite would
    # change what they sample (seen as wrong numbers on the results card)
    pair = _pair(64, 64)
    if pair is None:
        return
    _, m = pair
    a = np.full((8, 8, 4), 10, np.uint8)
    m.upload_texture("k", a)
    first = m._tex["k"]
    m.upload_texture("k", a + 1)
    assert m._tex["k"] != first


def _main_under(rc=None, raises=None, metal=False):
    """Run the package's __main__ with the CLI and the re-exec stubbed.
    Returns (exit code, the environment a re-run was started with or None)."""
    import os
    import runpy
    from osu_std_renderer import cli
    from osu_std_renderer.render import perf
    saved = (cli.main, os.execve, perf.METAL_IN_USE)
    reran = []

    class _Reexec(Exception):
        pass

    def fake_main():
        perf.METAL_IN_USE = metal
        if raises is not None:
            raise raises
        return rc

    def fake_execve(exe, argv, env):
        reran.append((argv, env))
        raise _Reexec()
    cli.main, os.execve = fake_main, fake_execve
    code = None
    try:
        try:
            runpy.run_module("osu_std_renderer", run_name="__main__")
        except SystemExit as e:
            code = e.code
        except _Reexec:
            code = "re-run"
    finally:
        cli.main, os.execve, perf.METAL_IN_USE = saved
    return code, (reran[0] if reran else None)


def test_a_failed_metal_render_is_run_again_on_opengl():
    code, rerun = _main_under(rc=1, metal=True)
    assert code == "re-run" and rerun[1]["R3D_STD_METAL"] == "0"
    assert rerun[0][1:3] == ["-m", "osu_std_renderer"]
    code, rerun = _main_under(raises=RuntimeError("device lost"), metal=True)
    assert code == "re-run" and rerun[1]["R3D_STD_METAL"] == "0"
    code, rerun = _main_under(raises=SystemExit(3), metal=True)
    assert code == "re-run"


def test_nothing_else_is_run_again():
    assert _main_under(rc=0, metal=True) == (0, None)            # it worked
    assert _main_under(rc=2, metal=False) == (2, None)           # a GL render failed: its own failure
    assert _main_under(raises=SystemExit(0), metal=True) == (0, None)
    try:                                                          # a GL exception is not swallowed
        _main_under(raises=RuntimeError("x"), metal=False)
        raise AssertionError("expected the exception")
    except RuntimeError:
        pass
    try:                                                          # a cancelled job is not re-run
        _main_under(raises=KeyboardInterrupt(), metal=True)
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass


def test_which_switches_ask_for_metal():
    """R3D_STD_METAL decides when set; otherwise the node's R3D_METAL does;
    R3D_STD_STOCK=1 always means OpenGL. Read from the CLI's own source so the
    rule cannot drift from this table."""
    import os
    import subprocess
    code = ("import sys; from osu_std_renderer.render import perf;"
            "print(int(bool(perf.envflag('R3D_STD_METAL', perf.envflag('R3D_METAL')) and not perf.STOCK)))")
    src = open(os.path.join(os.path.dirname(__file__), "..", "osu_std_renderer", "cli.py")).read()
    assert "perf.envflag(\"R3D_STD_METAL\", perf.envflag(\"R3D_METAL\"))" in src and "not perf.STOCK" in src

    def asked(**env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("R3D_")}
        e.update(env)
        return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True,
                              cwd=os.path.join(os.path.dirname(__file__), ".."), check=True).stdout.strip() == "1"
    assert not asked()
    assert asked(R3D_STD_METAL="1")
    assert asked(R3D_METAL="1")                               # the node's switch
    assert not asked(R3D_METAL="1", R3D_STD_METAL="0")        # std kept on GL by hand
    assert asked(R3D_METAL="0", R3D_STD_METAL="1")
    assert not asked(R3D_METAL="1", R3D_STD_STOCK="1")
    assert not asked(R3D_STD_METAL="1", R3D_STD_STOCK="1")
