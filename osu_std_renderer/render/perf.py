"""Lightweight phase timing + frame-stream hashing for perf work.

Two independent env switches, both OFF by default (zero output change):

  R3D_TIMING=1     accumulate per-phase wall time + counters, print a
                   breakdown to stderr at process exit.
  R3D_FRAME_MD5=1  hash every raw RGB frame pushed to ffmpeg (blake2b)
                   and print one digest at pipe close — bit-identical
                   output proof across perf changes.

The timers themselves stay armed even when disabled (a perf_counter pair
per coarse phase, ~40 pairs/frame ≈ 40 µs — noise at an 11 ms frame), so
the measured build is the shipped build; only the report is gated.
"""
from __future__ import annotations

import atexit
import os
import sys
import time
from collections import defaultdict

def envflag(name: str, default: bool = False) -> bool:
    """Parse an R3D_* switch the way a reader expects.

    `bool(os.environ.get(X))` is True for the STRING "0", so `R3D_FOO=0` turns the
    feature ON. That cost us one invalid A/B (both arms instrumented) before the
    report appeared in the arm that was supposed to be silent."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


# ---- which render-path speedups are on with no switch set --------------------
# On a Mac the whole set is ON by default: that is where every one of them was
# built, timed and checked frame for frame against the stock path. Anywhere
# else nothing is on until that platform has been validated and this rule is
# widened. Each one stays a switch (R3D_X=0 off, R3D_X=1 on, on any platform),
# and R3D_STD_STOCK=1 turns the whole set off at once: the stock render path,
# for regression gates ("does stock still equal main?") and for bisecting.
STOCK = envflag("R3D_STD_STOCK")
FAST_DEFAULT = sys.platform == "darwin" and not STOCK

# via envflag, so `R3D_TIMING=0` DISABLES rather than enables (row 42).
TIMING = envflag("R3D_TIMING")
FRAME_MD5 = envflag("R3D_FRAME_MD5")
# R3D_CPROF=<region>  cProfile ONE named region (see P()). Reports call COUNTS
# reliably; its per-call times are inflated by the enable/disable, so read counts
# here and timings from T()/ACC.
CPROF = os.environ.get("R3D_CPROF", "")

ACC: dict[str, float] = defaultdict(float)
CNT: dict[str, int] = defaultdict(int)
# Wall-clock milestones for the WHOLE process, not just the frame loop.
# T()/ACC divides by pushed frames, which makes FIXED costs invisible: taiko had
# 1.84 s of setup + encoder drain (22% of a short render) hiding behind exactly
# that blind spot, because its "done: N frames in Xs" line started after setup and
# ended before the encoder finished. These marks make that span visible.
MARKS: list[tuple[str, float]] = []

# Reference instants captured as early as import allows. T_IMPORT is the monotonic
# zero for every mark; T_IMPORT_EPOCH exists only to subtract R3D_T0 (see below),
# because interpreter startup + import cost happens BEFORE any in-process timer
# can run, so the only way to see it is to compare against a stamp taken by the
# shell that launched us:
#     R3D_T0=$(date +%s) ... or python -c 'import time;print(time.time())'
T_IMPORT = time.perf_counter()
T_IMPORT_EPOCH = time.time()
try:
    _T0_SHELL: float | None = float(os.environ["R3D_T0"])
except (KeyError, ValueError):
    _T0_SHELL = None


def mark(label: str, leaf: bool = False) -> None:
    if leaf:
        LEAF.add(label)
    if TIMING:
        MARKS.append((label, time.perf_counter()))
        _cur[0] = label


# Phases NEST (hud_combo inside hud_draw inside frame_render), so summing every
# label double-counts. TOP accumulates only depth-0 spans, which is what may be
# compared against wall time to find UNMEASURED work.
TOP: dict[str, float] = defaultdict(float)
_depth = [0]
# Each depth-0 T span is attributed to the mark-span it STARTED in, keyed
# (span_start_label, phase). That turns the report into an accounting tree:
# every mark-span's duration is known exactly, so whatever its children do not
# explain is a RESIDUAL -- real unmeasured work, as opposed to work that a mark
# already covers in one lump. Summed residuals are the number that must reach 0.
SPAN: dict[tuple[str, str], float] = defaultdict(float)
_cur = ["startup"]
# Spans declared TERMINAL: their own work is one named cost (a blocking wait on a
# worker thread, a shader compile, a subprocess finishing), so there is nothing
# left to subdivide and the residual is the measurement rather than a gap.
LEAF: set[str] = set()


class T:
    """with T("phase"): ... — accumulate wall time under a label."""
    __slots__ = ("name", "t0", "d", "span")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.d = _depth[0]
        # Attribute at ENTER: a span that straddles a mark is charged to where it
        # began, which keeps every span in exactly one bucket.
        self.span = _cur[0]
        _depth[0] += 1
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        _depth[0] -= 1
        ACC[self.name] += dt
        CNT[self.name] += 1
        if self.d == 0:                 # only outermost spans sum to wall time
            TOP[self.name] += dt
            SPAN[(self.span, self.name)] += dt
        return False


def count(name: str, n: int = 1) -> None:
    CNT[name] += n


class P:
    """with P("name"): ...  — cProfile this region iff R3D_CPROF matches `name`.

    Wrapped around a REGION rather than the process on purpose: a top-level
    profile drowns the interesting part in frame-loop noise, and on a threaded
    renderer it only follows the thread that enabled it."""

    __slots__ = ("on",)
    _prof = None

    def __init__(self, name: str):
        self.on = bool(CPROF) and CPROF == name
        if self.on and P._prof is None:
            import cProfile
            P._prof = cProfile.Profile()

    def __enter__(self):
        if self.on:
            P._prof.enable()
        return self

    def __exit__(self, *exc):
        if self.on:
            P._prof.disable()
        return False


def _report_cprof() -> None:
    if P._prof is None:
        return
    import io
    import pstats
    buf = io.StringIO()
    pstats.Stats(P._prof, stream=buf).sort_stats("tottime").print_stats(30)
    print(f"\n=== R3D_CPROF region {CPROF!r} (read CALL COUNTS, not times) ===",
          file=sys.stderr)
    print(buf.getvalue(), file=sys.stderr, flush=True)


def _report() -> None:
    # Close the span HERE rather than via a separate atexit hook: atexit runs
    # LIFO, so a hook registered later would fire first and the ordering would
    # depend on import order. Appending from inside the report cannot race.
    MARKS.append(("process_exit", time.perf_counter()))
    frames = CNT.get("encode_push", 0) or 1
    lines = []

    if len(MARKS) > 1:
        t0, tN = MARKS[0][1], MARKS[-1][1]
        wall = tN - t0
        lines += ["", "=== R3D_TIMING accounting tree (whole process) ===",
                  "  every span's duration is exact; RESIDUAL is what its "
                  "children fail to explain", ""]
        residual_total = 0.0
        # A residual is only a PROBLEM when it is big: a span that hides 1%+ of
        # wall is an un-subdivided lump and needs marks inside it. Small ones are
        # already as named as they are worth making.
        LUMP = 0.01 * wall
        lumps, tiny_n, tiny_s = [], 0, 0.0
        for (l0, ta), (l1, tb) in zip(MARKS, MARKS[1:]):
            dur = tb - ta
            kids = {n: v for (sp, n), v in SPAN.items() if sp == l0}
            resid = dur - sum(kids.values())
            residual_total += resid
            if resid > LUMP and l0 not in LEAF:
                lumps.append((resid, f"{l0} -> {l1}"))
            if dur < 0.005 * wall and not kids:
                tiny_n += 1
                tiny_s += dur
                continue
            lines.append(f"  {l0} -> {l1}".ljust(50)
                         + f"{dur:7.2f}s  {dur / wall * 100:5.1f}% of wall")
            for n in sorted(kids, key=lambda k: -kids[k]):
                lines.append(f"      {n:<34} {kids[n]:7.2f}s"
                             f"  {kids[n] / dur * 100 if dur else 0:5.1f}% of span")
            flag = ("  (terminal)" if l0 in LEAF
                    else "  <-- LUMP, subdivide" if resid > LUMP else "")
            lines.append(f"      {'(span own work)':<34} {resid:7.2f}s"
                         f"  {resid / dur * 100 if dur else 0:5.1f}% of span{flag}")
        if tiny_n:
            lines.append(f"  [{tiny_n} spans under 0.5% of wall, childless]"
                         .ljust(50) + f"{tiny_s:7.2f}s  "
                         f"{tiny_s / wall * 100:5.1f}% of wall")
        lines += [
            "",
            f"  {'WALL (perf_import -> exit)':<34} {wall:7.2f}s",
            f"  {'EXPLAINED':<34} {wall - residual_total:7.2f}s  "
            f"{(wall - residual_total) / wall * 100 if wall else 0:5.1f}%",
            f"  {'in named T() phases':<34} {wall - residual_total:7.2f}s  "
            f"{(wall - residual_total) / wall * 100 if wall else 0:5.1f}%",
            f"  {'in span own-work (named, coarse)':<34} "
            f"{residual_total - sum(r for r, _ in lumps):7.2f}s  "
            f"{(residual_total - sum(r for r, _ in lumps)) / wall * 100:5.1f}%",
            f"  {'OPAQUE (lumps >1% of wall)':<34} "
            f"{sum(r for r, _ in lumps):7.2f}s  "
            f"{sum(r for r, _ in lumps) / wall * 100 if wall else 0:5.1f}%"
            "   <- COMPLETE when ~0",
        ]
        for r, name in sorted(lumps, reverse=True)[:6]:
            lines.append(f"      lump  {name:<44} {r:7.2f}s")
        if _T0_SHELL is not None:
            boot = T_IMPORT_EPOCH - _T0_SHELL
            lines += [f"  {'+ interp+import (pre-timer, R3D_T0)':<34} {boot:7.2f}s",
                      f"  {'= TOTAL PROCESS':<34} {wall + boot:7.2f}s"]
        else:
            lines.append("  (set R3D_T0=$(python -c 'import time;print(time.time())')"
                         " to also capture interpreter startup)")

    lines += ["", f"=== R3D_TIMING per-phase (over {frames} pushed frames) ==="]
    for name in sorted(ACC, key=lambda k: -ACC[k]):
        tot = ACC[name]
        lines.append(f"  {name:<24} {tot:8.2f}s total  "
                     f"{tot / frames * 1000.0:8.3f} ms/frame  "
                     f"{CNT[name] / frames:8.1f} calls/frame")
    for name in sorted(CNT):
        if name not in ACC:
            lines.append(f"  {name:<24} {CNT[name] / frames:8.1f} /frame  "
                         f"({CNT[name]} total)")
    print("\n".join(lines), file=sys.stderr, flush=True)


if TIMING:
    MARKS.append(("perf_import", T_IMPORT))
    atexit.register(_report)
if CPROF:
    atexit.register(_report_cprof)
