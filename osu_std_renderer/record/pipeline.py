"""The deterministic record loop — RENDER_PLAN.md §5.2 EXACT
(app/app.go mainLoopRecord).

    updateFPS   = max(fps, 1000)        # simulation always ≥ 1000 Hz
    updateDelta = 1000 / updateFPS      # ms per Update
    fpsDelta    = 1000 / fps            # ms per rendered frame
    audioDelta  = 1.0                   # ms per audio push

    while not player.Update(updateDelta):   # WALL CLOCK NEVER CONSULTED
        deltaSumA += updateDelta
        while deltaSumA >= audioDelta: PushAudio(); deltaSumA -= audioDelta
        deltaSumF += updateDelta
        if deltaSumF >= fpsDelta: render frame; deltaSumF -= fpsDelta

Determinism: time advances only by fixed updateDelta; encoding speed never
affects output. MotionBlur (fps × OversampleMultiplier subframes blended
before readback, §4.13/§5.6) is a later phase; the loop shape already
accommodates it (raise fps, blend every N frames).

NO-BASS NOTE: the reference pushes 1 ms of BASS-decoded PCM per audio tick
(§5.6). We keep the audio ticks in the loop for semantic parity, but our
AudioMixer (record/audio.py) mixes the full track OFFLINE with numpy and
hands ffmpeg a wav — push_1ms() is a no-op counter by default. This is the
same model the in-house mania/catch/taiko engines ship.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from ..render import perf


class Player(Protocol):
    """What the loop drives (states.Player in the reference, §5.3)."""

    def update(self, delta_ms: float) -> bool:
        """Advance sim time by delta_ms. True at MapEnd."""
        ...

    def draw(self) -> "np.ndarray | list[np.ndarray]":
        """Render the current frame. Returns either one HxWx3 uint8
        frame (top-left origin) or an ORDERED list of ready frames — an
        async-readback player (PBO ring) returns [] while its pipeline
        fills and may also expose drain() for the tail at map end."""
        ...


def predict_draw_times(fps: int, start_ms: float, end_ms: float,
                       speed: float = 1.0, rate_fn=None):
    """Yield the map time of every draw() RecordPipeline.run() will issue for
    a fixed-timestep player (render.scene.ScenePlayer), WITHOUT rendering.

    Repeats the loop's own float arithmetic operation for operation, so the
    times are bit-equal to the ones the real loop reaches. Keep it in step
    with run() below and ScenePlayer.update(). Used to prebake the results
    outro off-thread; consumers must treat a mismatch as a cache miss, never
    as an error."""
    update_delta = 1000.0 / max(fps, 1000)
    fps_delta = 1000.0 / fps
    t = start_ms
    delta_sum_f = 0.0
    while True:
        if rate_fn is None:
            t += update_delta * speed
        else:
            t += update_delta * rate_fn(t)
        if t >= end_ms:
            return
        delta_sum_f += update_delta
        if delta_sum_f >= fps_delta:
            yield t
            delta_sum_f -= fps_delta


class RecordPipeline:
    """Fixed-timestep driver: Player → frame sink (+ audio tick)."""

    def __init__(self, fps: int,
                 push_frame: Callable[[np.ndarray], None],
                 push_audio_1ms: Callable[[], None] | None = None,
                 progress: Callable[[float], None] | None = None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.push_frame = push_frame
        self.push_audio_1ms = push_audio_1ms
        self.progress = progress

    def run(self, player: Player, total_ms: float | None = None) -> int:
        """Run to MapEnd. Returns the number of video frames pushed."""
        update_fps = max(self.fps, 1000)
        update_delta = 1000.0 / update_fps
        fps_delta = 1000.0 / self.fps
        audio_delta = 1.0

        delta_sum_a = 0.0
        delta_sum_f = 0.0
        elapsed = 0.0
        frames = 0

        # The sim ticks at >=1000 Hz (update_fps above) while frames land at
        # self.fps, so update() is called ~17x more often than draw(). That ratio
        # is why sim_update is timed separately: a cost of 20us per update is
        # invisible per-call but is the same order as a whole frame's render.
        perf.mark("loop_start")
        while True:
            with perf.T("sim_update"):
                done = player.update(update_delta)
            if done:
                break
            elapsed += update_delta

            if self.push_audio_1ms is not None:
                # Timed around the inner while, not inside it: at 1 kHz vs 1 ms
                # this drains ~1 push per iteration, so wrapping each push would
                # cost more in timer calls than the push itself.
                with perf.T("audio_push"):
                    delta_sum_a += update_delta
                    while delta_sum_a >= audio_delta:
                        self.push_audio_1ms()
                        delta_sum_a -= audio_delta

            delta_sum_f += update_delta
            if delta_sum_f >= fps_delta:
                with perf.T("draw"):
                    out = player.draw()
                if isinstance(out, np.ndarray):
                    with perf.T("push_frame"):
                        self.push_frame(out)
                    frames += 1
                else:
                    for f in out:      # pipelined player: 0..n ready frames
                        with perf.T("push_frame"):
                            self.push_frame(f)
                        frames += 1
                delta_sum_f -= fps_delta
                if self.progress is not None and total_ms:
                    with perf.T("progress"):
                        self.progress(min(1.0, elapsed / total_ms))
        perf.mark("loop_end")
        drain = getattr(player, "drain", None)
        if drain is not None:
            with perf.T("drain_tail"):
                for f in drain():      # PBO ring tail — order preserved
                    self.push_frame(f)
                    frames += 1
        perf.mark("drain_end")
        return frames
