"""ffmpeg raw-RGB pipe — RENDER_PLAN.md §5.6 semantics on our stack; adapted
from the mania v2 encoder (OsuManiaRenderer_v2/osu_mania_renderer_v2/
encode.py: encoder probing order, rawvideo stdin input shape) simplified to
the synchronous single-process model the catch/taiko engines use.

Differences from the reference (§5.6) — deliberate:
  * ONE ffmpeg process, video on stdin + audio as a FILE input, instead of
    danser's two processes + mux step. Our audio is premixed offline
    (record/audio.py — NO BASS), so there is nothing to stream in
    lockstep; ffmpeg muxes in the same invocation.
  * Pixel path is rgb24 (renderer reads back RGB); the GPU RGB→YUV shader
    + PBO pool (§5.6 readback) is a later perf phase — mania v2's
    gpu/readback.py already proves it on this stack.
  * loudnorm (single-pass, I=-18:TP=-1.5, the 2026-07-31 audio directive)
    is applied to the audio filter chain like every in-house engine.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..render import perf

LOUDNORM = "loudnorm=I=-18:TP=-1.5:LRA=11"

# ---- x264 knobs behind env hooks (parity row 16) ----------------------------
# taiko spells these R3D_X264_* and catch R3D_CATCH_X264_*; the un-prefixed form
# is the useful one for a fleet, so std takes that.
#
# THE DEFAULTS PRESERVE std's CURRENT OUTPUT EXACTLY (preset "faster", no extra
# x264-params, caller's crf). Adding the hooks is therefore byte-identical, which
# keeps "can we tune the encoder" a separate, measurable question from "did we
# silently change what we ship". taiko's tuned values are ultrafast +
# cabac=1:8x8dct=1:ref=2 -- reachable here without a code edit.
_X264_PRESET = os.environ.get("R3D_X264_PRESET", "faster")
_X264_CRF = os.environ.get("R3D_X264_CRF")       # None = use the caller's crf
_X264_PARAMS = os.environ.get("R3D_X264_PARAMS", "")
_X264_THREADS = os.environ.get("R3D_X264_THREADS")


def _x264_extra() -> list:
    out = []
    if _X264_PARAMS:
        out += ["-x264-params", _X264_PARAMS]
    if _X264_THREADS:
        out += ["-threads", _X264_THREADS]
    return out


class EncoderError(RuntimeError):
    pass


def probe_encoder(encoder: str = "auto") -> str:
    """Resolve 'auto' → preferred encoder available on this system.
    Preference: h264_nvenc → h264_vaapi → libx264 (pool A/B are NVENC;
    pool C is AMD/VAAPI with system ffmpeg)."""
    if encoder != "auto":
        return encoder
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise EncoderError("ffmpeg not found on PATH")
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                         capture_output=True, text=True, check=False).stdout
    for cand in ("h264_nvenc", "h264_vaapi", "libx264"):
        if cand in out:
            return cand
    return "libx264"


def nvenc_target_bps(w: int, h: int, fps: float) -> int:
    """Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy, 2026-07).

    Replaces the flat per-engine bitrate: scale a 4 Mbps 720p30 reference
    by pixel rate with a perceptual exponent (0.70 -- deliberately NOT
    linear), clamped to [2.5, 16] Mbps.  Anchors: 720p30=4.0M,
    720p60=6.5M, 1080p30=7.1M, 1080p60=11.5M, 1440p60/1080p120+=16M cap.
    Callers pair the target with maxrate=1.5x / bufsize=2x for NVENC VBR.
    Same formula in all four engines (catch/taiko/std/mania v2).
    """
    ref = 1280.0 * 720.0 * 30.0
    target = 4_000_000.0 * ((float(w) * float(h) * float(fps)) / ref) ** 0.70
    return int(min(16_000_000.0, max(2_500_000.0, target)))


def preview_video_bps(total_dur_s: "float | None") -> int:
    """Video bitrate of the lean preview embed. Mirrors the contributor
    client's makeEmbedVariant (and the bot's _transcode_embed_unbounded):
    ~1.4 Mbps, lowered on long maps so the file stays <= ~24 MiB, floor 500k.
    Same formula as the catch engine's inline preview."""
    vbps = 1_400_000
    if total_dur_s and total_dur_s > 0:
        vbps = int(24 * 1024 * 1024 * 8 / total_dur_s) - 128_000
        vbps = max(500_000, min(1_400_000, vbps))
    return vbps


def _null_sink_cmd() -> list[str]:
    """HARNESS ONLY (R3D_STD_NULL_SINK=1): swap ffmpeg for `cat` so the renderer
    runs with the encoder's cost REMOVED. The gap against a normal run is the
    encoder's share of wall, which is the budget every encoder-side optimisation
    (GPU-YUV, socketpair, mapped readback, preset) is competing over. Produces no
    usable output -- measurement only."""
    return ["cat"]


def build_ffmpeg_cmd(*, encoder: str, resolution: tuple[int, int], fps: int,
                     output_path: Path, audio_path: Path | None = None,
                     audio_offset_ms: int = 0, video_bitrate: int | None = None,
                     crf: int = 16, audio_bitrate: str = "192k",
                     loudnorm: bool = True, extra_vf: str = "",
                     encoder_device: str | None = None,
                     preview_path: Path | None = None,
                     total_dur_s: float | None = None,
                     pix_fmt: str = "rgb24",
                     stream_master: bool = False) -> list[str]:
    """rawvideo rgb24 on stdin → encoder → faststart mp4 (§5.6 shape).

    `preview_path` (INLINE PREVIEW, R3D_PREVIEW_INLINE=1 in the CLI; default
    None) makes the SAME ffmpeg process also write a lean 720p30 libx264
    preview embed as a second output. With it None the command is built
    exactly as before. `total_dur_s` (video length, if known) only sizes the
    preview's bitrate."""
    w, h = resolution
    is_vaapi = encoder == "h264_vaapi"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if is_vaapi:
        # VAAPI needs the DRM render node initialised before the encoder and the
        # frames uploaded to a GPU surface (format=nv12,hwupload below). Without
        # this ffmpeg cannot open h264_vaapi -> dies at startup -> BrokenPipe on
        # the first frame write. std was NVENC-only until AMD contributors ran it.
        cmd += ["-vaapi_device", encoder_device or "/dev/dri/renderD128"]
    # pix_fmt yuv420p = the renderer converted on the GPU (R3D_STD_GPU_YUV) and
    # swscale has nothing left to do; vflip below is format-agnostic.
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "pipe:0"]
    if audio_path is not None:
        if audio_offset_ms:
            cmd += ["-itsoffset", f"{audio_offset_ms / 1000.0:.3f}"]
        cmd += ["-i", str(audio_path)]
    # Frames arrive BOTTOM-UP on stdin and ffmpeg's vflip filter restores
    # them (an exact row reorder on rawvideo — zero pixel math; the mania
    # v2 prod encoder ships the same shape). This lets the writer hand
    # the GL readback buffer to the pipe zero-copy instead of paying a
    # ~6 MB negative-stride flip copy per frame — see FfmpegPipe._writer.
    _vf = "vflip" + ("," + extra_vf if extra_vf else "")
    # master-only tail of the video chain (VAAPI surface upload). Kept apart
    # from `_vf` so the inline-preview graph can apply it on the master
    # branch only; without a preview it is appended to `-vf` as before.
    _vm_tail = "format=nv12,hwupload" if is_vaapi else ""
    # video codec args are collected in `vc` (appended below) so the
    # two-output preview command can place them after its own -map.
    vc: list[str] = ["-c:v", encoder]
    if encoder == "libx264":
        if video_bitrate:
            _vb = int(video_bitrate)
            vc += ["-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                    "-bufsize", str(_vb * 2), "-preset", _X264_PRESET,
                    "-profile:v", "high"] + _x264_extra()
        else:
            vc += ["-crf", _X264_CRF or str(crf), "-preset", _X264_PRESET,
                    "-profile:v", "high"] + _x264_extra()
    elif encoder in ("h264_nvenc", "hevc_nvenc"):
        # Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy,
        # 2026-07): replaces the 2026-07-21 CQ scheme (cq=crf+4, -b:v 0,
        # maxrate (w*h)/150k) with the shared target-VBR ladder so all four
        # engines land on the same size/quality curve -- see nvenc_target_bps.
        _tgt = video_bitrate or nvenc_target_bps(w, h, fps)
        vc += ["-rc", "vbr", "-b:v", str(_tgt),
                "-maxrate", str(int(_tgt * 1.5)), "-bufsize", str(_tgt * 2),
                "-profile:v", "high"]
    elif is_vaapi:
        _vb = video_bitrate or nvenc_target_bps(w, h, fps)
        vc += ["-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                "-bufsize", str(_vb * 2)]
    elif video_bitrate:
        vc += ["-b:v", str(video_bitrate)]
    if not is_vaapi:
        # VAAPI output pixel format is set by the hwupload filtergraph (nv12 on a
        # GPU surface); forcing -pix_fmt yuv420p here conflicts with the encoder.
        vc += ["-pix_fmt", "yuv420p"]
    af: list[str] = []
    acodec: list[str] = []
    if audio_path is not None:
        # LOUDNORM DUCK FIX (#17): when the music was pre-normalised upstream
        # (loudnorm=False), do NOT loudnorm the mixed song+hits (that ducked the
        # song under hits) -- apply only a clamp-only true-peak limiter to catch
        # summed peaks without ducking.
        af = ([LOUDNORM] if loudnorm
              else ["alimiter=limit=0.95:level=disabled:attack=1:release=20"])
        # STREAMABLE MASTER (R3D_STREAM_MASTER=1 in the CLI; default False):
        # the loudness pass the contributor client would run on the finished
        # file (loudnorm on the MIXED output) is applied here instead, so
        # nothing has to rewrite the master after the render.
        if stream_master and LOUDNORM not in af:
            af = af + [LOUDNORM]
        acodec = ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
                  "-shortest"]
    _mfast = [] if stream_master else ["-movflags", "+faststart"]
    if preview_path is None:
        cmd += ["-vf", _vf + ("," + _vm_tail if _vm_tail else "")]
        cmd += vc
        if audio_path is not None:
            if af:
                cmd += ["-af", ",".join(af)]
            cmd += acodec
        # stream_master: no +faststart, which rewrites the whole file at close;
        # the master is then written front to back and the site moves the moov.
        cmd += _mfast + [str(output_path)]
        if os.environ.get("R3D_STD_NULL_SINK") == "1":
            return _null_sink_cmd()
        return cmd

    # TWO OUTPUTS FROM ONE PROCESS (inline preview). The frame pipe is read
    # once. The master's own video chain (`vflip` + extra_vf — frames arrive
    # BOTTOM-UP) runs BEFORE `split`, so both outputs are the right way up and
    # the master sees the same frames as with `-vf`; `split` then hands them
    # to the master encoder (unchanged settings; the VAAPI upload stays on
    # the master branch only) and to a 720p30 libx264 preview. Audio: the
    # master's own `-af` chain stays on the shared branch, then `asplit`; the
    # preview branch gets the loudness pass the contributor client would
    # otherwise apply before cutting its embed, so the preview needs no
    # post-processing at all.
    pfps = min(30, int(round(float(fps))))
    graph = [f"[0:v]{_vf},split=2[vm0][vp0];[vm0]{_vm_tail or 'null'}[vm];"
             f"[vp0]scale=-2:720,fps={pfps}[vp]"]
    if audio_path is not None:
        # the audio file is input 1 (input 0 is the rawvideo pipe).
        # `aformat=sample_rates=48000` PINS the shared branch to the master's
        # own output rate (its `-ar 48000`; the mixed wav is 48 kHz too, so
        # this converts nothing). Without it the preview's loudnorm — which
        # runs at 192 kHz internally — wins format negotiation back through
        # `asplit`, the master's limiter then runs at 192 kHz and the master
        # audio is resampled 48k→192k→48k: measurably NOT the bytes the
        # `-af` path produces (ffmpeg 8.1). Pinned, the 192 kHz conversion
        # sits on the preview branch only and the master is byte-identical.
        graph.append(f"[1:a]{','.join(af) or 'anull'},"
                     f"aformat=sample_rates=48000[aout]")
        if stream_master:
            # the shared branch already carries the loudness pass (af above):
            # master and preview get the same normalised audio
            graph.append("[aout]asplit=2[am][ap]")
        else:
            graph.append(f"[aout]asplit=2[am][ap0];[ap0]{LOUDNORM}[ap]")
    cmd += ["-filter_complex", ";".join(graph)]
    # output 1: the master, exactly as without the preview
    cmd += ["-map", "[vm]"] + (["-map", "[am]"] if audio_path is not None
                               else [])
    cmd += vc + acodec
    cmd += _mfast + [str(output_path)]
    # output 2: the preview. libx264 on every node, deliberately: a second
    # NVENC/VAAPI session can fail to open (session limits), and one failed
    # output kills the whole process and with it the render.
    vbps = preview_video_bps(total_dur_s)
    cmd += ["-map", "[vp]"] + (["-map", "[ap]"] if audio_path is not None
                               else [])
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-b:v", str(vbps), "-maxrate", str(int(vbps * 1.25)),
            "-bufsize", str(vbps * 2), "-g", "30",
            "-threads", str(max(2, min(4, (os.cpu_count() or 4) - 2)))]
    if audio_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-shortest"]
    cmd += ["-movflags", "+faststart", str(preview_path)]
    if os.environ.get("R3D_STD_NULL_SINK") == "1":
        return _null_sink_cmd()
    return cmd


class _SockStdin:
    """File-like shim over the socketpair end.

    `sendall`, not `write`, is deliberate: a raw SocketIO.write may write
    PARTIALLY and return a short count, and the writer thread ignores write()'s
    return value -- that would silently truncate a frame."""

    __slots__ = ("_sk",)

    def __init__(self, sk):
        self._sk = sk

    def write(self, b):
        self._sk.sendall(b)
        return len(b)

    def flush(self):
        pass

    def fileno(self):
        return self._sk.fileno()

    def close(self):
        import socket as _sock
        try:
            self._sk.shutdown(_sock.SHUT_WR)
        except OSError:
            pass
        self._sk.close()


class FfmpegPipe:
    """Spawn ffmpeg, push raw frames, close. Mirrors mania v2's FfmpegPipe
    contract minus asyncio (the worker wraps the CLI in a subprocess
    already; in-process async buys nothing here).

    Frames are handed to a writer thread over a small bounded queue: the
    serialisation (`tobytes` — a negative-stride flip copy) and the
    blocking pipe write happen OFF the render thread, overlapping the next
    frame's draw. Order is FIFO so the byte stream ffmpeg sees (and the
    R3D_FRAME_MD5 hash, computed writer-side) is unchanged. The queue
    bounds memory (4 × ~2.7 MB frames) and provides natural backpressure
    when ffmpeg is the bottleneck; writer errors surface loudly on the
    next push() instead of deadlocking the producer."""

    _QUEUE_FRAMES = 4

    def __init__(self, cmd: list[str], recycle=None):
        # recycle: optional callable(frame) invoked writer-side once a
        # frame's bytes are in the pipe — the GL renderer's readback
        # buffer pool (SpriteRenderer.recycle_frame). Pure bookkeeping:
        # the byte stream ffmpeg sees is untouched.
        self.cmd = cmd
        self._recycle = recycle
        self.proc: subprocess.Popen | None = None
        self._q: "queue.Queue" = queue.Queue(maxsize=self._QUEUE_FRAMES)
        self._thread: threading.Thread | None = None
        self._werr: BaseException | None = None
        self._hash = None
        self._hash_frames = 0
        if perf.FRAME_MD5:
            import hashlib
            self._hash = hashlib.blake2b(digest_size=16)

    def __enter__(self) -> "FfmpegPipe":
        # macOS pushes every frame through the 64 KiB default pipe -- ~95 kernel
        # handoffs for ONE 6.22 MB rgb24 frame -- because F_SETPIPE_SZ is
        # Linux-only. A unix socketpair CAN be grown (SO_SNDBUF/SO_RCVBUF).
        # Catch measured pipe 256 fps -> socketpair 406 fps on this machine
        # against a 419 fps file-fed ceiling. Bytes on the wire are unchanged,
        # so output is byte-identical.
        if sys.platform == "darwin" and os.environ.get("R3D_MAC_SOCKET_PIPE") == "1":
            import socket as _sock
            _par, _chi = _sock.socketpair(_sock.AF_UNIX, _sock.SOCK_STREAM)
            for _s, _opt in ((_par, _sock.SO_SNDBUF), (_chi, _sock.SO_RCVBUF)):
                try:
                    _s.setsockopt(_sock.SOL_SOCKET, _opt, 1 << 20)
                except OSError:
                    pass          # keep the default buffer; still correct
            self.proc = subprocess.Popen(
                self.cmd, stdin=_chi.fileno(), stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, bufsize=0)
            _chi.close()
            self.proc.stdin = _SockStdin(_par)     # type: ignore[assignment]
        else:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                import fcntl
                # F_SETPIPE_SZ = 1031 (Linux); same intent as the socketpair
                fcntl.fcntl(self.proc.stdin.fileno(), 1031, 1 << 20)
            except (OSError, ImportError, AttributeError):
                pass              # not Linux, or not permitted -> default size
        self._thread = threading.Thread(target=self._writer,
                                        name="ffmpeg-writer", daemon=True)
        self._thread.start()
        return self

    def _writer(self) -> None:
        stdin = self.proc.stdin
        while True:
            frame = self._q.get()
            if frame is None:
                return
            if self._werr is not None:
                if self._recycle is not None and not hasattr(frame, "result"):
                    self._recycle(frame)
                continue          # drain (never write after an error)
            try:
                if hasattr(frame, "result"):
                    # a deferred frame (concurrent.futures.Future, e.g. the
                    # sub-1080p results outro's downscale pool). Resolved HERE,
                    # in queue order, so the stream stays FIFO.
                    frame = frame.result()
                if self._hash is not None:
                    # R3D_FRAME_MD5 hashes the TOP-DOWN semantic stream —
                    # the same bytes the pre-vflip pipe pushed, so digests
                    # stay comparable across the pipeline change.
                    self._hash.update(frame.tobytes())
                    self._hash_frames += 1
                # ffmpeg runs `vflip` (build_ffmpeg_cmd), so the pipe wants
                # the frame BOTTOM-UP. PBO frames are flipud views over a
                # contiguous readback buffer, so frame[::-1] recovers that
                # buffer C-contiguous → a zero-copy pipe write. Fresh
                # top-down frames (the SSAA results tail) fall back to the
                # same negative-stride flip copy the old path did.
                if frame.ndim == 1:
                    # planar yuv420p: already in GL (bottom-up) row order and
                    # contiguous, and ffmpeg's vflip reorders the rows. Reversing
                    # a FLAT planar buffer would reverse every byte and scramble
                    # the three planes into each other.
                    stdin.write(frame)
                else:
                    flipped = frame[::-1]
                    if flipped.flags["C_CONTIGUOUS"]:
                        stdin.write(flipped)
                    else:
                        stdin.write(flipped.tobytes())
                if self._recycle is not None:
                    self._recycle(frame)
            except BaseException as e:  # noqa: BLE001 - surfaced on push()
                self._werr = e

    def push(self, frame_rgb) -> None:
        assert self.proc is not None and self._thread is not None
        if self._werr is not None:
            raise EncoderError(f"ffmpeg writer failed: {self._werr!r}")
        with perf.T("encode_push"):
            self._q.put(frame_rgb)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is None:
            return
        # The queue still holds frames the writer has not handed to ffmpeg, so
        # this join is real encode time, not teardown. It is invisible to any
        # frames/second figure because it happens after the last push.
        perf.mark("enc:queue_drain")
        if self._thread is not None:
            with perf.T("enc_writer_join"):
                self._q.put(None)
                self._thread.join()
        perf.mark("enc:writer_joined")
        if self._hash is not None:
            print(f"frame-stream-hash: {self._hash.hexdigest()} "
                  f"({self._hash_frames} frames)", file=sys.stderr, flush=True)
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        _, err = None, b""
        # Named, not left as a residual: this is ffmpeg's OWN remaining encode
        # time (lookahead flush + trailer), real wall cost that no frames/second
        # figure can show because it happens after the last frame is handed over.
        with perf.T("ffmpeg_finish"):
            try:
                err = self.proc.stderr.read() if self.proc.stderr else b""
            finally:
                code = self.proc.wait()
        perf.mark("enc:ffmpeg_reaped")
        if exc_type is None and self._werr is not None \
                and not isinstance(self._werr, BrokenPipeError):
            raise EncoderError(f"ffmpeg writer failed: {self._werr!r}")
        if exc_type is None and code != 0:
            raise EncoderError(
                f"ffmpeg exited {code}: {err.decode(errors='replace')[-2000:]}")
