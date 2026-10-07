"""The inline preview on the Mac's media engine (record/encode.py).

What must hold:
  * the master's ffmpeg arguments do not depend on which encoder makes the
    preview, so the master cannot change;
  * the default is the CPU preview unless the caller passes preview_hw;
  * asking is not getting: R3D_PREVIEW_VT and the platform default decide what
    is wanted, the probe decides what is used;
  * an ffmpeg failure that names videotoolbox switches the media engine off for
    the next render, and nothing else does;
  * where a hardware session really opens, the real two-output command makes a
    master and a preview of the right length.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from osu_std_renderer.record import encode
from osu_std_renderer.render import perf


def _cmd(**kw):
    base = dict(encoder="libx264", resolution=(1280, 720), fps=60,
                output_path="m.mp4", preview_path="m.embed.mp4",
                total_dur_s=60.0, pix_fmt="yuv420p")
    base.update(kw)
    return encode.build_ffmpeg_cmd(**base)


def test_master_arguments_do_not_depend_on_the_preview_encoder():
    for extra in ({}, {"stream_master": True}, {"compact_path": "m-embed-sm.mp4"},
                  {"audio_path": "a.wav"}):
        cpu, hw = _cmd(**extra), _cmd(preview_hw=True, **extra)
        i, j = cpu.index("m.mp4"), hw.index("m.mp4")
        # the one difference allowed before the master's output path: the
        # lead-in filter at the end of the PREVIEW branch of the graph
        g = hw.index("-filter_complex") + 1
        lead = "," + encode.vt_lead_in_filter(30)
        assert hw[g].count(lead + "[vp]") == 1 and hw[g].count("tpad") == 1
        graph = hw[g].replace(lead, "")
        if "audio_path" in extra:
            # ... and, with audio, the preview's audio ended at the video's known length
            end = f";[ap_full]atrim=end={60.0:.6f}[ap]"
            assert graph.endswith(end) and graph.count("[ap_full]") == 2
            graph = graph[:-len(end)].replace("[ap_full]", "[ap]")
        same = hw[:g] + [graph] + hw[g + 1:j + 1]
        assert cpu[:i + 1] == same, f"master part differs with {extra}"
        assert "h264_videotoolbox" not in cpu
        assert hw.count("h264_videotoolbox") == 1


def test_shortest_is_not_used_on_a_preview_with_a_lead_in():
    # it compares the streams before the lead-in is cut off: with the audio ended
    # at the video's length it would cut the last half second of video
    cpu, hw = _cmd(audio_path="a.wav"), _cmd(audio_path="a.wav", preview_hw=True)
    tail = lambda c: c[c.index("m.mp4") + 1:]
    assert "-shortest" in tail(cpu) and "-shortest" not in tail(hw)
    assert "-shortest" in hw[:hw.index("m.mp4")]            # the master keeps it


def test_no_known_length_with_audio_keeps_the_cpu_preview():
    # the lead-in needs the video's length to end the preview's audio at
    cmd = _cmd(preview_hw=True, audio_path="a.wav", total_dur_s=None)
    assert "h264_videotoolbox" not in cmd and "tpad" not in cmd[cmd.index("-filter_complex") + 1]
    assert "h264_videotoolbox" in _cmd(preview_hw=True, total_dur_s=None)      # video only: nothing to end


def test_default_is_the_cpu_preview():
    cmd = _cmd()
    assert "h264_videotoolbox" not in cmd
    assert cmd.count("libx264") == 2            # master and preview


def test_hw_preview_arguments():
    cmd = _cmd(preview_hw=True, compact_path="m-embed-sm.mp4")
    i = cmd.index("h264_videotoolbox")
    tail = cmd[i:cmd.index("m.embed.mp4")]
    vbps = encode.preview_video_bps(60.0)
    assert tail[tail.index("-b:v") + 1] == str(int(vbps * encode._VT_RATE))
    assert tail[tail.index("-allow_sw") + 1] == "0"     # never Apple's software encoder
    assert tail[tail.index("-constant_bit_rate") + 1] == "1" and "-maxrate" not in tail
    # the lead-in: half a second of the first frame in front, a keyframe forced on
    # the first real frame, and the lead-in's packets dropped and the rest shifted back
    n = encode._vt_lead_frames(30)
    assert n == 15
    assert tail[tail.index("-force_key_frames") + 1] == f"expr:eq(n,{n})"
    bsf = tail[tail.index("-bsf:v") + 1]
    assert bsf.startswith("noise=drop=lt(pts*tb") and f"setts=pts=PTS-{n}/30/TB:dts=DTS-{n}/30/TB" in bsf
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert f"scale=-2:720,tpad=start={n}:start_mode=clone[vp]" in graph
    assert "tpad" not in _cmd(compact_path="m-embed-sm.mp4")[_cmd(compact_path="m-embed-sm.mp4").index("-filter_complex") + 1]
    assert tail[tail.index("-g") + 1] == "30"           # a keyframe every second, as before
    assert "-threads" not in tail
    # the Discord copy is a deliverable: it stays on x264
    k = cmd.index("m.embed.mp4")
    assert "libx264" in cmd[k:] and "h264_videotoolbox" not in cmd[k:]


def test_the_live_preview_sink_is_the_same_for_both_encoders():
    d = tempfile.mkdtemp()
    old = os.environ.get("R3D_PREVIEW_LIVE")
    os.environ["R3D_PREVIEW_LIVE"] = "1"
    try:
        p = os.path.join(d, "m.embed.mp4")
        cpu, hw = _cmd(preview_path=p), _cmd(preview_path=p, preview_hw=True)
        assert cpu[cpu.index("-f", cpu.index("libx264", cpu.index("m.mp4"))):] \
            == hw[hw.index("-f", hw.index("h264_videotoolbox")):]
        assert hw[-1].endswith("live.m3u8")
    finally:
        if old is None:
            os.environ.pop("R3D_PREVIEW_LIVE", None)
        else:
            os.environ["R3D_PREVIEW_LIVE"] = old
        shutil.rmtree(d, ignore_errors=True)


def _decide(env, fast_default, probe_says):
    saved = (os.environ.get("R3D_PREVIEW_VT"), perf.FAST_DEFAULT, encode._vt_probe,
             encode._vt_decision)
    asked = []

    def fake_probe(ignore_off=False):
        asked.append(ignore_off)
        return probe_says
    try:
        if env is None:
            os.environ.pop("R3D_PREVIEW_VT", None)
        else:
            os.environ["R3D_PREVIEW_VT"] = env
        perf.FAST_DEFAULT = fast_default
        encode._vt_probe = fake_probe
        encode._vt_decision = None
        return encode.preview_on_media_engine(), asked
    finally:
        if saved[0] is None:
            os.environ.pop("R3D_PREVIEW_VT", None)
        else:
            os.environ["R3D_PREVIEW_VT"] = saved[0]
        perf.FAST_DEFAULT, encode._vt_probe, encode._vt_decision = saved[1:]


def test_decision_rule():
    assert _decide(None, True, True) == (True, [False])     # a Mac, session opens
    assert _decide(None, True, False)[0] is False           # a Mac, no session
    assert _decide(None, False, True) == (False, [])        # not a Mac / stock: never asked
    assert _decide("0", True, True) == (False, [])          # switched off by hand
    assert _decide("1", False, True) == (True, [True])      # asked for anywhere; overrides the day off
    assert _decide("1", False, False)[0] is False           # asking is not getting


def test_only_a_failure_naming_videotoolbox_switches_it_off():
    d = tempfile.mkdtemp()
    saved = encode._r3d_cache_dir
    encode._r3d_cache_dir = lambda: d
    try:
        hw, cpu = _cmd(preview_hw=True), _cmd()
        mark = os.path.join(d, "vt-preview-off")
        assert not encode.note_preview_failure(cpu, b"[h264_videotoolbox @ 0x1] Error")
        assert not encode.note_preview_failure(hw, b"No space left on device")
        assert not os.path.exists(mark)
        assert encode.note_preview_failure(
            hw, b"[h264_videotoolbox @ 0x1] Error: cannot create compression session: -12908")
        assert os.path.exists(mark)
        assert encode._vt_probe() is False                  # off for the day, without probing
        with open(mark, "w") as fh:
            fh.write("1")                                   # the day is over
        # whatever the answer is on this box, it is now the probe's, not the marker's
        assert encode._vt_probe() == encode._vt_probe(ignore_off=True)
    finally:
        encode._r3d_cache_dir = saved
        shutil.rmtree(d, ignore_errors=True)


def test_real_two_output_encode_where_a_session_opens():
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    d = tempfile.mkdtemp()
    saved = encode._r3d_cache_dir
    encode._r3d_cache_dir = lambda: d                      # the real probe, a private cache
    try:
        if not ff or not fp or not encode._vt_probe():
            print("SKIP (no ffmpeg with a hardware H.264 session here)")
            return
        # the probe asks for exactly what a render asks for
        assert encode._vt_codec_args(300_000)[:1] == ["-c:v"] and "-constant_bit_rate" in encode._vt_codec_args(1)
        w, h, n = 320, 180, 120
        out, prev = os.path.join(d, "m.mp4"), os.path.join(d, "m.embed.mp4")
        cmd = encode.build_ffmpeg_cmd(encoder="libx264", resolution=(w, h), fps=60, output_path=out,
                                      preview_path=prev, total_dur_s=n / 60.0, pix_fmt="rgb24",
                                      preview_hw=True)
        rng = np.random.default_rng(3)
        base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        frames = b"".join(np.roll(base, 3 * i, axis=1).tobytes() for i in range(n))
        p = subprocess.run(cmd, input=frames, capture_output=True, cwd=d, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]

        def packets(f):
            j = json.loads(subprocess.run(
                [fp, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                 "stream=codec_name,nb_read_packets,height", "-of", "json", f],
                capture_output=True, text=True).stdout)["streams"][0]
            return j["codec_name"], int(j["nb_read_packets"]), int(j["height"])
        assert packets(out) == ("h264", n, h)
        codec, pk, ph = packets(prev)
        assert codec == "h264" and ph == 720 and abs(pk - n // 2) <= 1, (codec, pk, ph)
        # the lead-in is gone: the file starts on a keyframe at time 0 and decodes
        # to exactly as many frames as it has packets, with nothing to conceal
        first = sorted((float(p["pts_time"]), p["flags"]) for p in json.loads(subprocess.run(
            [fp, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
             "-of", "json", prev], capture_output=True, text=True).stdout)["packets"])[0]
        assert abs(first[0]) < 1e-6 and "K" in first[1], first
        dec = subprocess.run([ff, "-v", "error", "-i", prev, "-map", "0:v", "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], capture_output=True)
        assert dec.returncode == 0 and not dec.stderr.strip(), dec.stderr[-200:]
        assert len(dec.stdout) == pk * 1280 * 720, (len(dec.stdout) / (1280 * 720), pk)
    finally:
        encode._r3d_cache_dir = saved
        shutil.rmtree(d, ignore_errors=True)


def test_real_encode_with_audio_keeps_every_frame_and_ends_the_audio_with_the_video():
    """The lead-in lengthens the video's timeline until it is cut off, and the
    audio is not lengthened with it. Both ways of getting that wrong are silent:
    half a second of extra audio, or the last half second of video cut."""
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    d = tempfile.mkdtemp()
    saved = encode._r3d_cache_dir
    encode._r3d_cache_dir = lambda: d
    try:
        if not ff or not fp or not encode._vt_probe():
            print("SKIP (no ffmpeg with a hardware H.264 session here)")
            return
        w, h, n = 320, 180, 240                       # 4 s at 60 fps
        wav = os.path.join(d, "a.wav")
        subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=7",
                        "-ac", "2", wav], check=True)          # longer than the video, as a real mix is
        out, prev = os.path.join(d, "m.mp4"), os.path.join(d, "m.embed.mp4")
        cmd = encode.build_ffmpeg_cmd(encoder="libx264", resolution=(w, h), fps=60, output_path=out,
                                      audio_path=wav, loudnorm=False, preview_path=prev,
                                      total_dur_s=n / 60.0, pix_fmt="rgb24", preview_hw=True)
        base = np.random.default_rng(5).integers(0, 256, (h, w, 3), dtype=np.uint8)
        frames = b"".join(np.roll(base, 3 * i, axis=1).tobytes() for i in range(n))
        p = subprocess.run(cmd, input=frames, capture_output=True, cwd=d, timeout=120)
        assert p.returncode == 0, p.stderr.decode(errors="replace")[-400:]
        j = json.loads(subprocess.run([fp, "-v", "error", "-count_packets", "-show_entries",
                                       "stream=codec_type,nb_read_packets,start_time,duration", "-of", "json", prev],
                                      capture_output=True, text=True).stdout)["streams"]
        v = next(s for s in j if s["codec_type"] == "video")
        a = next(s for s in j if s["codec_type"] == "audio")
        assert int(v["nb_read_packets"]) == n // 2, v                    # every frame, none cut at the end
        assert abs(float(v["start_time"])) < 0.05 and abs(float(a["start_time"])) < 0.05, (v, a)
        assert abs(float(a["duration"]) - n / 60.0) < 0.06, a            # the audio ends with the video
    finally:
        encode._r3d_cache_dir = saved
        shutil.rmtree(d, ignore_errors=True)
