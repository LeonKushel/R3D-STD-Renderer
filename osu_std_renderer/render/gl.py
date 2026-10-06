"""Minimal moderngl sprite batch — adapted from the production catch
renderer (osu-catch/osu_catch_renderer/gl.py), context creation split into
context.py. Draws textured/solid quads with straight-alpha blending in
painter's order plus an additive pass (hit explosions / glow), then reads
back tightly-packed RGB24 for the ffmpeg pipe.

Batched draw (perf-optimize): all sprite parameters ride per-vertex
attributes in ONE dynamic VBO built per draw() call, and consecutive
sprites sharing a texture collapse into a single indexed glDrawElements —
~444 draw calls/frame became ~40 (med-map profile). The vertex/fragment
math is expression-identical to the per-sprite uniform path it replaces
(same rotate/NDC lines, `flat` colour so no interpolation), so the raster
output is bit-identical; the per-frame blake2b frame-stream hash proved it
on the med benchmark map.

The mania v2 gpu/ package (atlas, texture arrays, PBO readback) remains
the performance end-state; texture-atlas packing is deliberately NOT done
here — mipmapped LINEAR sampling at atlas edges cannot be proven
pixel-identical against per-texture repeat wrapping.
"""
from __future__ import annotations

import os
import sys
from array import array as _array
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import moderngl
except Exception as e:  # noqa: BLE001
    raise RuntimeError("moderngl is required for the std renderer") from e

from . import perf
from .context import create_context

# R3D_STD_GPU_YUV: convert RGB -> yuv420p ON THE GPU and read back 1.5 bytes/px
# instead of 3, feeding ffmpeg `-pix_fmt yuv420p` so swscale converts nothing.
# Ported from taiko (fleet-validated max|d|=0 vs swscale, +42.7% with a real
# encoder); encode.py's docstring had already flagged this as "a later perf phase".
#
# THE CONVERSION IS BIT-EXACT vs swscale, and the formula is NOT the one in
# libswscale's C source. ffmpeg's BGR24 fast path (`ff_rgb24toyv12`) truncates the
# 2x2 average BEFORE the matrix: rx = (r11+r12+r21+r22) >> 2 then >> 15. The
# shipped arm64 NEON build does NOT -- it sums and shifts ONCE (>> 17), keeping the
# extra precision. Matching the source gives 83-88% exact; matching the binary
# gives max|d|=0. Read the source for the shape, measure the binary for the
# arithmetic.
#
# NOTE it is a LOSS against a null sink because its whole win is relieving encoder
# back-pressure -- benchmark with a REAL encoder or the conclusion inverts.
_GPU_YUV = perf.envflag("R3D_STD_GPU_YUV")

# R3D_STD_MAP_READBACK: hand the writer a pointer INTO the pixel-pack buffer
# (glMapBufferRange) instead of copying the PBO into a numpy array. Parity row 7.
# Live on std specifically because std has no composite thread -- that is what made
# the same port measure ZERO on catch, where the render thread was already idling.
_MAP_READBACK = perf.envflag("R3D_STD_MAP_READBACK")
# ABLATIONS, MEASUREMENT ONLY -- they do not preserve output. They exist to size
# the draw-call win BEFORE building an atlas (row 11), because gl_sprites_draw
# also covers vertex serialisation and the VBO write, so the call count cannot be
# priced by extrapolation. NOMERGE = one draw per quad (the ceiling). ONETEX =
# every sprite on one texture, i.e. the minimum achievable call count (the floor).
_ABL_NOMERGE = perf.envflag("R3D_STD_ABL_NOMERGE")
_ABL_ONETEX = perf.envflag("R3D_STD_ABL_ONETEX")
# MEASUREMENT ONLY (wrong pixels): price the per-sprite Python serialisation.
# NOSER reuses a stale params block instead of running np.fromiter over 13
# scalars per sprite; NOVBO additionally skips the vertex fill + VBO upload.
# The fps delta vs a normal run is the CEILING for any faster serialiser.
_ABL_NOSER = perf.envflag("R3D_STD_ABL_NOSER")
_ABL_NOVBO = perf.envflag("R3D_STD_ABL_NOVBO")
_abl_params: dict = {}
# R3D_STD_SER_ARRAY (default OFF): serialise the per-sprite params through a
# C-level array('f').extend of one tuple per sprite instead of np.fromiter over
# a generator that yields 13 scalars per sprite (13 generator resumes each).
# Same values, same double->float32 cast, so the frame stream is unchanged.
_SER_ARRAY = perf.envflag("R3D_STD_SER_ARRAY")
# Readback LATENCY in frames, decoupled from pool size (they are the same number in
# the unmapped path). Taiko measured latency itself as flat from 3 to 16, so this
# stays at std's historical 3; it exists so the pool can grow without adding delay.
_PBO_LAT = max(1, int(os.environ.get("R3D_STD_PBO_LAT", "3")))
# MUST match record/encode.py FfmpegPipe._QUEUE_FRAMES. Not imported: encode.py
# imports this module and the cycle would break the build.
_WRITER_QUEUE_FRAMES = 4
# Slack above the holder budget. Catch showed that a pool which is merely "big
# enough" corrupts output SILENTLY the moment anything holds a frame a beat longer
# than assumed -- and a mapped slot that gets reused early is exactly that bug.
_PBO_MARGIN = 3

_GL_PIXEL_PACK_BUFFER = 0x88EB
_GL_MAP_READ_BIT = 0x0001
_GL_MAP_WRITE_BIT = 0x0002
_GL_READ_FRAMEBUFFER = 0x8CA8
_GL_DRAW_FRAMEBUFFER = 0x8CA9
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_NEAREST = 0x2600
_gl_c = None


def _load_gl_c():
    """ctypes handle on the already-loaded GL, for the calls moderngl lacks.

    Platform-aware on purpose: taiko's copy of this hardcodes the macOS framework
    path, and std also runs on the Linux render boxes."""
    global _gl_c
    if _gl_c is None:
        import ctypes
        import ctypes.util
        if sys.platform == "darwin":
            names = ["/System/Library/Frameworks/OpenGL.framework/OpenGL"]
        else:
            names = ["libGL.so.1", "libGL.so"]
            found = ctypes.util.find_library("GL")
            if found:
                names.insert(0, found)
        lib = None
        for n in names:
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError(f"could not load GL from {names}")
        lib.glBindBuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glBindBuffer.restype = None
        lib.glMapBufferRange.argtypes = [ctypes.c_uint, ctypes.c_ssize_t,
                                         ctypes.c_ssize_t, ctypes.c_uint]
        lib.glMapBufferRange.restype = ctypes.c_void_p
        lib.glUnmapBuffer.argtypes = [ctypes.c_uint]
        lib.glUnmapBuffer.restype = ctypes.c_ubyte
        lib.glBindFramebuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glBindFramebuffer.restype = None
        lib.glBlitFramebuffer.argtypes = [ctypes.c_int] * 8 + [ctypes.c_uint,
                                                               ctypes.c_uint]
        lib.glBlitFramebuffer.restype = None
        _gl_c = lib
    return _gl_c
_RGB2YUV_SHIFT = 15


def _yuv_coef(k, scale):
    """swscale's own coefficient derivation (utils.c:694). int() TRUNCATES after
    the +0.5 -- it is not a round, and a 1-LSB error here shows up as thousands of
    differing pixels that look like a rounding bug."""
    return int(k * scale / 255.0 * (1 << _RGB2YUV_SHIFT) + 0.5)


def rgb_to_yuv420p(rgb):
    """CPU twin of the GPU conversion, bit-identical to it and to swscale.

    Not on the render path -- it is the ORACLE the GPU shader is validated
    against, which is what chains the shader to swscale-exactness.

    `rgb` is (h, w, 3) uint8. Returns a flat uint8 yuv420p buffer (Y | U | V).
    int64 accumulation on purpose: the >> must be an arithmetic shift on a signed
    type, and numpy's uint promotion rules make that easy to get subtly wrong."""
    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    S = _RGB2YUV_SHIFT
    RY, GY, BY = (_yuv_coef(k, 219) for k in (0.299, 0.587, 0.114))
    RU, GU, BU = -_yuv_coef(0.169, 224), -_yuv_coef(0.331, 224), _yuv_coef(0.500, 224)
    RV, GV, BV = _yuv_coef(0.500, 224), -_yuv_coef(0.419, 224), -_yuv_coef(0.081, 224)
    y = (((RY * r + GY * g + BY * b) >> S) + 16).astype(np.uint8)

    def s4(a):
        return a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]

    sr, sg, sb = s4(r), s4(g), s4(b)
    # >> (S+2), NOT an averaged RGB then >> S -- see _GPU_YUV above.
    u = (((RU * sr + GU * sg + BU * sb) >> (S + 2)) + 128).astype(np.uint8)
    v = (((RV * sr + GV * sg + BV * sb) >> (S + 2)) + 128).astype(np.uint8)
    out = np.empty(w * h * 3 // 2, np.uint8)
    out[:w * h] = y.ravel()
    out[w * h:w * h + u.size] = u.ravel()
    out[w * h + u.size:] = v.ravel()
    return out

_VERT = """
#version 330
in vec2 in_pos;      // unit quad corner [-0.5,0.5]
in vec2 in_uv;
in vec2 in_center;   // sprite center in px (origin top-left)
in vec2 in_size;     // sprite w,h in px
in float in_rot;     // radians
in vec4 in_color;
in vec2 in_uv_off;   // texture sub-rect (spinner-metre reveal)
in vec2 in_uv_scale;
uniform vec2 u_screen;   // (w, h) in px
out vec2 v_uv;
flat out vec4 v_color;
void main() {
    vec2 p = in_pos * in_size;
    float c = cos(in_rot), s = sin(in_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = in_center + p;
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv * in_uv_scale + in_uv_off;
    v_color = in_color;
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
flat in vec4 v_color;
uniform sampler2D u_tex;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    f_color = t * v_color;
}
"""

# floats per vertex: in_pos(2) in_uv(2) center(2) size(2) rot(1) color(4)
# uv_off(2) uv_scale(2)
_VERT_FLOATS = 17
_SPRITE_BYTES = 4 * _VERT_FLOATS * 4          # 4 corners × 17 f4


@dataclass(slots=True)
class Sprite:
    """A single textured/coloured quad to draw this frame (back-to-front).

    slots=True: ~444 of these are created per frame; slots cut the
    per-instance dict alloc + speed up the 13 attribute reads the batch
    serialiser does per sprite. dataclasses.replace works unchanged."""
    x: float                 # screen px, center
    y: float                 # screen px, center
    w: float
    h: float
    texture_key: str | None = None      # None = solid colour quad
    color: tuple[float, float, float, float] = (1, 1, 1, 1)
    rotation: float = 0.0
    additive: bool = False   # additive blend (glow / hit explosion)
    # texture sub-rect (uv offset/scale) — the spinner-metre bottom-up
    # reveal draws only the bottom fraction of its texture
    uv_off: tuple[float, float] = (0.0, 0.0)
    uv_scale: tuple[float, float] = (1.0, 1.0)


class SpriteRenderer:
    def __init__(self, width: int, height: int,
                 ctx: "moderngl.Context | None" = None):
        self.width = width
        self.height = height
        self.ctx = ctx or create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.prog["u_screen"].value = (float(width), float(height))
        self.prog["u_tex"].value = 0

        # unit-quad corners + uv, replicated per sprite in _draw (v grows
        # downward with screen y — same corner order the old TRIANGLE_STRIP
        # used; the index pattern below re-emits its two triangles)
        self._corners = np.array([
            [-0.5, -0.5, 0.0, 0.0],
            [ 0.5, -0.5, 1.0, 0.0],
            [-0.5,  0.5, 0.0, 1.0],
            [ 0.5,  0.5, 1.0, 1.0],
        ], dtype="f4")

        self._capacity = 0
        self.vbo: "moderngl.Buffer | None" = None
        self._ibo: "moderngl.Buffer | None" = None
        self.vao: "moderngl.VertexArray | None" = None
        self._ensure_capacity(2048)

        # texture-backed colour attachment (was a renderbuffer): the bloom
        # post-pass samples the scene, and fbo.read() works the same
        self.color_tex = self.ctx.texture((width, height), 4)
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])
        self._textures: dict[str, "moderngl.Texture"] = {}
        self._nomip_keys: set[str] = set()
        # key -> (texture, framebuffer over it) for blit_texture_from
        self._blit_targets: dict = {}
        self._max_tex_cached: int | None = None
        self._warned_big: set = set()
        self._white = self._make_texture_rgba(np.full((1, 1, 4), 255, dtype="u1"))
        # optional per-sprite post-transform (Sprite -> Sprite), applied to
        # every sprite in draw(). The fail animation installs this to drop
        # the frozen playfield's objects off-screen; None = identity.
        self.post_xform = None
        # pipelined readback ring (danser's recording-pipeline pattern:
        # glReadPixels DMAs into a pixel-pack buffer while the CPU builds
        # the next frames; mapping a buffer ~2 frames later never stalls)
        self._pbos: list["moderngl.Buffer"] | None = None
        self._pbo_head = 0
        self._pbo_tail = 0
        # Which slots currently hold a LIVE glMapBufferRange pointer. Unmapping
        # can only happen on this (context-owning) thread, so a slot is unmapped
        # lazily when the ring wraps back to it -- see _pool_depth.
        self._mapped: list[bool] = []
        # Readback latency, decoupled from pool size: equal in the unmapped path
        # (preserving today's behaviour exactly), smaller than it when mapping.
        self._lat = self._PBO_RING
        # recycled CPU-side frame buffers for the PBO readback: a fresh
        # 6 MB np.empty per frame costs an mmap + page-fault storm; the
        # encoder's writer thread hands frames back via recycle_frame()
        # once ffmpeg has them, so steady state reuses ~6 warm buffers.
        # deque append/pop are GIL-atomic (writer thread + render thread).
        self._frame_pool: "deque[np.ndarray]" = deque()
        self._frame_pool_ids: set[int] = set()

    def _ensure_capacity(self, n_sprites: int) -> None:
        """Size the dynamic VBO + static index buffer for n_sprites quads."""
        if n_sprites <= self._capacity:
            return
        cap = max(n_sprites, self._capacity * 2, 2048)
        if self.vao is not None:
            self.vao.release()
        if self._ibo is not None:
            self._ibo.release()
        if self.vbo is None:
            self.vbo = self.ctx.buffer(reserve=cap * _SPRITE_BYTES,
                                       dynamic=True)
        else:
            self.vbo.orphan(cap * _SPRITE_BYTES)
        # two triangles per quad: (0,1,2) + (2,1,3) — the same coverage the
        # old strip produced for corners v0..v3
        idx = (np.arange(cap, dtype="u4")[:, None] * 4
               + np.array([0, 1, 2, 2, 1, 3], dtype="u4")[None, :])
        self._ibo = self.ctx.buffer(np.ascontiguousarray(idx))
        # persistent vertex scratch: the unit-quad corner/uv block of every
        # row is constant, so it is written once here and _draw only fills
        # the per-sprite attribute columns (same bytes as a fresh build)
        self._verts = np.empty((cap, 4, _VERT_FLOATS), dtype="f4")
        self._verts[:, :, 0:4] = self._corners
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 2f 2f 2f 1f 4f 2f 2f",
              "in_pos", "in_uv", "in_center", "in_size", "in_rot",
              "in_color", "in_uv_off", "in_uv_scale")],
            index_buffer=self._ibo, index_element_size=4,
        )
        self._capacity = cap

    # --- texture management ---------------------------------------------------

    def upload_texture(self, key: str, rgba: np.ndarray,
                       clamp: bool = False, mipmaps: bool = True) -> None:
        """rgba: HxWx4 uint8 array (top-left origin). Re-uploading a key
        releases the previous texture (the HUD hp bar re-uploads per
        frame — without the release that's a VRAM leak). clamp=True sets
        clamp-to-edge wrapping (the flashlight overlay samples uv beyond
        [0,1] and needs the edge texel, not a repeat). mipmaps=False skips
        the mipmap build + uses plain LINEAR — for a texture drawn at ~1:1
        every frame (the SSAA base blit) building a full mip chain each
        frame is pure waste."""
        with perf.T("tex_upload"):
            self._upload_texture(key, rgba, clamp=clamp, mipmaps=mipmaps)

    def _upload_texture(self, key: str, rgba: np.ndarray,
                        clamp: bool = False, mipmaps: bool = True) -> None:
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        old = self._textures.get(key)
        tex = self._make_texture_rgba(rgba, mipmaps=mipmaps)
        if clamp:
            tex.repeat_x = False
            tex.repeat_y = False
        self._textures[key] = tex
        if old is not None:
            try:
                old.release()
            except Exception:  # noqa: BLE001 - context may be tearing down
                pass

    def write_texture(self, key: str, rgba: np.ndarray,
                      clamp: bool = False) -> None:
        """Per-frame texture update: same-size re-writes go through
        glTexSubImage2D on the EXISTING texture object — no allocation, no
        release, no mipmap chain (plain LINEAR). Only for textures drawn
        at 1:1 where the mip chain is never sampled (the HUD hp bars);
        the first call (or a size change) allocates a LINEAR no-mip
        texture."""
        with perf.T("tex_upload"):
            if rgba.dtype != np.uint8:
                rgba = rgba.astype("u1")
            if rgba.shape[2] == 3:
                a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
                rgba = np.concatenate([rgba, a], axis=2)
            h, w = rgba.shape[:2]
            tex = self._textures.get(key)
            if key in self._nomip_keys and tex is not None \
                    and tex.size == (w, h):
                tex.write(rgba)
                return
            self._upload_texture(key, rgba, clamp=clamp, mipmaps=False)
            self._nomip_keys.add(key)

    def write_texture_rgb(self, key: str, rgb: np.ndarray) -> None:
        """write_texture for an OPAQUE HxWx3 frame, without the RGB->RGBA
        widening: a 3-component texture samples alpha = 1.0, exactly what the
        appended 255 plane gave, and the np.full + np.concatenate that built
        that plane was ~2.9 ms per 720p frame (the sub-1080p results outro
        re-sends the whole scene-behind every frame). LINEAR, no mips - the
        same sampler state write_texture leaves."""
        with perf.T("tex_upload"):
            h, w = rgb.shape[:2]
            tex = self._textures.get(key)
            if not (key in self._nomip_keys and tex is not None
                    and tex.size == (w, h) and tex.components == 3):
                if tex is not None:
                    tex.release()
                tex = self.ctx.texture((w, h), 3, alignment=1)
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._textures[key] = tex
                self._nomip_keys.add(key)
            tex.write(np.ascontiguousarray(rgb), alignment=1)

    def blit_texture_from(self, key: str, src: "SpriteRenderer") -> None:
        """Make texture `key` hold `src`'s rendered frame, entirely on the GPU.

        Same texels the round trip `write_texture_rgb(key, src.read_rgb())`
        leaves, without the readback and the upload: a 1:1 NEAREST blit is an
        exact copy, and its destination rows are flipped because GL keeps a
        framebuffer's row 0 at the BOTTOM while our textures keep the image's
        top row first (read_rgb's flipud). Alpha is then forced to 1.0, which
        is what the 3-component texture sampled. LINEAR, no mips, default
        wrap: the sampler state write_texture_rgb leaves. Both renderers must
        share a GL context."""
        w, h = src.width, src.height
        ent = self._blit_targets.get(key)
        if ent is None or ent[0].size != (w, h):
            old = self._textures.pop(key, None)
            if old is not None:
                old.release()
            if ent is not None:
                ent[1].release()
            tex = self.ctx.texture((w, h), 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            ent = (tex, self.ctx.framebuffer(color_attachments=[tex]))
            self._blit_targets[key] = ent
            self._textures[key] = tex
            self._nomip_keys.add(key)
        tex, fbo = ent
        g = _load_gl_c()
        g.glBindFramebuffer(_GL_READ_FRAMEBUFFER, src.fbo.glo)
        g.glBindFramebuffer(_GL_DRAW_FRAMEBUFFER, fbo.glo)
        g.glBlitFramebuffer(0, 0, w, h, 0, h, w, 0,
                            _GL_COLOR_BUFFER_BIT, _GL_NEAREST)
        fbo.color_mask = (False, False, False, True)
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        fbo.color_mask = (True, True, True, True)
        # the raw binds above went behind moderngl's back
        self.fbo.use()

    def has_texture(self, key: str) -> bool:
        return key in self._textures

    def release_texture(self, key: str) -> None:
        """Free a cached texture by key (storyboard LRU eviction). No-op if
        the key is absent."""
        tex = self._textures.pop(key, None)
        self._nomip_keys.discard(key)
        if tex is not None:
            try:
                tex.release()
            except Exception:  # noqa: BLE001 - context may be tearing down
                pass

    # ---- GPU Argon bar (parity row 39) ----------------------------------
    #
    # The numpy bar IS a port of lazer's sh_ArgonBarPath.fs -- this puts it back in
    # shader form. `d` (distance to the full path) and `s` (progress of the nearest
    # point) are built ONCE in ArgonBarField.__init__ and never change, so they
    # upload once as one RG32F texture; everything that varies per frame (a, b, the
    # two endpoint positions, colours) is a uniform. That is what makes this worth
    # doing: the per-frame CPU cost AND the per-frame upload both go to zero, where a
    # naive port that uploaded core/inv/mix8 every frame would move 12 bytes/texel
    # instead of 4 and be SLOWER than the thing it replaced.
    #
    # Rendered into its own texture rather than straight into the scene, so the
    # existing sprite batch and painter's order are untouched.

    _BAR_VERT = ("#version 400\nin vec2 in_pos;\n"
                 "void main(){ gl_Position = vec4(in_pos,0.0,1.0); }")
    _BAR_FRAG = """#version 400
    uniform sampler2D field;        // R = d, G = s
    uniform vec2  u_pos_a, u_pos_b;
    uniform float u_a, u_b, u_radius, u_agp, u_alpha_mult, u_scale, u_margin, u_w;
    uniform vec3  u_bar;
    uniform vec4  u_glow;
    uniform int   u_xgrad;
    out vec4 frag;
    // Quantise to 8 bit, reproducing the CPU's f64 ROUNDING DECISION in f32.
    //
    // The whole fidelity problem is one multiply. In f32 the spacing at 229 is
    // ~1.5e-5, so `0.89999998 * 255.0` cannot be represented and snaps to
    // EXACTLY 229.5 -- which every round-half rule then takes UP, producing 6337
    // alpha values that were +1 and never -1. The CPU scales in f64, gets
    // 229.49999999999963, and rounds DOWN.
    //
    // fma recovers the bits the f32 product threw away: p + err is the EXACT
    // product, so (p - floor(p)) + err is the true fractional part and decides
    // the tie correctly. Doing the scale in `double` instead also fixes it, but
    // fp64 is software-emulated here -- it cost 33.6 s of shader time and
    // dropped the render from 177 to 49 fps, i.e. 3.6x SLOWER than the numpy
    // path it was replacing.
    float q8(float v) {
        float p = v * 255.0;
        float err = fma(v, 255.0, -p);        // exact residual of the product
        float fl = floor(p);
        float frac = (p - fl) + err;
        if (frac > 0.5) return (fl + 1.0) / 255.0;
        if (frac < 0.5) return fl / 255.0;
        // genuine tie: half to EVEN, matching np.rint
        return (mod(fl, 2.0) == 0.0 ? fl : fl + 1.0) / 255.0;
    }
    void main() {
        // NO y-flip. This target is SAMPLED by the sprite path exactly like an
        // uploaded texture, and an upload stores numpy row 0 as texel row 0. So
        // fragment row r must hold numpy row r. Flipping here (as this did until
        // 2026-10-01) drew the bar fill UPSIDE DOWN in every real frame; it went
        // unseen because the only fidelity test read the FBO back and flipped it
        // again before comparing.
        ivec2 ij = ivec2(gl_FragCoord.xy);
        vec2 fs = texelFetch(field, ij, 0).rg;
        float d = fs.r, sp = fs.g;
        float xx = (float(ij.x) + 0.5) / u_scale - u_margin;
        float yy = (float(ij.y) + 0.5) / u_scale - u_margin;
        vec2 p = vec2(xx, yy);
        float d_clipped = clamp(d, 0.0, u_radius);
        float D;
        if (u_b <= u_a + 1e-9) {            // point case
            D = clamp(length(p - u_pos_a), 0.0, u_radius);
        } else {
            bool inside = (u_a <= 0.0) ? (sp <= u_b) : (sp >= u_a && sp <= u_b);
            float da = clamp(length(p - u_pos_a), 0.0, u_radius);
            float db = clamp(length(p - u_pos_b), 0.0, u_radius);
            D = min(da, db);
            if (inside) D = d_clipped;
        }
        float core = clamp((u_radius - u_agp) - D, 0.0, 1.0);
        float inv  = 1.0 - core;
        float ga = clamp(1.0 - (D - u_radius + u_agp) / max(u_agp, 1e-9), 0.0, 1.0);
        // repeated squaring, NOT pow(ga,8.0): GLSL pow is exp2(8*log2(x)), a
        // transcendental approximation whose relative error the 8th power
        // amplifies. Three exact squarings match numpy's integer power.
        ga = ga * ga; ga = ga * ga; ga = ga * ga;
        vec3  rgb = core * u_bar + inv * u_glow.rgb;
        float a   = core + inv * (ga * u_glow.a);
        if (u_xgrad != 0) a *= 0.8 + 0.2 * clamp(xx / max(u_w, 1e-9), 0.0, 1.0);
        a *= u_alpha_mult;
        a = clamp(a, 0.0, 1.0);
        // Round half DOWN (ceil(x-0.5)), which is what matches the CPU here.
        //
        // Not a style choice. The CPU scales in f64, where e.g. 0.8 + 0.2*0.5
        // gives 229.49999999999963 and np.rint takes it DOWN to 229. In f32 the
        // spacing at 229 is ~1.5e-5, so that value cannot be represented and
        // snaps to EXACTLY 229.5 -- after which round-half-up AND round-half-even
        // both go UP, which is why every one of 6337 differing alpha values was
        // +1 and never -1. The bar's inputs are nice decimals (0.8/0.2/0.5), so
        // the true product sits just BELOW the .5 boundary essentially every
        // time; rounding ties down reproduces the f64 result instead of the
        // f32 snapping artefact. (roundEven and pow->squaring both changed the
        // count by zero, which is what pointed at this multiply.)
        frag = vec4(q8(rgb.r), q8(rgb.g), q8(rgb.b), q8(a));
    }"""

    def ensure_bar_field(self, d: np.ndarray, sp: np.ndarray, scale: float,
                         margin: float, width_l: float) -> None:
        """Upload the frame-invariant (d, s) field once."""
        if getattr(self, "_bar_ready", False):
            return
        h, w = d.shape
        packed = np.empty((h, w, 2), dtype="f4")
        packed[..., 0] = d
        packed[..., 1] = sp
        self._bar_field = self.ctx.texture((w, h), 2, packed.tobytes(),
                                           dtype="f4")
        self._bar_field.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._bar_prog = self.ctx.program(vertex_shader=self._BAR_VERT,
                                          fragment_shader=self._BAR_FRAG)
        self._bar_quad = self.ctx.buffer(
            np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._bar_vao = self.ctx.vertex_array(
            self._bar_prog, [(self._bar_quad, "2f4", "in_pos")])
        self._bar_prog["u_scale"] = float(scale)
        self._bar_prog["u_margin"] = float(margin)
        self._bar_prog["u_w"] = float(width_l)
        self._bar_fbos = {}
        self._bar_hw = (w, h)
        self._bar_ready = True

    def render_bar(self, key: str, a: float, b: float, pos_a, pos_b,
                   radius: float, glow_portion: float, bar_rgb, glow_rgba,
                   xgrad: bool = False, alpha_mult: float = 1.0) -> None:
        """Render one Argon bar into texture `key` entirely on the GPU."""
        with perf.T("hud_health_gpu"):
            w, h = self._bar_hw
            ent = self._bar_fbos.get(key)
            if ent is None:
                tex = self.ctx.texture((w, h), 4)
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                ent = (tex, self.ctx.framebuffer(color_attachments=[tex]))
                self._bar_fbos[key] = ent
                self._textures[key] = tex      # the sprite path draws this
                self._nomip_keys.add(key)
            tex, fbo = ent
            pr = self._bar_prog
            self._bar_field.use(location=0)
            pr["field"] = 0
            pr["u_a"] = float(a)
            pr["u_b"] = float(b)
            pr["u_pos_a"] = (float(pos_a[0]), float(pos_a[1]))
            pr["u_pos_b"] = (float(pos_b[0]), float(pos_b[1]))
            pr["u_radius"] = float(radius)
            pr["u_agp"] = float(radius * glow_portion)
            pr["u_bar"] = (float(bar_rgb[0]), float(bar_rgb[1]),
                           float(bar_rgb[2]))
            pr["u_glow"] = (float(glow_rgba[0]), float(glow_rgba[1]),
                            float(glow_rgba[2]), float(glow_rgba[3]))
            pr["u_xgrad"] = 1 if xgrad else 0
            pr["u_alpha_mult"] = float(alpha_mult)
            self.ctx.disable(moderngl.BLEND)   # writing values, not compositing
            fbo.use()
            self._bar_vao.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.BLEND)
            self.fbo.use()                     # restore the scene target

    def _max_tex(self) -> int:
        if self._max_tex_cached is None:
            try:
                self._max_tex_cached = int(
                    self.ctx.info.get("GL_MAX_TEXTURE_SIZE", 16384))
            except Exception:  # noqa: BLE001 - a missing key must not be fatal
                self._max_tex_cached = 16384
        return self._max_tex_cached

    def _fit_texture(self, rgba: np.ndarray) -> np.ndarray:
        """Downscale an oversized asset to the device limit.

        Skin dimensions are UNTRUSTED INPUT. Past GL_MAX_TEXTURE_SIZE the upload
        does NOT raise: it sets GL_INVALID_VALUE (0x501), the driver substitutes
        the zero texture, and the element draws BLACK while the render reports
        success. Measured on an M1 Max (limit 16384): a 20000x12 asset uploaded
        "fine" and sampled as (0, 0, 0). A crash would have been kinder.

        Downscaled rather than refused: a slightly smaller element beats a black
        one, and nothing over 16384 px was ever going to be sampled at native
        size. Assets within the limit are returned UNTOUCHED, so every real skin
        stays byte-identical. This is the invariant (no texture exceeds the
        device limit) rather than a per-element special case."""
        h, w = rgba.shape[:2]
        mx = self._max_tex()
        if w <= mx and h <= mx:
            return rgba
        scale = mx / float(max(w, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        if (w, h) not in self._warned_big:
            self._warned_big.add((w, h))
            print(f"skin texture {w}x{h} exceeds GL_MAX_TEXTURE_SIZE {mx} - "
                  f"downscaling to {nw}x{nh} (it would otherwise draw BLACK)",
                  file=sys.stderr)
        from PIL import Image
        im = Image.fromarray(np.ascontiguousarray(rgba), "RGBA")
        return np.asarray(im.resize((nw, nh), Image.LANCZOS), dtype="u1")

    def _make_texture_rgba(self, rgba: np.ndarray,
                           mipmaps: bool = True) -> "moderngl.Texture":
        rgba = self._fit_texture(rgba)
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    # --- drawing --------------------------------------------------------------

    def begin(self, clear=(0.0, 0.0, 0.0)) -> None:
        self.fbo.use()
        self.ctx.clear(*clear)

    def draw(self, sprites: list[Sprite]) -> None:
        with perf.T("gl_sprites_draw"):
            perf.count("sprites", len(sprites))
            self._draw(sprites)

    def _draw(self, sprites: list[Sprite]) -> None:
        if self.post_xform is not None:
            sprites = [self.post_xform(sp) for sp in sprites]
        if not sprites:
            return
        # painter's order per pass: every non-additive sprite in order,
        # THEN every additive sprite in order (the exact two-phase order
        # the per-sprite loop produced) — one partition pass
        normal: list = []
        additive: list = []
        for sp in sprites:
            (additive if sp.additive else normal).append(sp)
        if additive:
            ordered = normal + additive
            n_norm = len(normal)
        else:
            ordered = sprites
            n_norm = len(sprites)
        n = len(ordered)
        self._ensure_capacity(n)

        # np.fromiter into a preallocated (n, 13) block — the same scalars
        # in the same order as the old list-of-tuples np.array (identical
        # f4 casts), without its per-element dtype discovery
        params = _abl_params.get(n) if (_ABL_NOSER or _ABL_NOVBO) else None
        _stale = params is not None
        if params is None and _SER_ARRAY:
            buf = _array("f")
            ext = buf.extend
            for sp in ordered:
                ext((sp.x, sp.y, sp.w, sp.h, sp.rotation,
                     *sp.color, *sp.uv_off, *sp.uv_scale))
            # a colour/uv tuple of the wrong length would shift every later
            # field: fall back to the indexed path rather than draw garbage
            if len(buf) == n * 13:
                params = np.frombuffer(buf, dtype="f4").reshape(n, 13)
                perf.count("ser_array")
            else:
                perf.count("ser_array_fallback")
        if params is None:
            params = np.fromiter(
                (v for sp in ordered for v in (
                    sp.x, sp.y, sp.w, sp.h, sp.rotation,
                    sp.color[0], sp.color[1], sp.color[2], sp.color[3],
                    sp.uv_off[0], sp.uv_off[1], sp.uv_scale[0], sp.uv_scale[1])),
                dtype="f4", count=n * 13).reshape(n, 13)
            if _ABL_NOSER or _ABL_NOVBO:
                _abl_params[n] = params
        if not (_ABL_NOVBO and _stale):
            verts = self._verts[:n]          # corners/uv pre-filled, constant
            verts[:, :, 4:] = params[:, None, :]
            self.vbo.orphan()
            self.vbo.write(verts)

        textures = self._textures
        white = self._white
        texs = [textures.get(sp.texture_key, white) if sp.texture_key
                else white for sp in ordered]
        if _ABL_ONETEX:            # ablation: collapse every run into one
            texs = [white] * len(texs)

        with perf.T("gl_runs"):
            self._run_pass(texs, 0, n_norm)
        if n_norm < n:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            with perf.T("gl_runs"):
                self._run_pass(texs, n_norm, n)
            self.ctx.blend_func = (moderngl.SRC_ALPHA,
                                   moderngl.ONE_MINUS_SRC_ALPHA)

    def _run_pass(self, texs, start: int, end: int) -> None:
        """Draw quads [start, end) grouping consecutive same-texture runs
        into single indexed draws (primitive order == list order, so the
        painter's-algorithm blending is unchanged)."""
        i = start
        render = self.vao.render
        while i < end:
            tex = texs[i]
            j = i + 1
            if not _ABL_NOMERGE:   # ablation: one draw per quad
                while j < end and texs[j] is tex:
                    j += 1
            tex.use(location=0)
            render(moderngl.TRIANGLES, vertices=(j - i) * 6, first=i * 6)
            perf.count("draw_calls")
            i = j

    _PBO_RING = 3

    def _pool_depth(self) -> int:
        """How many PBOs the ring needs.

        Unmapped: the historical 3 (latency == depth). Mapped: a slot's pointer
        stays live until the WRITER is done with it, so the slot must not come
        round again while anything downstream still holds it -- the writer queue,
        the frame mid-write, the one just popped, and the readback latency, plus
        slack. Getting this merely "big enough" is how catch corrupted output
        silently."""
        if not _MAP_READBACK:
            return self._PBO_RING
        return _WRITER_QUEUE_FRAMES + 1 + 1 + self._lat + _PBO_MARGIN

    def _ensure_pbos(self, size: int) -> None:
        if self._pbos is not None:
            if size != self._pbo_size:
                # Would silently read the wrong number of bytes per frame. The
                # rgb (w*h*3) and yuv (w*h*3/2) paths are mutually exclusive in
                # one render, so this can only fire on a wiring mistake.
                raise RuntimeError(
                    f"PBO ring already sized {self._pbo_size}, asked for {size}")
            return
        self._lat = _PBO_LAT if _MAP_READBACK else self._PBO_RING
        n = self._pool_depth()
        self._pbos = [self.ctx.buffer(reserve=size) for _ in range(n)]
        if _MAP_READBACK:          # scaffolding: only meaningful for the mapped path
            perf.count("pbo_pool_depth", n)
            perf.count("pbo_latency", self._lat)
        self._mapped = [False] * n
        self._pbo_size = size

    def _unmap_for_write(self, idx: int):
        """A MAPPED buffer cannot receive glReadPixels, so unmap before reusing
        the slot. _pool_depth guarantees the writer is long done with it."""
        buf = self._pbos[idx]
        if _MAP_READBACK and self._mapped[idx]:
            g = _load_gl_c()
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
            g.glUnmapBuffer(_GL_PIXEL_PACK_BUFFER)
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            self._mapped[idx] = False
            perf.count("pbo_unmapped")   # inside the _MAP_READBACK branch already
        return buf

    def _map_slot(self, idx: int) -> np.ndarray:
        """Flat uint8 view straight into the pixel-pack buffer -- no copy."""
        import ctypes as _ct
        g = _load_gl_c()
        buf = self._pbos[idx]
        g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
        ptr = g.glMapBufferRange(_GL_PIXEL_PACK_BUFFER, 0, self._pbo_size,
                                 _GL_MAP_READ_BIT | _GL_MAP_WRITE_BIT)
        g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
        if not ptr:
            raise RuntimeError("glMapBufferRange returned NULL")
        self._mapped[idx] = True
        perf.count("pbo_mapped")
        return np.ctypeslib.as_array(
            (_ct.c_uint8 * self._pbo_size).from_address(ptr))

    def read_rgb_async(self) -> "np.ndarray | None":
        """Queue an async readback of the current fbo into a small PBO
        ring and return the OLDEST completed frame (top-left origin), or
        None while the ring is still filling. Frames come back in strict
        submission order — the record loop pushes them straight to ffmpeg,
        so the byte stream is identical to the synchronous read_rgb path,
        just ~RING-1 frames late. read_drain() flushes the tail."""
        with perf.T("readback"):
            self._ensure_pbos(self.width * self.height * 3)
            buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
            self.fbo.read_into(buf, components=3, alignment=1)
            self._pbo_head += 1
            if self._pbo_head - self._pbo_tail < self._lat:
                return None
            return self._pop_pbo()

    def _pop_pbo(self) -> np.ndarray:
        idx = self._pbo_tail % len(self._pbos)
        self._pbo_tail += 1
        if _MAP_READBACK:
            flat = self._map_slot(idx)
            arr = flat.reshape((self.height, self.width, 3))
            return np.flipud(arr)  # same orientation contract as read_rgb
        buf = self._pbos[idx]
        arr = self._frame_buf()
        buf.read_into(arr)         # same bytes buf.read() returned, copied
        return np.flipud(arr)      # same orientation contract as read_rgb

    def _frame_buf(self) -> np.ndarray:
        """A (h, w, 3) u1 frame buffer — pooled when the encoder has
        recycled one, else freshly allocated and registered."""
        try:
            return self._frame_pool.pop()
        except IndexError:
            arr = np.empty((self.height, self.width, 3), dtype="u1")
            # pooled arrays live for the whole render, so their ids are
            # stable and can never be re-issued to a foreign object
            self._frame_pool_ids.add(id(arr))
            return arr

    def recycle_frame(self, frame) -> None:
        """Return a pooled readback frame once the encoder is done with
        it (called from the ffmpeg writer thread). Frames from other
        sources (sync read_rgb / SSAA composites) are ignored."""
        base = frame.base if frame.base is not None else frame
        if id(base) in self._frame_pool_ids:
            # yuv frames are flat; rgb frames are (h, w, 3)
            if base.ndim == 1:
                self._yuv_pool.append(base)
            else:
                self._frame_pool.append(base)

    def read_drain(self) -> list:
        """Return every frame still in flight, oldest first (map end, or
        an SSAA/results frame about to take the synchronous path)."""
        out = []
        with perf.T("readback"):
            while self._pbos is not None and self._pbo_tail < self._pbo_head:
                out.append(self._pop_pbo())
        return out

    # ---- GPU RGB -> yuv420p ----------------------------------------------
    #
    # ORIENTATION: nothing is flipped here, deliberately. Frames already reach
    # ffmpeg BOTTOM-UP and `-vf vflip` reorders the rows (see encode.py). That
    # stays correct for yuv420p: with an even height a vertical flip maps chroma
    # row j to ch-1-j, whose luma pair is still even-aligned, so the planes are
    # bit-identical to converting the top-down RGB.

    def _ensure_yuv(self):
        """Lazily build the conversion pass. Built on first use so an unused flag
        costs nothing (two programs, three textures, ~3 MB of FBOs)."""
        if getattr(self, "_yuv_ready", False):
            return
        w, h = self.width, self.height
        if (w & 1) or (h & 1):
            raise RuntimeError(
                f"R3D_STD_GPU_YUV needs even dimensions, got {w}x{h}")
        RY, GY, BY = (_yuv_coef(k, 219) for k in (0.299, 0.587, 0.114))
        RU = -_yuv_coef(0.169, 224); GU = -_yuv_coef(0.331, 224)
        BU = _yuv_coef(0.500, 224)
        RV = _yuv_coef(0.500, 224); GV = -_yuv_coef(0.419, 224)
        BV = -_yuv_coef(0.081, 224)
        S = _RGB2YUV_SHIFT
        vert = ("#version 330\nin vec2 in_pos;\n"
                "void main(){ gl_Position = vec4(in_pos,0.0,1.0); }")
        # INTEGER math throughout: the shifts must be exact. Texture samples come
        # back as normalised floats, so `int(v*255.0 + 0.5)` recovers the byte --
        # float32 holds 0..255 exactly, and the +0.5 stops the n/255*255 round-trip
        # landing a hair under n and truncating to n-1.
        frag_y = f"""#version 330
        uniform sampler2D scene;
        out float outY;
        void main() {{
            vec3 c = texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb;
            int r = int(c.r*255.0+0.5), g = int(c.g*255.0+0.5), b = int(c.b*255.0+0.5);
            outY = float((({RY}*r + {GY}*g + {BY}*b) >> {S}) + 16) / 255.0;
        }}"""
        # U and V share the 2x2 gather, so one pass with two attachments halves
        # the sampling versus a pass each.
        frag_uv = f"""#version 330
        uniform sampler2D scene;
        layout(location=0) out float outU;
        layout(location=1) out float outV;
        void main() {{
            ivec2 q = ivec2(gl_FragCoord.xy) * 2;
            ivec3 s = ivec3(0);
            for (int dy=0; dy<2; ++dy) for (int dx=0; dx<2; ++dx) {{
                vec3 c = texelFetch(scene, q + ivec2(dx,dy), 0).rgb;
                s += ivec3(int(c.r*255.0+0.5), int(c.g*255.0+0.5), int(c.b*255.0+0.5));
            }}
            outU = float((({RU}*s.r + {GU}*s.g + {BU}*s.b) >> {S + 2}) + 128) / 255.0;
            outV = float((({RV}*s.r + {GV}*s.g + {BV}*s.b) >> {S + 2}) + 128) / 255.0;
        }}"""
        self._yuv_quad = self.ctx.buffer(
            np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._yuv_prog_y = self.ctx.program(vertex_shader=vert,
                                            fragment_shader=frag_y)
        self._yuv_prog_uv = self.ctx.program(vertex_shader=vert,
                                             fragment_shader=frag_uv)
        self._yuv_vao_y = self.ctx.vertex_array(
            self._yuv_prog_y, [(self._yuv_quad, "2f4", "in_pos")])
        self._yuv_vao_uv = self.ctx.vertex_array(
            self._yuv_prog_uv, [(self._yuv_quad, "2f4", "in_pos")])
        self._tex_y = self.ctx.texture((w, h), 1, dtype="f1")
        self._tex_u = self.ctx.texture((w // 2, h // 2), 1, dtype="f1")
        self._tex_v = self.ctx.texture((w // 2, h // 2), 1, dtype="f1")
        self._fbo_y = self.ctx.framebuffer(color_attachments=[self._tex_y])
        self._fbo_uv = self.ctx.framebuffer(
            color_attachments=[self._tex_u, self._tex_v])
        self._yuv_size = w * h * 3 // 2
        self._yuv_stage = None
        self._yuv_pool: deque = deque()
        self._yuv_ready = True

    def _run_yuv_passes(self, src_tex):
        """Both conversion passes over `src_tex`. Blending MUST be off: these
        passes write computed values, not composites, and a stray blend would
        silently corrupt the planes."""
        self.ctx.disable(moderngl.BLEND)
        src_tex.use(location=0)
        self._yuv_prog_y["scene"] = 0
        self._yuv_prog_uv["scene"] = 0
        self._fbo_y.use()
        self._yuv_vao_y.render(moderngl.TRIANGLES)
        self._fbo_uv.use()
        self._yuv_vao_uv.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.BLEND)
        self.fbo.use()                 # restore: callers expect the scene fbo bound

    def _queue_yuv(self) -> "np.ndarray | None":
        """Read the three planes into ONE pbo at their yuv420p offsets, so the
        writer still pushes a single contiguous block and nothing downstream has
        to know the frame is planar."""
        w, h = self.width, self.height
        ysz, csz = w * h, (w // 2) * (h // 2)
        self._ensure_pbos(self._yuv_size)
        buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
        self._fbo_y.read_into(buf, components=1, alignment=1, write_offset=0)
        self._fbo_uv.read_into(buf, components=1, alignment=1, attachment=0,
                               write_offset=ysz)
        self._fbo_uv.read_into(buf, components=1, alignment=1, attachment=1,
                               write_offset=ysz + csz)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < self._lat:
            return None
        return self._pop_pbo_flat()

    def read_yuv_async(self) -> "np.ndarray | None":
        """yuv420p twin of read_rgb_async: same ring, same FIFO ordering, but
        1.5 bytes/px instead of 3."""
        with perf.T("readback"):
            self._ensure_yuv()
            self._run_yuv_passes(self.color_tex)
            return self._queue_yuv()

    def yuv_from_rgb(self, rgb) -> "np.ndarray | None":
        """Convert a CPU-composited frame (the SSAA results outro) on the GPU and
        queue it into the SAME ring.

        Not optional: taiko measured the pure-numpy twin at 14.01 ms/frame, and a
        synchronous GPU read here at 4.27 ms/frame, which turned a 42.7% win on a
        long map into a 6.5% LOSS on a short one where the outro is a sixth of the
        frames. The stall was the sync, not the copies.

        `rgb` is TOP-DOWN (the SSAA path builds it through PIL). The gameplay
        planes come out of GL BOTTOM-UP, and the pipe applies one vflip to the
        whole stream, so this frame has to be flipped to the GL convention first
        or the outro alone ends up upside down. The reversed view is not
        contiguous, hence the copy -- outro frames are a small minority."""
        with perf.T("readback"):
            self._ensure_yuv()
            h, w = rgb.shape[:2]
            if (w, h) != (self.width, self.height):
                raise RuntimeError(f"yuv_from_rgb: expected "
                                   f"{self.width}x{self.height}, got {w}x{h}")
            rgb = np.ascontiguousarray(rgb[::-1])
            if self._yuv_stage is None:
                self._yuv_stage = self.ctx.texture((w, h), 3)
                self._yuv_stage.filter = (moderngl.NEAREST, moderngl.NEAREST)
            # write the array DIRECTLY (buffer protocol) -- .tobytes() copies
            # 6.2 MB per frame before the upload even starts
            self._yuv_stage.write(memoryview(rgb))
            self._run_yuv_passes(self._yuv_stage)
            return self._queue_yuv()

    def _pop_pbo_flat(self) -> np.ndarray:
        """_pop_pbo's flat sibling — yuv420p is planar, not (h, w, c)-shaped, and
        it must NOT be flipped (see the ORIENTATION note above)."""
        idx = self._pbo_tail % len(self._pbos)
        self._pbo_tail += 1
        if _MAP_READBACK:
            return self._map_slot(idx)
        try:
            arr = self._yuv_pool.pop()
        except IndexError:
            arr = np.empty(self._yuv_size, dtype="u1")
            self._frame_pool_ids.add(id(arr))
        self._pbos[idx].read_into(arr)
        return arr

    def read_yuv_drain(self) -> list:
        """Every yuv frame still in flight, oldest first."""
        out = []
        with perf.T("readback"):
            while self._pbos is not None and self._pbo_tail < self._pbo_head:
                out.append(self._pop_pbo_flat())
        return out

    def read_rgb(self) -> np.ndarray:
        """HxWx3 uint8, top-left origin (ready for ffmpeg rgb24)."""
        with perf.T("readback"):
            data = self.fbo.read(components=3, alignment=1)
            arr = np.frombuffer(data, dtype="u1").reshape(
                (self.height, self.width, 3))
            return np.flipud(arr)  # moderngl reads bottom-left origin

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass
