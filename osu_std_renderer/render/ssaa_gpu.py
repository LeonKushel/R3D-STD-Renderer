"""Pillow's 8-bit LANCZOS downscale, exactly, on the GPU.

Below 1080p the results outro is composited at >=1080p and downscaled to the
output size (scene._frame_rgb_ssaa). The downscale used to be
``Image.resize(..., LANCZOS)`` on the CPU: ~23 ms per frame at 1080 -> 720,
behind a synchronous read of the whole hi-res frame. Pillow's resampler is
fixed-point integer arithmetic (Resample.c, 8 bits per channel):

    k_int  = (int)(k * 2**22 +- 0.5)              coefficient, per tap
    out    = clip8((2**21 + sum(pixel * k_int)) >> 22)

horizontal pass first, into an 8-bit intermediate, then vertical. Integers are
exact on every GPU, so the same two passes run here as fragment shaders with
``int`` accumulators and ``texelFetch`` (no filtering, no float maths on the
data path) and produce the same bytes. The coefficients are computed the way
Pillow computes them (`coeffs`), and `GpuLanczos.self_check` runs one noise
frame through the GPU and through Pillow itself before the path is trusted:
any difference and the caller stays on the CPU path.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

PRECISION_BITS = 32 - 8 - 2          # Pillow's PRECISION_BITS for 8 bpc


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    x = x * math.pi
    return math.sin(x) / x


def _lanczos(x: float) -> float:
    if -3.0 <= x < 3.0:
        return _sinc(x) * _sinc(x / 3)
    return 0.0


def coeffs(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pillow's precompute_coeffs + normalize_coeffs_8bpc for LANCZOS.

    Returns (bounds, kk): bounds[i] = (first source index, tap count) and
    kk[i, :count] = the fixed-point coefficients for output index i."""
    scale = filterscale = in_size / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = 3.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    bounds = np.zeros((out_size, 2), np.int32)
    kk = np.zeros((out_size, ksize), np.int32)
    ss = 1.0 / filterscale
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        w = [_lanczos((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        ww = 0.0
        for v in w:                       # the C loop's summation order
            ww += v
        if ww != 0.0:
            w = [v / ww for v in w]
        for x, v in enumerate(w):
            f = v * (1 << PRECISION_BITS)
            kk[xx, x] = int(-0.5 + f) if v < 0 else int(0.5 + f)
        bounds[xx] = (xmin, xmax)
    return bounds, kk


def _pass_cpu(img: np.ndarray, bounds: np.ndarray, kk: np.ndarray,
              axis: int) -> np.ndarray:
    a = np.moveaxis(img, axis, 0).astype(np.int64)
    out = np.empty((bounds.shape[0],) + a.shape[1:], np.int64)
    half = 1 << (PRECISION_BITS - 1)
    for xx in range(bounds.shape[0]):
        lo, n = bounds[xx]
        acc = np.tensordot(kk[xx, :n].astype(np.int64), a[lo:lo + n],
                           axes=(0, 0))
        out[xx] = (acc + half) >> PRECISION_BITS
    return np.moveaxis(np.clip(out, 0, 255).astype(np.uint8), 0, axis)


def resize_exact_cpu(img: np.ndarray, ow: int, oh: int) -> np.ndarray:
    """numpy twin of the shader pair (and of Pillow). Tests only: slow."""
    ih, iw = img.shape[:2]
    tmp = img
    if iw != ow:
        tmp = _pass_cpu(tmp, *coeffs(iw, ow), axis=1)
    if ih != oh:
        tmp = _pass_cpu(tmp, *coeffs(ih, oh), axis=0)
    return tmp


_VERT = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# One fragment = one output sample along the resampled axis. u_axis picks the
# axis; u_flip_src > 0 reads the source rows mirrored (see GpuLanczos.run).
_FRAG = """
#version 330
uniform sampler2D u_src;      // RGBA8
uniform isampler2D u_k;       // R32I,  (ksize, out_size): coefficients
uniform isampler2D u_b;       // RG32I, (out_size, 1): (first, count)
uniform int u_axis;           // 0 = horizontal pass, 1 = vertical pass
uniform int u_src_h;          // vertical pass: source height
uniform int u_out_h;          // vertical pass: output height
out vec4 f_color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec3 ss = ivec3(1 << %(half)d);
    if (u_axis == 0) {
        ivec2 b = texelFetch(u_b, ivec2(p.x, 0), 0).rg;
        for (int i = 0; i < b.y; ++i) {
            ivec3 px = ivec3(texelFetch(u_src, ivec2(b.x + i, p.y), 0).rgb
                             * 255.0 + 0.5);
            ss += px * texelFetch(u_k, ivec2(i, p.x), 0).r;
        }
    } else {
        // framebuffer row p.y counts from the BOTTOM; Pillow's row from the top
        int yy = u_out_h - 1 - p.y;
        ivec2 b = texelFetch(u_b, ivec2(yy, 0), 0).rg;
        for (int i = 0; i < b.y; ++i) {
            ivec3 px = ivec3(texelFetch(
                u_src, ivec2(p.x, u_src_h - 1 - (b.x + i)), 0).rgb
                * 255.0 + 0.5);
            ss += px * texelFetch(u_k, ivec2(i, yy), 0).r;
        }
    }
    ivec3 o = clamp(ss >> %(bits)d, 0, 255);
    f_color = vec4(vec3(o) / 255.0, 1.0);
}
""" % {"half": PRECISION_BITS - 1, "bits": PRECISION_BITS}


class GpuLanczos:
    """Downscale an (iw, ih) RGBA8 texture into an (ow, oh) framebuffer.

    Both the source texture and the destination framebuffer hold their rows
    bottom-up (they are render targets); the result is what Pillow gives for
    the same image stored top-down."""

    def __init__(self, ctx: "moderngl.Context", iw: int, ih: int,
                 ow: int, oh: int):
        self.ctx = ctx
        self.iw, self.ih, self.ow, self.oh = iw, ih, ow, oh
        self.prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.prog["u_src"].value = 0
        self.prog["u_k"].value = 1
        self.prog["u_b"].value = 2
        self._vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._vao = ctx.vertex_array(self.prog, [(self._vbo, "2f", "in_pos")])
        self._kh, self._bh = self._coef_textures(iw, ow)
        self._kv, self._bv = self._coef_textures(ih, oh)
        # horizontal pass target: output width, SOURCE height
        self._tmp = ctx.texture((ow, ih), 4)
        self._tmp.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._tmp_fbo = ctx.framebuffer(color_attachments=[self._tmp])

    def _coef_textures(self, n_in: int, n_out: int):
        bounds, kk = coeffs(n_in, n_out)
        k = self.ctx.texture((kk.shape[1], n_out), 1,
                             np.ascontiguousarray(kk).tobytes(), dtype="i4")
        b = self.ctx.texture((n_out, 1), 2,
                             np.ascontiguousarray(bounds).tobytes(), dtype="i4")
        for t in (k, b):          # an integer texture is incomplete under LINEAR
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return k, b

    def run(self, src_tex: "moderngl.Texture",
            dst_fbo: "moderngl.Framebuffer") -> None:
        """src_tex (iw x ih) -> dst_fbo (ow x oh). Leaves dst_fbo bound and
        blending enabled, the state every SpriteRenderer draw expects."""
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)        # writing values, not compositing
        try:
            self.prog["u_src_h"].value = self.ih
            self.prog["u_out_h"].value = self.oh
            # 1) horizontal: (iw, ih) -> (ow, ih), rows untouched
            self._tmp_fbo.use()
            src_tex.use(0)
            self._kh.use(1)
            self._bh.use(2)
            self.prog["u_axis"].value = 0
            self._vao.render(moderngl.TRIANGLES, vertices=3)
            # 2) vertical: (ow, ih) -> (ow, oh)
            dst_fbo.use()
            self._tmp.use(0)
            self._kv.use(1)
            self._bv.use(2)
            self.prog["u_axis"].value = 1
            self._vao.render(moderngl.TRIANGLES, vertices=3)
        finally:
            ctx.enable(moderngl.BLEND)

    def self_check(self) -> bool:
        """One fixed noise frame through the GPU and through Pillow itself.
        True only when every byte agrees."""
        from PIL import Image
        rng = np.random.default_rng(20261006)
        noise = rng.integers(0, 256, (self.ih, self.iw, 3), dtype=np.uint8)
        ref = np.asarray(Image.fromarray(noise).resize(
            (self.ow, self.oh), Image.LANCZOS), dtype=np.uint8)
        ctx = self.ctx
        prev = ctx.fbo
        src = ctx.texture((self.iw, self.ih), 3,
                          np.ascontiguousarray(noise[::-1]).tobytes(),
                          alignment=1)
        src.filter = (moderngl.NEAREST, moderngl.NEAREST)
        dst_tex = ctx.texture((self.ow, self.oh), 4)
        dst = ctx.framebuffer(color_attachments=[dst_tex])
        try:
            self.run(src, dst)
            got = np.frombuffer(dst.read(components=3, alignment=1),
                                dtype="u1").reshape((self.oh, self.ow, 3))[::-1]
            return bool(np.array_equal(got, ref))
        finally:
            dst.release()
            dst_tex.release()
            src.release()
            prev.use()
