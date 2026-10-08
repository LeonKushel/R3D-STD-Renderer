"""The sub-1080p results outro on the Metal core: Pillow's 8-bit LANCZOS as two
integer passes (render/ssaa_gpu.py, the same arithmetic and the same tables),
drawn from the supersampled card straight into the frame."""
from __future__ import annotations

import numpy as np

from ..ssaa_gpu import coeffs
from . import core as mc

_MSL = """
#include <metal_stdlib>
using namespace metal;
struct LU { int axis; int src_h; int out_h; int pad; };
struct V { float4 pos [[position]]; };
vertex V lz_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    V o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); return o;
}
// rows are stored bottom-up (the core is created that way); Pillow's rows run
// from the top, hence the mirrored indices in the vertical pass
fragment float4 lz_fs(V in [[stage_in]],
                      texture2d<float> src [[texture(0)]],
                      texture2d<int> k [[texture(1)]],
                      texture2d<int> b [[texture(2)]],
                      constant LU& u [[buffer(2)]]) {
    int2 p = int2(in.pos.xy);
    int3 ss = int3(1 << %(half)d);
    if (u.axis == 0) {
        int2 bb = b.read(uint2(p.x, 0)).rg;
        for (int i = 0; i < bb.y; ++i) {
            int3 px = int3(src.read(uint2(bb.x + i, p.y)).rgb * 255.0 + 0.5);
            ss += px * k.read(uint2(i, p.x)).r;
        }
    } else {
        int yy = u.out_h - 1 - p.y;
        int2 bb = b.read(uint2(yy, 0)).rg;
        for (int i = 0; i < bb.y; ++i) {
            int3 px = int3(src.read(uint2(p.x, u.src_h - 1 - (bb.x + i))).rgb * 255.0 + 0.5);
            ss += px * k.read(uint2(i, yy)).r;
        }
    }
    int3 o = clamp(ss >> %(bits)d, 0, 255);
    return float4(float3(o) / 255.0, 1.0);
}
""" % {"half": 21, "bits": 22}


class MetalLanczos:
    """Downscale the supersampled renderer's picture into the frame.
    `run` and `self_check` mirror render/ssaa_gpu.py GpuLanczos."""

    def __init__(self, spr, spr_hi):
        self.spr, self.hi = spr, spr_hi
        self.core = c = spr.core
        self.iw, self.ih, self.ow, self.oh = spr_hi.width, spr_hi.height, spr.width, spr.height
        self.pipe = c.pipe_create(_MSL, "lz_vs", "lz_fs", mc.RGBA8, mc.BLEND_OFF)
        self._kh, self._bh = self._tables(self.iw, self.ow)
        self._kv, self._bv = self._tables(self.ih, self.oh)
        self._tmp = c.tex_create(self.ow, self.ih, mc.RGBA8, target=True,
                                 flags=mc.NEAREST | mc.CLAMP)

    def _tables(self, n_in: int, n_out: int):
        bounds, kk = coeffs(n_in, n_out)
        c = self.core
        k = c.tex_create(kk.shape[1], n_out, mc.R32I, np.ascontiguousarray(kk, "i4"),
                         flags=mc.NEAREST | mc.CLAMP)
        b = c.tex_create(n_out, 1, mc.RG32I, np.ascontiguousarray(bounds, "i4"),
                         flags=mc.NEAREST | mc.CLAMP)
        return k, b

    def _passes(self, src: int, dst: int) -> None:
        c = self.core
        u = np.array([0, self.ih, self.oh, 0], "i4")
        c.set_pass(self._tmp, (0.0, 0.0, 0.0, 1.0))
        c.draw(self.pipe, 3, uniforms=u, textures=(src, self._kh, self._bh))
        u = np.array([1, self.ih, self.oh, 0], "i4")
        c.set_pass(dst, (0.0, 0.0, 0.0, 1.0))
        c.draw(self.pipe, 3, uniforms=u, textures=(self._tmp, self._kv, self._bv))

    def run(self, src_tex=None, dst_fbo=None) -> None:
        """The supersampled renderer's recorded picture -> the frame."""
        self.spr._replay()           # the frame is open and holds the scene behind the card
        self.hi._replay()
        self._passes(self.hi.color_tex, 0)

    def self_check(self) -> bool:
        """One fixed noise frame through these passes and through Pillow."""
        from PIL import Image
        rng = np.random.default_rng(20261006)
        noise = rng.integers(0, 256, (self.ih, self.iw, 3), dtype=np.uint8)
        ref = np.asarray(Image.fromarray(noise).resize((self.ow, self.oh), Image.LANCZOS), np.uint8)
        c = self.core
        rgba = np.empty((self.ih, self.iw, 4), np.uint8)
        rgba[..., :3] = noise[::-1]
        rgba[..., 3] = 255
        src = c.tex_create(self.iw, self.ih, mc.RGBA8, rgba, flags=mc.NEAREST | mc.CLAMP)
        dst = c.tex_create(self.ow, self.oh, mc.RGBA8, target=True)
        try:
            c.ensure_frame()
            self._passes(src, dst)
            c.aux_commit()
            got = c.tex_read(dst)[::-1, :, :3]
            return bool(np.array_equal(got, ref))
        finally:
            c.tex_free(src)
            c.tex_free(dst)
