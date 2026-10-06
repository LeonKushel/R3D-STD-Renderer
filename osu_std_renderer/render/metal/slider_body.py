"""Slider bodies on the Metal core: the same three passes as
render/slider_body.py (distance field by the depth trick, shade, composite),
the same geometry, the same constants.

What differs is WHEN they run. The GL renderer shares one target pair, so each
slider's build and composite interleave and the scene framebuffer is left and
re-entered per slider. Here every body gets its own target from a pool, all
the builds are replayed before the scene pass opens, and the scene pass only
composites them: three GPU passes per slider become two, and the scene pass is
never broken."""
from __future__ import annotations

import numpy as np

from .. import perf
from ..slider_body import (AA_BAND_PORTION, BodyStyle, BodyTexture, _disc_mesh,
                           _tent_mesh, border_portion, shade2, sub_path)
from . import core as mc

_MSL = """
#include <metal_stdlib>
using namespace metal;

struct GeomU { float2 screen; float radius; float pad; };
struct SegInst { packed_float2 p1; float len; packed_float2 dir; };
struct DV { float4 pos [[position]]; float d; };

// window depth == distance, so the depth test keeps the NEAREST centreline:
// the union of every swept circle in one pass. Rows run BOTTOM-UP, as in the GL
// renderer these passes are ported from (the core is created bottom_up), so a
// screen y maps to +y in clip space.
vertex DV seg_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                 constant packed_float3* mesh [[buffer(0)]],
                 constant SegInst* inst [[buffer(1)]],
                 constant GeomU& u [[buffer(2)]]) {
    float3 m = float3(mesh[vid]);
    SegInst I = inst[iid];
    float2 dir = float2(I.dir);
    float2 nrm = float2(-dir.y, dir.x);
    float2 px = float2(I.p1) + dir * (m.x * I.len) + nrm * (m.y * u.radius);
    DV o;
    o.pos = float4(px.x / u.screen.x * 2.0 - 1.0, px.y / u.screen.y * 2.0 - 1.0, m.z, 1.0);
    o.d = m.z;
    return o;
}

vertex DV cap_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                 constant packed_float3* mesh [[buffer(0)]],
                 constant packed_float2* inst [[buffer(1)]],
                 constant GeomU& u [[buffer(2)]]) {
    float3 m = float3(mesh[vid]);
    float2 px = float2(inst[iid]) + m.xy * u.radius;
    DV o;
    o.pos = float4(px.x / u.screen.x * 2.0 - 1.0, px.y / u.screen.y * 2.0 - 1.0, m.z, 1.0);
    o.d = m.z;
    return o;
}

fragment float4 dist_fs(DV in [[stage_in]]) { return float4(in.d, 0.0, 0.0, 1.0); }

struct QuadU { float4 border; float4 inner; float4 outer; float4 misc; };   // misc: border_portion, blend, screen.xy
struct QV { float4 pos [[position]]; };

vertex QV quad_vs(uint vid [[vertex_id]],
                  constant packed_float2* v [[buffer(0)]],
                  constant QuadU& u [[buffer(2)]]) {
    float2 p = float2(v[vid]);
    QV o;
    o.pos = float4(p.x / u.misc.z * 2.0 - 1.0, p.y / u.misc.w * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

constant float SHADOW = 0.06640625;   // 34/512

fragment float4 shade_fs(QV in [[stage_in]],
                         texture2d<float> dist [[texture(0)]],
                         constant QuadU& u [[buffer(2)]]) {
    float d = dist.read(uint2(in.pos.xy)).r;
    if (d >= 0.9995) discard_fragment();              // untouched background
    float inv = 1.0 - d;                              // 0 = rim, 1 = centreline
    float border_end = SHADOW + u.misc.x;
    float4 shadow = float4(0.0, 0.0, 0.0, 0.5 * min(inv / SHADOW, 1.0) * u.border.a);
    float4 body = mix(u.outer, u.inner,
                      clamp((inv - border_end) / max(1.0 - border_end, 1e-4), 0.0, 1.0));
    float4 c = mix(shadow, u.border, smoothstep(SHADOW - u.misc.y, SHADOW + u.misc.y, inv));
    c = mix(c, body, smoothstep(border_end - u.misc.y, border_end + u.misc.y, inv));
    return c;
}

// composite: misc = (alpha, -, screen.xy)
fragment float4 comp_fs(QV in [[stage_in]],
                        texture2d<float> body [[texture(0)]],
                        constant QuadU& u [[buffer(2)]]) {
    float4 t = body.read(uint2(in.pos.xy));
    return float4(t.rgb, t.a * u.misc.x);
}
"""


class MetalSliderBodyRenderer:
    def __init__(self, spr, width: int, height: int):
        self.spr = spr
        self.core = c = spr.core
        self.width, self.height = width, height
        self._seg = c.pipe_create(_MSL, "seg_vs", "dist_fs", mc.R32F, mc.BLEND_OFF, depth=True)
        self._cap = c.pipe_create(_MSL, "cap_vs", "dist_fs", mc.R32F, mc.BLEND_OFF, depth=True)
        self._shade = c.pipe_create(_MSL, "quad_vs", "shade_fs", mc.RGBA8, mc.BLEND_OFF)
        self._comp = c.pipe_create(_MSL, "quad_vs", "comp_fs", mc.RGBA8, mc.BLEND_ALPHA)
        self._dist = c.tex_create(width, height, mc.R32F, target=True, depth=True,
                                  flags=mc.CLAMP | mc.NEAREST)
        self._pool: list[int] = []        # one RGBA body target per slider built this frame
        self._tent = _tent_mesh()
        self._disc = _disc_mesh()
        spr.add_frame_reset(self._reset)
        self._used = 0

    def _reset(self) -> None:
        self._used = 0

    def _target(self) -> int:
        if self._used == len(self._pool):
            self._pool.append(self.core.tex_create(
                self.width, self.height, mc.RGBA8, target=True,
                flags=mc.CLAMP | mc.NEAREST))
        self._used += 1
        return self._pool[self._used - 1]

    def build_body(self, path_points, radius_px, style=None, snake=(0.0, 1.0)):
        return self.build_merged([(path_points, snake)], radius_px, style)

    def build_merged(self, items, radius_px, style=None):
        with perf.T("slider_build"):
            return self._build_merged(items, radius_px, style)

    def _build_merged(self, items, radius_px, style=None):
        # geometry exactly as render/slider_body.py _build_merged
        style = style or BodyStyle()
        arrs = []
        for path_points, snake in items:
            pts = sub_path(list(path_points), snake[0], snake[1])
            if pts:
                arrs.append(np.asarray(pts, dtype="f4").reshape(-1, 2))
        if not arrs or radius_px <= 0.0:
            return BodyTexture(None, (0, 0, 0, 0), radius_px, empty=True)
        pad = radius_px + 2.0
        x0 = max(0.0, min(float(a[:, 0].min()) for a in arrs) - pad)
        y0 = max(0.0, min(float(a[:, 1].min()) for a in arrs) - pad)
        x1 = min(float(self.width), max(float(a[:, 0].max()) for a in arrs) + pad)
        y1 = min(float(self.height), max(float(a[:, 1].max()) for a in arrs) + pad)
        if x1 <= x0 or y1 <= y0:
            return BodyTexture(None, (0, 0, 0, 0), radius_px, empty=True)
        segs = []
        for arr in arrs:
            if len(arr) < 2:
                continue
            deltas = arr[1:] - arr[:-1]
            lens = np.hypot(deltas[:, 0], deltas[:, 1]).astype("f4")
            keep = lens > 1e-4
            n = int(keep.sum())
            if n:
                seg = np.empty((n, 5), dtype="f4")
                seg[:, 0:2] = arr[:-1][keep]
                seg[:, 2] = lens[keep]
                seg[:, 3:5] = deltas[keep] / lens[keep, None]
                segs.append(seg)
        seg_all = np.concatenate(segs) if segs else None
        caps = np.concatenate(arrs)

        geom = np.array([self.width, self.height, radius_px, 0.0], "f4")
        border = (*style.border_color, style.alpha)
        inner = (*shade2(style.body_color, style.inner_offset), style.inner_alpha * style.alpha)
        outer = (*shade2(style.body_color, style.outer_offset), style.outer_alpha * style.alpha)
        quad_u = np.array([*border, *inner, *outer, border_portion(style.border_width),
                           max(AA_BAND_PORTION, 0.75 / radius_px),
                           self.width, self.height], "f4")
        quad = np.array([x0, y0, x1, y0, x0, y1, x1, y1], "f4")
        target = self._target()
        c, dist, tent, disc = self.core, self._dist, self._tent, self._disc

        def build():
            c.set_pass(dist, (1.0, 1.0, 1.0, 1.0))
            if seg_all is not None:
                c.draw(self._seg, 6, vertices=tent, instances=seg_all, uniforms=geom,
                       prim=mc.TRIANGLE_STRIP, icount=len(seg_all))
            c.draw(self._cap, len(disc) // 3, vertices=disc, instances=caps,
                   uniforms=geom, icount=len(caps))
            c.set_pass(target, (0.0, 0.0, 0.0, 0.0))
            c.draw(self._shade, 4, vertices=quad, uniforms=quad_u, textures=(dist,),
                   prim=mc.TRIANGLE_STRIP)
        self.spr.add_pre(build)
        return BodyTexture(target, (x0, y0, x1, y1), radius_px)

    def draw_body(self, body, target_fbo=None, alpha: float = 1.0) -> None:
        if body.empty:
            return
        with perf.T("slider_composite"):
            x0, y0, x1, y1 = body.aabb
            quad = np.array([x0, y0, x1, y0, x0, y1, x1, y1], "f4")
            u = np.zeros(16, "f4")
            u[12], u[14], u[15] = alpha, self.width, self.height
            c, tex = self.core, body.texture
            self.spr.add_scene(lambda: c.draw(self._comp, 4, vertices=quad, uniforms=u,
                                              textures=(tex,), prim=mc.TRIANGLE_STRIP))

    def release(self) -> None:
        return None
