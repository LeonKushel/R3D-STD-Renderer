import Metal

// The two things every engine needs and none should rewrite: the instanced
// sprite pass and the frame's RGB -> yuv420p conversion.

// ---- sprites ------------------------------------------------------------------
// 14 tightly-packed floats per instance, the layout every engine's serialiser
// already produces. PACKED types are mandatory: MSL aligns float4 to 16 bytes.
// Each texture is sampled with ITS OWN sampler (mip chain or not, clamp or
// repeat), which is what makes a frame match the GL path that sets those per
// texture.
let R3D_SPRITE_SRC = """
#include <metal_stdlib>
using namespace metal;

struct Inst {
    packed_float2 center;
    packed_float2 size;
    float         rot;
    packed_float4 color;
    packed_float2 uv_off;
    packed_float2 uv_scale;
    float         tex;
};
static_assert(sizeof(Inst) == 56, "Inst must be 14 tightly-packed floats");

struct VOut {
    float4 pos [[position]];
    float2 uv;
    float4 color [[flat]];
    uint   tex [[flat]];
};

constant float2 QUAD[4] = { float2(0,0), float2(1,0), float2(0,1), float2(1,1) };

vertex VOut r3d_sprite_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                          constant Inst*  insts   [[buffer(0)]],
                          constant float4& screen [[buffer(1)]])   // w, h, ysign
{
    Inst I = insts[iid];
    float2 q = QUAD[vid];
    float2 local = (q - 0.5) * float2(I.size);
    float  c = cos(I.rot), s = sin(I.rot);
    float2 rotd = float2(local.x * c - local.y * s, local.x * s + local.y * c);
    float2 px = float2(I.center) + rotd;          // pixel space, top-left origin
    VOut o;
    o.pos   = float4(px.x / screen.x * 2.0 - 1.0,
                     (1.0 - px.y / screen.y * 2.0) * screen.z, 0.0, 1.0);
    o.uv    = q * float2(I.uv_scale) + float2(I.uv_off);
    o.color = float4(I.color);
    o.tex   = uint(I.tex + 0.5);
    return o;
}

fragment float4 r3d_sprite_fs(VOut in [[stage_in]],
                              array<texture2d<float>, 16> tex [[texture(0)]],
                              array<sampler, 16> smp [[sampler(0)]])
{
    return tex[in.tex].sample(smp[in.tex], in.uv) * in.color;
}
"""

func r3dMakeSpritePipelines(_ dev: MTLDevice)
    -> (MTLRenderPipelineState, MTLRenderPipelineState)?
{
    guard let lib = try? dev.makeLibrary(source: R3D_SPRITE_SRC, options: nil),
          let vs = lib.makeFunction(name: "r3d_sprite_vs"),
          let fs = lib.makeFunction(name: "r3d_sprite_fs") else { return nil }
    func pipe(_ additive: Bool) -> MTLRenderPipelineState? {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vs; d.fragmentFunction = fs
        let a = d.colorAttachments[0]!
        a.pixelFormat = .rgba8Unorm
        a.isBlendingEnabled = true
        a.rgbBlendOperation = .add; a.alphaBlendOperation = .add
        a.sourceRGBBlendFactor = .sourceAlpha; a.sourceAlphaBlendFactor = .sourceAlpha
        a.destinationRGBBlendFactor = additive ? .one : .oneMinusSourceAlpha
        a.destinationAlphaBlendFactor = additive ? .one : .oneMinusSourceAlpha
        return try? dev.makeRenderPipelineState(descriptor: d)
    }
    guard let pn = pipe(false), let pa = pipe(true) else { return nil }
    return (pn, pa)
}

// ---- RGB -> yuv420p, ffmpeg's own arithmetic ------------------------------------
// libswscale's general path for rgb24/rgba at default flags, recovered from its
// own output and exact on every sample tested (21 M, ffmpeg 8.1):
//   Y   = ((((RY*R + GY*G + BY*B) + (0x801 << 8)) >> 9) + 32) >> 6
//   row = ((CU*R2 + CG*G2 + CB*B2) + (0x4001 << 9)) >> 10        R2 = two pixels across, summed
//   C   = clamp((sum(TAP[k] * row[2j-3+k]) + (1 << 18)) >> 19, 0, 255)   rows clamped at the edges
// The eight weights are the FULL-PRECISION ones (sum 8192). Their 8-bit rounding
// (-4 -11 31 112) is right on only 92% of noise samples.
//
// The planes always leave BOTTOM-UP, the row order std's ffmpeg command takes
// before its vflip: as-is when the frame is stored bottom-up (GL order), flipped
// when it is top-down. The conversion is the same read in either direction
// (symmetric weights, symmetric clamp).
let R3D_YUV_SRC = """
#include <metal_stdlib>
using namespace metal;

constant int RY =  8414, GY =  16519, BY =  3208;
constant int RU = -4865, GU =  -9528, BU = 14392;
constant int RV = 14392, GV = -12061, BV = -2332;
constant int TAP[8] = { -116, -344, 984, 3572, 3572, 984, -344, -116 };

struct Geom { int w; int h; int flip; int pad; };

static inline int3 rgb8(texture2d<float, access::read> src, uint x, uint y) {
    float4 c = src.read(uint2(x, y));
    return int3(int(c.r * 255.0 + 0.5), int(c.g * 255.0 + 0.5), int(c.b * 255.0 + 0.5));
}

kernel void r3d_yuv420p(texture2d<float, access::read> src [[texture(0)]],
                        device uchar  *dst [[buffer(0)]],
                        constant Geom& g   [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]])
{
    int cw = g.w >> 1, ch = g.h >> 1;
    if (int(gid.x) >= cw || int(gid.y) >= ch) return;
    int x = int(gid.x) * 2, y = int(gid.y) * 2;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int3 p = rgb8(src, uint(x + dx), uint(y + dy));
            int Y = ((((RY * p.r + GY * p.g + BY * p.b) + (0x801 << 8)) >> 9) + 32) >> 6;
            int oy = g.flip != 0 ? (g.h - 1 - (y + dy)) : (y + dy);
            dst[oy * g.w + (x + dx)] = uchar(clamp(Y, 0, 255));
        }
    }
    int accU = 0, accV = 0;
    for (int k = 0; k < 8; ++k) {
        int sy = clamp(y + k - 3, 0, g.h - 1);
        int3 s = rgb8(src, uint(x), uint(sy)) + rgb8(src, uint(x + 1), uint(sy));
        accU += TAP[k] * (((RU * s.r + GU * s.g + BU * s.b) + (0x4001 << 9)) >> 10);
        accV += TAP[k] * (((RV * s.r + GV * s.g + BV * s.b) + (0x4001 << 9)) >> 10);
    }
    int uoff = g.w * g.h;
    int ci = (g.flip != 0 ? (ch - 1 - int(gid.y)) : int(gid.y)) * cw + int(gid.x);
    dst[uoff + ci]           = uchar(clamp((accU + (1 << 18)) >> 19, 0, 255));
    dst[uoff + cw * ch + ci] = uchar(clamp((accV + (1 << 18)) >> 19, 0, 255));
}
"""

func r3dMakeYuvPipeline(_ dev: MTLDevice) -> MTLComputePipelineState? {
    guard let lib = try? dev.makeLibrary(source: R3D_YUV_SRC, options: nil),
          let fn = lib.makeFunction(name: "r3d_yuv420p") else { return nil }
    return try? dev.makeComputePipelineState(function: fn)
}
