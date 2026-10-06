import Metal
import Foundation

// R3D Metal core, second generation: a C ABI for ctypes.
//
// The first core (osu-catch-renderer `_metal`) could create a texture, draw
// sprites into the frame and hand the frame back. std needs more, and none of it
// is std-specific, so it lives here for every engine:
//   * textures can be UPDATED and FREED, carry their own sampling state, and come
//     in the formats the passes need (RGBA8, R8, R32F, RG32F, R32I, RG32I);
//   * a texture can be a render TARGET (optionally depth-tested), and any pass
//     can draw into one;
//   * pipelines are created from MSL the caller supplies, so a new pass is a
//     shader string on the Python side, not a new Swift file and a rebuild;
//   * clipping (scissor);
//   * the frame's RGB -> yuv420p conversion is chained into the frame's own
//     command buffer with ffmpeg's exact arithmetic, so the encoder gets the
//     shared buffer the GPU wrote and nothing is copied on the way.
//
// Threading: every call comes from ONE thread (the render thread). A pointer
// returned by r3d_frame_acquire stays valid until its ring slot comes round
// again, which the caller sizes the ring for.

// Pixel formats, by the small integer the Python side uses.
func r3dFormat(_ code: Int32) -> (MTLPixelFormat, Int)? {
    switch code {
    case 0: return (.rgba8Unorm, 4)
    case 1: return (.r8Unorm, 1)
    case 2: return (.r32Float, 4)
    case 3: return (.rg32Float, 8)
    case 4: return (.r32Sint, 4)
    case 5: return (.rg32Sint, 8)
    default: return nil
    }
}

final class Tex {
    let t: MTLTexture
    let bpp: Int
    let sampler: Int              // bit0 mip chain, bit1 clamp (else repeat), bit2 nearest
    var depth: MTLTexture?        // present on depth-tested targets
    init(_ t: MTLTexture, _ bpp: Int, _ sampler: Int) {
        self.t = t; self.bpp = bpp; self.sampler = sampler
    }
}

final class Pipe {
    let state: MTLRenderPipelineState
    let depth: MTLDepthStencilState?
    init(_ s: MTLRenderPipelineState, _ d: MTLDepthStencilState?) { state = s; depth = d }
}

public final class Core {
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let w: Int, h: Int, ring: Int
    let bpr: Int
    // +1: frames are stored top-down (row 0 = top of the picture).
    // -1: GL's order, row 0 = BOTTOM. An engine that must match its own GL
    // output pixel for pixel needs this: a quad edge that lands exactly on a
    // pixel centre is given to one neighbour or the other by the fill rule,
    // which is applied in framebuffer space, so a flipped framebuffer moves
    // such an edge by one row (seen as thin HUD lines one row off).
    let ysign: Float
    let yuvLen: Int
    var frameBufs: [MTLBuffer] = []       // shared memory under each frame target
    var frameTex: [MTLTexture] = []
    var yuvBufs: [MTLBuffer] = []
    var scratch: [MTLBuffer] = []         // per-slot bump allocator for vertex/instance data
    var scratchOff = 0
    static let SCRATCH = 16 << 20
    var inflight: [MTLCommandBuffer?]
    var head = 0, tail = 0
    var cur: MTLCommandBuffer?
    var enc: MTLRenderCommandEncoder?
    var encW = 0, encH = 0
    var tex: [Int32: Tex] = [:]
    var nextTex: Int32 = 1
    var pipes: [Int32: Pipe] = [:]
    var nextPipe: Int32 = 1
    var samplers: [MTLSamplerState] = []
    var spriteNormal: MTLRenderPipelineState!
    var spriteAdd: MTLRenderPipelineState!
    var yuvPipe: MTLComputePipelineState!
    var lastErr = ""

    init?(w: Int, h: Int, ring: Int, bottomUp: Bool) {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue() else { return nil }
        dev = d; queue = q; self.w = w; self.h = h
        ysign = bottomUp ? -1.0 : 1.0
        self.ring = max(3, ring)
        let align = d.minimumLinearTextureAlignment(for: .rgba8Unorm)
        bpr = ((w * 4) + align - 1) / align * align
        yuvLen = w * h + 2 * ((w / 2) * (h / 2))
        inflight = Array(repeating: nil, count: self.ring)
        let td = MTLTextureDescriptor()
        td.pixelFormat = .rgba8Unorm
        td.width = w; td.height = h
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = .shared
        for _ in 0..<self.ring {
            guard let b = d.makeBuffer(length: bpr * h, options: .storageModeShared),
                  let t = b.makeTexture(descriptor: td, offset: 0, bytesPerRow: bpr),
                  let y = d.makeBuffer(length: max(yuvLen, 16), options: .storageModeShared),
                  let s = d.makeBuffer(length: Core.SCRATCH, options: .storageModeShared)
            else { return nil }
            frameBufs.append(b); frameTex.append(t); yuvBufs.append(y); scratch.append(s)
        }
        for flags in 0..<8 {
            let sd = MTLSamplerDescriptor()
            let nearest = (flags & 4) != 0
            sd.minFilter = nearest ? .nearest : .linear
            sd.magFilter = nearest ? .nearest : .linear
            sd.mipFilter = (flags & 1) != 0 ? .linear : .notMipmapped
            let mode: MTLSamplerAddressMode = (flags & 2) != 0 ? .clampToEdge : .repeat
            sd.sAddressMode = mode; sd.tAddressMode = mode
            guard let s = d.makeSamplerState(descriptor: sd) else { return nil }
            samplers.append(s)
        }
        guard let (pn, pa) = r3dMakeSpritePipelines(d),
              let yp = r3dMakeYuvPipeline(d) else { return nil }
        spriteNormal = pn; spriteAdd = pa; yuvPipe = yp
    }

    /// Bump-allocate `len` bytes in this frame's scratch buffer and copy `src` in.
    func stash(_ src: UnsafeRawPointer, _ len: Int) -> Int? {
        let off = (scratchOff + 15) & ~15
        if off + len > Core.SCRATCH { return nil }
        memcpy(scratch[head % ring].contents().advanced(by: off), src, len)
        scratchOff = off + len
        return off
    }
}

private var cores: [Int32: Core] = [:]
private var nextId: Int32 = 1
private let lock = NSLock()
func r3dCore(_ id: Int32) -> Core? { lock.lock(); defer { lock.unlock() }; return cores[id] }

@_cdecl("r3d_create")
public func r3d_create(_ w: Int32, _ h: Int32, _ ring: Int32, _ bottomUp: Int32) -> Int32 {
    guard let c = Core(w: Int(w), h: Int(h), ring: Int(ring), bottomUp: bottomUp != 0)
    else { return -1 }
    lock.lock(); defer { lock.unlock() }
    let id = nextId; nextId += 1
    cores[id] = c
    return id
}

@_cdecl("r3d_destroy")
public func r3d_destroy(_ id: Int32) -> Int32 {
    guard let c = r3dCore(id) else { return -1 }
    for cb in c.inflight { cb?.waitUntilCompleted() }
    lock.lock(); defer { lock.unlock() }
    cores.removeValue(forKey: id)
    return 0
}

@_cdecl("r3d_info")
public func r3d_info(_ id: Int32, _ outBpr: UnsafeMutablePointer<Int32>,
                     _ outYuv: UnsafeMutablePointer<Int32>,
                     _ name: UnsafeMutablePointer<CChar>, _ cap: Int32) -> Int32 {
    guard let c = r3dCore(id) else { return -1 }
    outBpr.pointee = Int32(c.bpr); outYuv.pointee = Int32(c.yuvLen)
    _ = c.dev.name.withCString { strlcpy(name, $0, Int(cap)) }
    return 0
}

@_cdecl("r3d_error")
public func r3d_error(_ id: Int32, _ out: UnsafeMutablePointer<CChar>, _ cap: Int32) -> Int32 {
    guard let c = r3dCore(id) else { return -1 }
    _ = c.lastErr.withCString { strlcpy(out, $0, Int(cap)) }
    return 0
}

// ---- textures ---------------------------------------------------------------

/// Create a texture. `px` may be NULL (targets). `mip != 0` builds the chain.
/// `flags`: bit0 sample through the mip chain, bit1 clamp to edge (else repeat),
/// bit2 nearest. `target != 0` makes it drawable; `depth != 0` adds a depth buffer.
@_cdecl("r3d_tex_create")
public func r3d_tex_create(_ id: Int32, _ w: Int32, _ h: Int32, _ fmt: Int32,
                           _ px: UnsafeRawPointer?, _ mip: Int32, _ flags: Int32,
                           _ target: Int32, _ depth: Int32) -> Int32 {
    guard let c = r3dCore(id), let (pf, bpp) = r3dFormat(fmt) else { return -1 }
    let td = MTLTextureDescriptor()
    td.pixelFormat = pf
    td.width = Int(w); td.height = Int(h)
    td.usage = target != 0 ? [.renderTarget, .shaderRead] : [.shaderRead]
    td.storageMode = .shared
    let levels = Int(floor(log2(Double(max(w, h))))) + 1
    let mipped = mip != 0 && levels > 1
    td.mipmapLevelCount = mipped ? levels : 1
    guard let t = c.dev.makeTexture(descriptor: td) else { return -1 }
    let x = Tex(t, bpp, Int(flags & 7) & (mipped ? 7 : 6))
    if depth != 0 {
        let dd = MTLTextureDescriptor()
        dd.pixelFormat = .depth32Float
        dd.width = Int(w); dd.height = Int(h)
        dd.usage = [.renderTarget]
        dd.storageMode = .private
        guard let dt = c.dev.makeTexture(descriptor: dd) else { return -1 }
        x.depth = dt
    }
    let tid = c.nextTex; c.nextTex += 1
    c.tex[tid] = x
    if let px = px { _ = r3dUpload(c, x, px) }
    return tid
}

func r3dUpload(_ c: Core, _ x: Tex, _ px: UnsafeRawPointer) -> Int32 {
    let t = x.t
    t.replace(region: MTLRegionMake2D(0, 0, t.width, t.height), mipmapLevel: 0,
              withBytes: px, bytesPerRow: t.width * x.bpp)
    if t.mipmapLevelCount > 1 {
        guard let cb = c.queue.makeCommandBuffer(),
              let bl = cb.makeBlitCommandEncoder() else { return -1 }
        bl.generateMipmaps(for: t)
        bl.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    return 0
}

/// Replace a texture's whole contents (same size and format), mips rebuilt.
/// The caller must not have the texture in a frame that is still being encoded.
@_cdecl("r3d_tex_update")
public func r3d_tex_update(_ id: Int32, _ tid: Int32, _ px: UnsafeRawPointer) -> Int32 {
    guard let c = r3dCore(id), let x = c.tex[tid] else { return -1 }
    return r3dUpload(c, x, px)
}

@_cdecl("r3d_tex_free")
public func r3d_tex_free(_ id: Int32, _ tid: Int32) -> Int32 {
    guard let c = r3dCore(id) else { return -1 }
    // frames still in flight keep the MTLTexture alive through their command
    // buffers, so dropping our reference here is safe
    return c.tex.removeValue(forKey: tid) != nil ? 0 : -1
}

/// Copy one mip level of a texture out (tests and self-checks only: it waits).
@_cdecl("r3d_tex_read")
public func r3d_tex_read(_ id: Int32, _ tid: Int32, _ level: Int32,
                         _ out: UnsafeMutableRawPointer) -> Int32 {
    guard let c = r3dCore(id), let x = c.tex[tid],
          Int(level) < x.t.mipmapLevelCount else { return -1 }
    for cb in c.inflight { cb?.waitUntilCompleted() }
    let w = max(1, x.t.width >> Int(level)), h = max(1, x.t.height >> Int(level))
    x.t.getBytes(out, bytesPerRow: w * x.bpp, from: MTLRegionMake2D(0, 0, w, h),
                 mipmapLevel: Int(level))
    return 0
}

/// Write one mip level directly (level sizes halve, rounding down, min 1).
@_cdecl("r3d_tex_set_level")
public func r3d_tex_set_level(_ id: Int32, _ tid: Int32, _ level: Int32,
                              _ px: UnsafeRawPointer) -> Int32 {
    guard let c = r3dCore(id), let x = c.tex[tid],
          Int(level) < x.t.mipmapLevelCount else { return -1 }
    let w = max(1, x.t.width >> Int(level)), h = max(1, x.t.height >> Int(level))
    x.t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: Int(level),
                withBytes: px, bytesPerRow: w * x.bpp)
    return 0
}

@_cdecl("r3d_tex_levels")
public func r3d_tex_levels(_ id: Int32, _ tid: Int32) -> Int32 {
    guard let c = r3dCore(id), let x = c.tex[tid] else { return -1 }
    return Int32(x.t.mipmapLevelCount)
}

// ---- pipelines from caller-supplied MSL ---------------------------------------

/// `blend`: 0 off, 1 (srcAlpha, 1-srcAlpha), 2 (srcAlpha, one), 3 (one, one).
/// `depthMode`: 0 none, 1 less + write. `safeMath != 0` compiles without
/// fast-math. Returns a pipeline id, or -1 (r3d_error).
@_cdecl("r3d_pipe_create")
public func r3d_pipe_create(_ id: Int32, _ src: UnsafePointer<CChar>,
                            _ vs: UnsafePointer<CChar>, _ fs: UnsafePointer<CChar>,
                            _ fmt: Int32, _ blend: Int32, _ depthMode: Int32,
                            _ safeMath: Int32) -> Int32 {
    guard let c = r3dCore(id), let (pf, _) = r3dFormat(fmt) else { return -1 }
    do {
        // Fast-math is what the GL driver gives a shader, so a pass ported from
        // GL matches it best compiled the same way (std's slider passes: exact
        // with it, off by one in 0.0015% of values without). A pass that settles
        // rounding ties itself (the health bar's fma) asks for safe maths.
        let opts = MTLCompileOptions()
        if safeMath != 0 {
            if #available(macOS 15.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
        }
        let lib = try c.dev.makeLibrary(source: String(cString: src), options: opts)
        guard let vf = lib.makeFunction(name: String(cString: vs)),
              let ff = lib.makeFunction(name: String(cString: fs)) else {
            c.lastErr = "function not found"; return -1
        }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vf; d.fragmentFunction = ff
        let a = d.colorAttachments[0]!
        a.pixelFormat = pf
        if blend != 0 {
            a.isBlendingEnabled = true
            a.rgbBlendOperation = .add; a.alphaBlendOperation = .add
            let s: MTLBlendFactor = blend == 3 ? .one : .sourceAlpha
            let dst: MTLBlendFactor = blend == 1 ? .oneMinusSourceAlpha : .one
            a.sourceRGBBlendFactor = s; a.sourceAlphaBlendFactor = s
            a.destinationRGBBlendFactor = dst; a.destinationAlphaBlendFactor = dst
        }
        var ds: MTLDepthStencilState? = nil
        if depthMode != 0 {
            d.depthAttachmentPixelFormat = .depth32Float
            let dd = MTLDepthStencilDescriptor()
            dd.depthCompareFunction = .less
            dd.isDepthWriteEnabled = true
            ds = c.dev.makeDepthStencilState(descriptor: dd)
        }
        let st = try c.dev.makeRenderPipelineState(descriptor: d)
        let pid = c.nextPipe; c.nextPipe += 1
        c.pipes[pid] = Pipe(st, ds)
        return pid
    } catch {
        c.lastErr = "\(error)"
        return -1
    }
}

// ---- a frame --------------------------------------------------------------

@_cdecl("r3d_frame_begin")
public func r3d_frame_begin(_ id: Int32) -> Int32 {
    guard let c = r3dCore(id) else { return -1 }
    guard let cb = c.queue.makeCommandBuffer() else { return -2 }
    c.cur = cb; c.enc = nil; c.scratchOff = 0
    return 0
}

/// Start a render pass. `tid == 0` is this frame's own target. `load`: 0 clear
/// to (r,g,b,a), 1 keep what is there. A depth-tested target clears depth to 1.
@_cdecl("r3d_pass")
public func r3d_pass(_ id: Int32, _ tid: Int32, _ load: Int32,
                     _ r: Float, _ g: Float, _ b: Float, _ a: Float) -> Int32 {
    guard let c = r3dCore(id), let cb = c.cur else { return -1 }
    c.enc?.endEncoding(); c.enc = nil
    let rp = MTLRenderPassDescriptor()
    var target: MTLTexture
    if tid == 0 {
        target = c.frameTex[c.head % c.ring]
    } else {
        guard let x = c.tex[tid] else { return -2 }
        target = x.t
        if let dt = x.depth {
            rp.depthAttachment.texture = dt
            rp.depthAttachment.loadAction = .clear
            rp.depthAttachment.storeAction = .dontCare
            rp.depthAttachment.clearDepth = 1.0
        }
    }
    rp.colorAttachments[0].texture = target
    rp.colorAttachments[0].loadAction = load == 0 ? .clear : .load
    rp.colorAttachments[0].storeAction = .store
    rp.colorAttachments[0].clearColor = MTLClearColor(
        red: Double(r), green: Double(g), blue: Double(b), alpha: Double(a))
    guard let e = cb.makeRenderCommandEncoder(descriptor: rp) else { return -3 }
    c.enc = e; c.encW = target.width; c.encH = target.height
    return 0
}

/// Clip later draws in this pass to a rectangle (top-left origin). w <= 0 resets.
@_cdecl("r3d_scissor")
public func r3d_scissor(_ id: Int32, _ x: Int32, _ y: Int32, _ w: Int32, _ h: Int32) -> Int32 {
    guard let c = r3dCore(id), let e = c.enc else { return -1 }
    if w <= 0 || h <= 0 {
        e.setScissorRect(MTLScissorRect(x: 0, y: 0, width: c.encW, height: c.encH))
        return 0
    }
    let x0 = max(0, min(Int(x), c.encW)), y0 = max(0, min(Int(y), c.encH))
    let x1 = max(x0, min(Int(x) + Int(w), c.encW)), y1 = max(y0, min(Int(y) + Int(h), c.encH))
    e.setScissorRect(MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
    return 0
}

/// One instanced sprite draw into the current pass. `inst` is count x 14 floats
/// (centre, size, rot, colour, uv_off, uv_scale, texture unit); `texIds` maps
/// unit 0..nTex-1 to textures (at most 16), each sampled with its own state.
@_cdecl("r3d_sprites")
public func r3d_sprites(_ id: Int32, _ inst: UnsafeRawPointer, _ count: Int32,
                        _ texIds: UnsafePointer<Int32>, _ nTex: Int32,
                        _ additive: Int32) -> Int32 {
    guard let c = r3dCore(id), let e = c.enc else { return -1 }
    let n = Int(count)
    if n <= 0 { return 0 }
    guard let off = c.stash(inst, n * 56) else { return -2 }
    e.setRenderPipelineState(additive != 0 ? c.spriteAdd : c.spriteNormal)
    e.setVertexBuffer(c.scratch[c.head % c.ring], offset: off, index: 0)
    var screen = SIMD4<Float>(Float(c.encW), Float(c.encH), c.ysign, 0)
    e.setVertexBytes(&screen, length: 16, index: 1)
    for u in 0..<Int(nTex) {
        guard let x = c.tex[texIds[u]] else { return -3 }
        e.setFragmentTexture(x.t, index: u)
        e.setFragmentSamplerState(c.samplers[x.sampler], index: u)
    }
    e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: n)
    return 0
}

/// A draw with a caller-made pipeline. Vertex data (buffer 0) and instance data
/// (buffer 1) are optional and copied; `u` is a small uniform block bound to
/// both stages at buffer 2. `prim`: 0 triangles, 1 triangle strip. Textures are
/// bound to fragment AND vertex stages with their own samplers.
@_cdecl("r3d_draw")
public func r3d_draw(_ id: Int32, _ pid: Int32,
                     _ v: UnsafeRawPointer?, _ vlen: Int32,
                     _ inst: UnsafeRawPointer?, _ ilen: Int32,
                     _ u: UnsafeRawPointer?, _ ulen: Int32,
                     _ texIds: UnsafePointer<Int32>?, _ nTex: Int32,
                     _ prim: Int32, _ vcount: Int32, _ icount: Int32) -> Int32 {
    guard let c = r3dCore(id), let e = c.enc, let p = c.pipes[pid] else { return -1 }
    e.setRenderPipelineState(p.state)
    if let ds = p.depth { e.setDepthStencilState(ds) }
    let sb = c.scratch[c.head % c.ring]
    if let v = v, vlen > 0 {
        guard let off = c.stash(v, Int(vlen)) else { return -2 }
        e.setVertexBuffer(sb, offset: off, index: 0)
    }
    if let i = inst, ilen > 0 {
        guard let off = c.stash(i, Int(ilen)) else { return -2 }
        e.setVertexBuffer(sb, offset: off, index: 1)
    }
    if let u = u, ulen > 0 {
        e.setVertexBytes(u, length: Int(ulen), index: 2)
        e.setFragmentBytes(u, length: Int(ulen), index: 2)
    }
    if let t = texIds {
        for k in 0..<Int(nTex) {
            guard let x = c.tex[t[k]] else { return -3 }
            e.setFragmentTexture(x.t, index: k)
            e.setFragmentSamplerState(c.samplers[x.sampler], index: k)
        }
    }
    e.drawPrimitives(type: prim == 1 ? .triangleStrip : .triangle, vertexStart: 0,
                     vertexCount: Int(vcount), instanceCount: max(1, Int(icount)))
    return 0
}

/// Finish the frame. `yuv != 0` chains the RGB -> yuv420p conversion of the
/// frame target into the same command buffer. Never waits.
@_cdecl("r3d_frame_commit")
public func r3d_frame_commit(_ id: Int32, _ yuv: Int32) -> Int32 {
    guard let c = r3dCore(id), let cb = c.cur else { return -1 }
    c.enc?.endEncoding(); c.enc = nil
    let s = c.head % c.ring
    if yuv != 0 {
        guard let e = cb.makeComputeCommandEncoder() else { return -3 }
        // planes leave in the frame's own row order when it is GL's (bottom-up),
        // and are flipped to bottom-up when the frame is top-down
        var g = (Int32(c.w), Int32(c.h), Int32(c.ysign > 0 ? 1 : 0), Int32(0))
        e.setComputePipelineState(c.yuvPipe)
        e.setTexture(c.frameTex[s], index: 0)
        e.setBuffer(c.yuvBufs[s], offset: 0, index: 0)
        withUnsafeBytes(of: &g) { e.setBytes($0.baseAddress!, length: 16, index: 1) }
        let tw = c.yuvPipe.threadExecutionWidth
        let th = max(1, c.yuvPipe.maxTotalThreadsPerThreadgroup / tw)
        e.dispatchThreads(MTLSize(width: c.w / 2, height: c.h / 2, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        e.endEncoding()
    }
    cb.commit()
    c.inflight[s] = cb
    c.head += 1
    c.cur = nil
    return 0
}

/// The oldest finished frame: `yuv != 0` its yuv420p bytes (rows bottom-up, the
/// order std's ffmpeg command expects before its vflip), else its RGBA pixels
/// (`bpr` bytes per row, in the core's row order: see `ysign`). NULL while fewer than `minInFlight` frames
/// are in flight, unless `force`. The pointer is the GPU's own shared buffer.
@_cdecl("r3d_frame_acquire")
public func r3d_frame_acquire(_ id: Int32, _ minInFlight: Int32, _ force: Int32,
                              _ yuv: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dCore(id), c.tail < c.head else { return nil }
    if force == 0 && (c.head - c.tail) < Int(minInFlight) { return nil }
    let s = c.tail % c.ring
    c.inflight[s]?.waitUntilCompleted()
    c.inflight[s] = nil
    c.tail += 1
    return yuv != 0 ? c.yuvBufs[s].contents() : c.frameBufs[s].contents()
}

/// Frames submitted and not yet acquired.
@_cdecl("r3d_frames_in_flight")
public func r3d_frames_in_flight(_ id: Int32) -> Int32 {
    guard let c = r3dCore(id) else { return -1 }
    return Int32(c.head - c.tail)
}
