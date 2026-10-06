"""ctypes face of libr3dcore.dylib (src/core.swift): the shared Metal core.

Thin on purpose. One method per exported function, numpy in, numpy views out;
everything engine-specific lives in renderer.py. Importing this module loads
the library, so import it only on a Mac and only when Metal is wanted."""
from __future__ import annotations

import ctypes
import os

import numpy as np

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libr3dcore.dylib"))
_i, _f, _p = ctypes.c_int32, ctypes.c_float, ctypes.c_void_p
_ip = ctypes.POINTER(ctypes.c_int32)


def _sig(name, res, *args):
    fn = getattr(_LIB, name)
    fn.restype, fn.argtypes = res, list(args)
    return fn


_create = _sig("r3d_create", _i, _i, _i, _i, _i)
_destroy = _sig("r3d_destroy", _i, _i)
_info = _sig("r3d_info", _i, _i, _ip, _ip, ctypes.c_char_p, _i)
_error = _sig("r3d_error", _i, _i, ctypes.c_char_p, _i)
_tex_create = _sig("r3d_tex_create", _i, _i, _i, _i, _i, _p, _i, _i, _i, _i)
_tex_update = _sig("r3d_tex_update", _i, _i, _i, _p)
_tex_free = _sig("r3d_tex_free", _i, _i, _i)
_tex_read = _sig("r3d_tex_read", _i, _i, _i, _i, _p)
_tex_set_level = _sig("r3d_tex_set_level", _i, _i, _i, _i, _p)
_tex_levels = _sig("r3d_tex_levels", _i, _i, _i)
_pipe_create = _sig("r3d_pipe_create", _i, _i, ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.c_char_p, _i, _i, _i, _i)
_frame_begin = _sig("r3d_frame_begin", _i, _i)
_pass = _sig("r3d_pass", _i, _i, _i, _i, _f, _f, _f, _f)
_scissor = _sig("r3d_scissor", _i, _i, _i, _i, _i, _i)
_sprites = _sig("r3d_sprites", _i, _i, _p, _i, _ip, _i, _i)
_draw = _sig("r3d_draw", _i, _i, _i, _p, _i, _p, _i, _p, _i, _ip, _i, _i, _i, _i)
_commit = _sig("r3d_frame_commit", _i, _i, _i)
_acquire = _sig("r3d_frame_acquire", _p, _i, _i, _i, _i)
_in_flight = _sig("r3d_frames_in_flight", _i, _i)

# pixel formats (src/core.swift r3dFormat)
RGBA8, R8, R32F, RG32F, R32I, RG32I = range(6)
_DTYPE = {RGBA8: ("u1", 4), R8: ("u1", 1), R32F: ("f4", 1), RG32F: ("f4", 2),
          R32I: ("i4", 1), RG32I: ("i4", 2)}
# sampling flags
MIP, CLAMP, NEAREST = 1, 2, 4
# blend modes for pipelines
BLEND_OFF, BLEND_ALPHA, BLEND_ADD, BLEND_ONE_ONE = range(4)
TRIANGLES, TRIANGLE_STRIP = 0, 1


class MetalError(RuntimeError):
    pass


class Core:
    def __init__(self, width: int, height: int, ring: int = 12,
                 bottom_up: bool = False):
        """bottom_up: store frames in GL's row order (row 0 = bottom), which is
        what makes edges that land exactly on a pixel centre fall where the GL
        renderer puts them."""
        self.width, self.height, self.ring = width, height, ring
        self.bottom_up = bottom_up
        self.id = _create(width, height, ring, int(bottom_up))
        if self.id < 0:
            raise MetalError("no Metal device, or the frame ring could not be made")
        bpr, ylen = ctypes.c_int32(), ctypes.c_int32()
        name = ctypes.create_string_buffer(128)
        _info(self.id, ctypes.byref(bpr), ctypes.byref(ylen), name, 128)
        self.bpr, self.yuv_len = bpr.value, ylen.value
        self.device = name.value.decode()
        self._fmt: dict[int, tuple[int, int, int]] = {}      # tex id -> (w, h, fmt)

    def _check(self, rc: int, what: str) -> int:
        if rc < 0:
            buf = ctypes.create_string_buffer(2048)
            _error(self.id, buf, 2048)
            raise MetalError(f"{what} failed ({rc}) {buf.value.decode()}")
        return rc

    # ---- textures ----
    def tex_create(self, w: int, h: int, fmt: int = RGBA8, data=None,
                   mip: bool = False, flags: int = 0, target: bool = False,
                   depth: bool = False) -> int:
        ptr = None
        if data is not None:
            data = np.ascontiguousarray(data)
            ptr = data.ctypes.data_as(_p)
        tid = self._check(_tex_create(self.id, w, h, fmt, ptr, int(mip), flags,
                                      int(target), int(depth)), "tex_create")
        self._fmt[tid] = (w, h, fmt)
        return tid

    def tex_update(self, tid: int, data) -> None:
        data = np.ascontiguousarray(data)
        self._check(_tex_update(self.id, tid, data.ctypes.data_as(_p)), "tex_update")

    def tex_free(self, tid: int) -> None:
        _tex_free(self.id, tid)
        self._fmt.pop(tid, None)

    def tex_read(self, tid: int, level: int = 0) -> np.ndarray:
        """One mip level as an array (h, w[, channels]). Waits for the GPU: tests only."""
        w, h, fmt = self._fmt[tid]
        w, h = max(1, w >> level), max(1, h >> level)
        dt, ch = _DTYPE[fmt]
        out = np.empty((h, w, ch) if ch > 1 else (h, w), dt)
        self._check(_tex_read(self.id, tid, level, out.ctypes.data_as(_p)), "tex_read")
        return out

    def tex_levels(self, tid: int) -> int:
        return _tex_levels(self.id, tid)

    def tex_set_level(self, tid: int, level: int, data) -> None:
        data = np.ascontiguousarray(data)
        self._check(_tex_set_level(self.id, tid, level, data.ctypes.data_as(_p)),
                    "tex_set_level")

    # ---- pipelines ----
    def pipe_create(self, src: str, vs: str, fs: str, fmt: int = RGBA8,
                    blend: int = BLEND_OFF, depth: bool = False,
                    safe_math: bool = False) -> int:
        return self._check(_pipe_create(self.id, src.encode(), vs.encode(),
                                        fs.encode(), fmt, blend, int(depth),
                                        int(safe_math)), "pipe_create")

    # ---- a frame ----
    def frame_begin(self) -> None:
        self._check(_frame_begin(self.id), "frame_begin")

    def set_pass(self, tid: int = 0, clear=None) -> None:
        """Start a pass on texture `tid` (0 = this frame). `clear` is an RGBA
        tuple, or None to keep what the target holds."""
        c = clear or (0.0, 0.0, 0.0, 0.0)
        self._check(_pass(self.id, tid, 0 if clear is not None else 1,
                          c[0], c[1], c[2], c[3]), "pass")

    def scissor(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> None:
        _scissor(self.id, x, y, w, h)

    def sprites(self, inst: np.ndarray, tex_ids, additive: bool) -> None:
        ids = (ctypes.c_int32 * len(tex_ids))(*tex_ids)
        self._check(_sprites(self.id, inst.ctypes.data_as(_p), len(inst), ids,
                             len(tex_ids), int(additive)), "sprites")

    def draw(self, pipe: int, vcount: int, *, vertices=None, instances=None,
             uniforms=None, textures=(), prim: int = TRIANGLES, icount: int = 1) -> None:
        def raw(a):
            if a is None:
                return None, 0
            a = np.ascontiguousarray(a)
            return a.ctypes.data_as(_p), a.nbytes
        vp, vl = raw(vertices)
        ip, il = raw(instances)
        up, ul = raw(uniforms)
        ids = (ctypes.c_int32 * len(textures))(*textures) if textures else None
        self._check(_draw(self.id, pipe, vp, vl, ip, il, up, ul, ids, len(textures),
                          prim, vcount, icount), "draw")

    def frame_commit(self, yuv: bool) -> None:
        self._check(_commit(self.id, int(yuv)), "frame_commit")

    def in_flight(self) -> int:
        return _in_flight(self.id)

    def frame_acquire(self, min_in_flight: int, force: bool, yuv: bool):
        """Oldest finished frame as a VIEW of the GPU's shared buffer, or None.
        yuv: flat uint8, rows bottom-up. rgba: (h, w, 4) uint8 in the core's
        row order (bottom-up when bottom_up)."""
        ptr = _acquire(self.id, min_in_flight, int(force), int(yuv))
        if not ptr:
            return None
        if yuv:
            return np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (self.yuv_len,))
        a = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
            (self.height, self.bpr))
        return a[:, :self.width * 4].reshape(self.height, self.width, 4)

    def close(self) -> None:
        if self.id >= 0:
            _destroy(self.id)
            self.id = -1
