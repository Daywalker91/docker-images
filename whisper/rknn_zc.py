"""
Schlanker ctypes-Zugriff auf die RKNN C-API (librknnrt.so) mit Zero-Copy-Ein-/Ausgaben.

rknnlite kopiert und konvertiert bei jedem inference() alle Eingaben. Beim
Whisper-Decoder sind das pro Token ~48 MB, obwohl sich nur eine Zeile des
Self-KV-Caches ändert. Hier liegen alle Ein-/Ausgaben dauerhaft in
NPU-Speicher (rknn_create_mem + rknn_set_io_mem) im nativen Format; Python
schreibt/liest direkt per numpy-View darauf.

Struct-Layouts aus rknn_api.h (rknn-toolkit2 v2.1.0).
"""
import ctypes
import numpy as np

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_NATIVE_INPUT_ATTR = 8
RKNN_QUERY_NATIVE_OUTPUT_ATTR = 9

RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

NPU_CORE_0 = 1
NPU_CORE_0_1_2 = 7

# rknn_tensor_type -> numpy
_DTYPES = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8, 4: np.int16, 5: np.uint16,
           6: np.int32, 7: np.uint32, 8: np.int64, 9: np.bool_}
_FMT = {0: "NCHW", 1: "NHWC", 2: "NC1HWC2", 3: "UNDEFINED"}


class TensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]

    def describe(self) -> str:
        dims = list(self.dims[: self.n_dims])
        return (f"#{self.index} {self.name.decode()} dims={dims} {_FMT.get(self.fmt, self.fmt)} "
                f"{np.dtype(_DTYPES.get(self.type, np.uint8)).name} size={self.size} stride={self.size_with_stride}")


class TensorMem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class InOutNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


_lib = None


def _api():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL("librknnrt.so")
        ctx_t = ctypes.c_uint64
        lib.rknn_init.argtypes = [ctypes.POINTER(ctx_t), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        lib.rknn_destroy.argtypes = [ctx_t]
        lib.rknn_query.argtypes = [ctx_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        lib.rknn_set_core_mask.argtypes = [ctx_t, ctypes.c_int]
        lib.rknn_run.argtypes = [ctx_t, ctypes.c_void_p]
        lib.rknn_create_mem.argtypes = [ctx_t, ctypes.c_uint32]
        lib.rknn_create_mem.restype = ctypes.POINTER(TensorMem)
        lib.rknn_destroy_mem.argtypes = [ctx_t, ctypes.POINTER(TensorMem)]
        lib.rknn_set_io_mem.argtypes = [ctx_t, ctypes.POINTER(TensorMem), ctypes.POINTER(TensorAttr)]
        lib.rknn_mem_sync.argtypes = [ctx_t, ctypes.POINTER(TensorMem), ctypes.c_int]
        _lib = lib
    return _lib


class RknnModel:
    """Ein RKNN-Modell mit festen Zero-Copy-Puffern für alle Ein- und Ausgaben."""

    def __init__(self, path: str, core_mask: int = NPU_CORE_0):
        lib = _api()
        self.ctx = ctypes.c_uint64()
        data = open(path, "rb").read()
        self._model_buf = ctypes.create_string_buffer(data, len(data))
        ret = lib.rknn_init(ctypes.byref(self.ctx), self._model_buf, len(data), 0, None)
        if ret != 0:
            raise RuntimeError(f"rknn_init fehlgeschlagen ({ret}): {path}")
        del self._model_buf  # Runtime hat das Modell übernommen
        lib.rknn_set_core_mask(self.ctx, core_mask)

        num = InOutNum()
        lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(num), ctypes.sizeof(num))
        self.inputs, self.outputs = [], []        # numpy-Views auf NPU-Speicher
        self.in_attrs, self.out_attrs = [], []
        self._mems = []
        for kind, n, query, views, attrs in (
            ("in", num.n_input, RKNN_QUERY_NATIVE_INPUT_ATTR, self.inputs, self.in_attrs),
            # Ausgaben im normalen Layout: native Ausgaben sind teils gepolstert
            # (z.B. 1x1x512 fp16 -> 8192 statt 1024 Bytes)
            ("out", num.n_output, RKNN_QUERY_OUTPUT_ATTR, self.outputs, self.out_attrs),
        ):
            for i in range(n):
                attr = TensorAttr(index=i)
                lib.rknn_query(self.ctx, query, ctypes.byref(attr), ctypes.sizeof(attr))
                if kind == "in":
                    attr.pass_through = 1     # Daten liegen schon im nativen Format
                size = max(attr.size, attr.size_with_stride)
                mem = lib.rknn_create_mem(self.ctx, size)
                if not mem:
                    raise RuntimeError(f"rknn_create_mem fehlgeschlagen ({kind} #{i}, {size} Bytes)")
                ret = lib.rknn_set_io_mem(self.ctx, mem, ctypes.byref(attr))
                if ret != 0:
                    raise RuntimeError(f"rknn_set_io_mem fehlgeschlagen ({ret}) für {attr.describe()}")
                self._mems.append(mem)
                dtype = np.dtype(_DTYPES[attr.type])
                raw = (ctypes.c_uint8 * size).from_address(mem.contents.virt_addr)
                # size_with_stride kann größer als size sein (Pufferreserve der Runtime);
                # die Werte selbst liegen trotzdem lückenlos hintereinander
                view = np.frombuffer(raw, dtype=dtype, count=attr.n_elems)
                views.append(view.reshape(list(attr.dims[: attr.n_dims])))
                attrs.append(attr)
        self._in_mems = self._mems[: num.n_input]
        self._out_mems = self._mems[num.n_input:]

    def run(self, dirty_inputs=None):
        """dirty_inputs: Indizes der seit dem letzten Lauf beschriebenen Eingaben (None = alle)."""
        lib = _api()
        for i in (range(len(self._in_mems)) if dirty_inputs is None else dirty_inputs):
            lib.rknn_mem_sync(self.ctx, self._in_mems[i], RKNN_MEMORY_SYNC_TO_DEVICE)
        ret = lib.rknn_run(self.ctx, None)
        if ret != 0:
            raise RuntimeError(f"rknn_run fehlgeschlagen ({ret})")
        for m in self._out_mems:
            lib.rknn_mem_sync(self.ctx, m, RKNN_MEMORY_SYNC_FROM_DEVICE)

    def release(self):
        lib = _api()
        for m in self._mems:
            lib.rknn_destroy_mem(self.ctx, m)
        self._mems = []
        if self.ctx.value:
            lib.rknn_destroy(self.ctx)
            self.ctx = ctypes.c_uint64()
