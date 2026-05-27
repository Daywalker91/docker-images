"""
ctypes-Wrapper für die rkllm C-Library (librkllmrt.so).
Basiert auf dem rkllm.h Header von airockchip/rknn-llm v1.2.1.
"""
import ctypes
import threading
import logging
import os

log = logging.getLogger(__name__)

# ── Library laden ──────────────────────────────────────────────────────────────
_lib = ctypes.CDLL(os.environ.get("RKLLM_LIB", "librkllmrt.so"))


# ── Enums ──────────────────────────────────────────────────────────────────────
LLM_RUN_NORMAL  = 0
LLM_RUN_WAITING = 1
LLM_RUN_FINISH  = 2
LLM_RUN_ERROR   = 3

RKLLM_RUN_NORMAL = 0
RKLLM_INPUT_PROMPT = 0


# ── Structs ────────────────────────────────────────────────────────────────────
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("reserved",       ctypes.c_uint8 * 112),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path",         ctypes.c_char_p),
        ("max_context_len",    ctypes.c_int32),
        ("max_new_tokens",     ctypes.c_int32),
        ("top_k",              ctypes.c_int32),
        ("top_p",              ctypes.c_float),
        ("temperature",        ctypes.c_float),
        ("repeat_penalty",     ctypes.c_float),
        ("frequency_penalty",  ctypes.c_float),
        ("mirostat",           ctypes.c_int32),
        ("mirostat_tau",       ctypes.c_float),
        ("mirostat_eta",       ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async",           ctypes.c_bool),
        ("img_start",          ctypes.c_char_p),
        ("img_end",            ctypes.c_char_p),
        ("img_content",        ctypes.c_char_p),
        ("extend_param",       RKLLMExtendParam),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text",     ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("_pad",         ctypes.c_uint8 * 64),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_type", ctypes.c_int),
        ("_data",      _InputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
    ]


# ── Callback-Typ ───────────────────────────────────────────────────────────────
LLMResultCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)

# ── Funktions-Signaturen ───────────────────────────────────────────────────────
_lib.rkllm_createDefaultParam.restype  = RKLLMParam
_lib.rkllm_createDefaultParam.argtypes = []

_lib.rkllm_init.restype  = ctypes.c_int
_lib.rkllm_init.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(RKLLMParam),
    LLMResultCallback,
]

_lib.rkllm_run.restype  = ctypes.c_int
_lib.rkllm_run.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(RKLLMInput),
    ctypes.POINTER(RKLLMInferParam),
    ctypes.c_void_p,
]

_lib.rkllm_destroy.restype  = ctypes.c_int
_lib.rkllm_destroy.argtypes = [ctypes.c_void_p]

_lib.rkllm_abort.restype  = ctypes.c_int
_lib.rkllm_abort.argtypes = [ctypes.c_void_p]


# ── RKLLMModel Klasse ──────────────────────────────────────────────────────────
class RKLLMModel:
    """Thread-sicherer Wrapper um die rkllm C-Library."""

    def __init__(self, model_path: str, max_context_len: int = 4096,
                 max_new_tokens: int = 2048, temperature: float = 0.7):
        self._handle = ctypes.c_void_p()
        self._lock   = threading.Lock()

        # Inferenz-State (pro Aufruf)
        self._output_tokens: list[str] = []
        self._done = threading.Event()
        self._error = False

        # Callback als Attribut halten damit GC ihn nicht löscht
        self._callback = LLMResultCallback(self._on_token)

        # Parameter aufbauen
        param = _lib.rkllm_createDefaultParam()
        param.model_path      = model_path.encode()
        param.max_context_len = max_context_len
        param.max_new_tokens  = max_new_tokens
        param.temperature     = temperature
        param.top_k           = 1
        param.top_p           = 0.9
        param.skip_special_token = True
        param.is_async        = False

        ret = _lib.rkllm_init(ctypes.byref(self._handle), ctypes.byref(param), self._callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init fehlgeschlagen: {ret}")
        log.info("rkllm Modell geladen: %s", model_path)

    def _on_token(self, result_ptr, userdata, state):
        """Wird von der C-Library pro Token aufgerufen."""
        if state == LLM_RUN_ERROR:
            self._error = True
            self._done.set()
            return
        if result_ptr and result_ptr.contents.text:
            token = result_ptr.contents.text.decode("utf-8", errors="replace")
            self._output_tokens.append(token)
        if state == LLM_RUN_FINISH:
            self._done.set()

    def generate(self, prompt: str) -> str:
        """Blockierende Inferenz – gibt den vollständigen Text zurück."""
        with self._lock:
            self._output_tokens = []
            self._done.clear()
            self._error = False

            rkllm_input              = RKLLMInput()
            rkllm_input.input_type   = RKLLM_INPUT_PROMPT
            rkllm_input._data.prompt_input = prompt.encode()

            infer_param = RKLLMInferParam()
            infer_param.mode = RKLLM_RUN_NORMAL

            ret = _lib.rkllm_run(
                self._handle,
                ctypes.byref(rkllm_input),
                ctypes.byref(infer_param),
                None,
            )
            if ret != 0:
                raise RuntimeError(f"rkllm_run fehlgeschlagen: {ret}")

            self._done.wait(timeout=120)

            if self._error:
                raise RuntimeError("rkllm Inferenz-Fehler")

            return "".join(self._output_tokens)

    def destroy(self):
        if self._handle:
            _lib.rkllm_destroy(self._handle)
            self._handle = None
