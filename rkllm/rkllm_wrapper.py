"""
ctypes-Wrapper für die rkllm C-Library (librkllmrt.so).
Struct-Layout basiert auf dem aktuellen rkllm.h Header von airockchip/rknn-llm (main, Stand 2026).
Quelle: https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h

Wichtig: Die Structs MÜSSEN exakt mit dem C-Header übereinstimmen (Feldreihenfolge + Typen).
Ein falsches Layout führt nicht zu einem Fehler, sondern zu stillschweigend falschen Werten
(z.B. landet temperature im falschen Feld) oder Memory Corruption.
"""
import ctypes
import logging
import os
import queue
import threading
import time

log = logging.getLogger(__name__)

# ── Library laden ──────────────────────────────────────────────────────────────
_lib = ctypes.CDLL(os.environ.get("RKLLM_LIB", "librkllmrt.so"))


# ── Enums (LLMCallState) ──────────────────────────────────────────────────────
RKLLM_RUN_NORMAL  = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH  = 2
RKLLM_RUN_ERROR   = 3

# ── Enums (RKLLMInputType) ────────────────────────────────────────────────────
RKLLM_INPUT_PROMPT     = 0
RKLLM_INPUT_TOKEN      = 1
RKLLM_INPUT_EMBED      = 2
RKLLM_INPUT_MULTIMODAL = 3

# ── Enums (RKLLMInferMode) ────────────────────────────────────────────────────
RKLLM_INFER_GENERATE              = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS            = 2

# ── CPU Bitmasks (für enabled_cpus_mask) ──────────────────────────────────────
CPU0 = 1 << 0
CPU1 = 1 << 1
CPU2 = 1 << 2
CPU3 = 1 << 3
CPU4 = 1 << 4
CPU5 = 1 << 5
CPU6 = 1 << 6
CPU7 = 1 << 7
CPU_ALL = 0xFF

# RK3588: Cortex-A76 (schnell) = CPU4-7, Cortex-A55 (langsam) = CPU0-3.
# Auf vielen Boards sind aber die Indizes vertauscht (A55 zuerst) – per ENV testbar.
CPU_BIG_CORES_RK3588 = CPU4 | CPU5 | CPU6 | CPU7


# ── Structs ────────────────────────────────────────────────────────────────────
class RKLLMExtendParam(ctypes.Structure):
    """
    typedef struct {
        int32_t base_domain_id;
        int8_t  embed_flash;
        int8_t  enabled_cpus_num;
        uint32_t enabled_cpus_mask;
        uint8_t n_batch;
        int8_t  use_cross_attn;
        uint8_t reserved[104];
    } RKLLMExtendParam;
    """
    _fields_ = [
        ("base_domain_id",    ctypes.c_int32),
        ("embed_flash",       ctypes.c_int8),
        ("enabled_cpus_num",  ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch",           ctypes.c_uint8),
        ("use_cross_attn",    ctypes.c_int8),
        ("reserved",          ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    """
    Layout 1:1 aus dem offiziellen flask_server.py Beispiel (main branch, Stand 2026),
    bestätigt durch airockchip-Maintainer als Referenz für release-v1.3.0+.
    Wichtiger Unterschied zur vorherigen Version: 'ignore_eos_token' zusätzlich vorhanden
    (zwischen skip_special_token und is_async), und die img_start/img_end/img_content
    Felder existieren NICHT mehr in RKLLMParam (wurden nach RKLLMImageInput verschoben).
    Ein falsches Layout hier verschiebt extend_param (inkl. embed_flash/enabled_cpus_mask)
    um mehrere Bytes und führt zu Memory Corruption / SIGSEGV in rkllm_run().
    """
    _fields_ = [
        ("model_path",         ctypes.c_char_p),
        ("max_context_len",    ctypes.c_int32),
        ("max_new_tokens",     ctypes.c_int32),
        ("top_k",              ctypes.c_int32),
        ("n_keep",             ctypes.c_int32),
        ("top_p",              ctypes.c_float),
        ("temperature",        ctypes.c_float),
        ("repeat_penalty",     ctypes.c_float),
        ("frequency_penalty",  ctypes.c_float),
        ("presence_penalty",   ctypes.c_float),
        ("mirostat",           ctypes.c_int32),
        ("mirostat_tau",       ctypes.c_float),
        ("mirostat_eta",       ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token",   ctypes.c_bool),
        ("is_async",           ctypes.c_bool),
        ("extend_param",       RKLLMExtendParam),
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale",             ctypes.c_float),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed",    ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens",  ctypes.c_size_t),
    ]


class RKLLMImageInput(ctypes.Structure):
    _fields_ = [
        ("image_embed",    ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image",        ctypes.c_size_t),
        ("image_start",    ctypes.c_char_p),
        ("image_end",      ctypes.c_char_p),
        ("image_content",  ctypes.c_char_p),
        ("image_width",    ctypes.c_size_t),
        ("image_height",   ctypes.c_size_t),
    ]


class RKLLMVideoInput(ctypes.Structure):
    _fields_ = [
        ("video_embed",       ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens",    ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video",           ctypes.c_size_t),
        ("video_start",       ctypes.c_char_p),
        ("video_end",         ctypes.c_char_p),
        ("video_content",     ctypes.c_char_p),
        ("frame_width",       ctypes.c_size_t),
        ("frame_height",      ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image",  RKLLMImageInput),
        ("video",  RKLLMVideoInput),
    ]


class _RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input",    ctypes.c_char_p),
        ("embed_input",     RKLLMEmbedInput),
        ("token_input",     RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    """
    typedef struct {
        const char* role;
        bool enable_thinking;
        RKLLMInputType input_type;
        union { ... };
    } RKLLMInput;
    """
    _anonymous_ = ("u",)
    _fields_ = [
        ("role",            ctypes.c_char_p),
        ("enable_thinking",  ctypes.c_bool),
        ("input_type",      ctypes.c_int),
        ("u",               _RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p),
    ]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache",  ctypes.c_int),
        ("prompt_cache_path",  ctypes.c_char_p),
    ]


class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k",             ctypes.c_int32),
        ("top_p",             ctypes.c_float),
        ("temperature",       ctypes.c_float),
        ("repeat_penalty",    ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty",  ctypes.c_float),
        ("mirostat",          ctypes.c_int32),
        ("mirostat_tau",      ctypes.c_float),
        ("mirostat_eta",      ctypes.c_float),
    ]


class RKLLMInferParam(ctypes.Structure):
    """
    Layout 1:1 aus dem offiziellen flask_server.py Beispiel. Zusätzlich zur
    vorherigen Version: sampling_params (Pointer, pro-Request überschreibbar)
    und max_new_tokens (überschreibt den Init-Wert für einen einzelnen Run).
    """
    _fields_ = [
        ("mode",                ctypes.c_int),
        ("lora_params",         ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("sampling_params",     ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history",        ctypes.c_int),
        ("max_new_tokens",      ctypes.c_int32),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size",     ctypes.c_int),
        ("num_tokens",    ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits",     ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    """Performance-Statistik – wird pro Result-Callback mitgeliefert (state == FINISH)."""
    _fields_ = [
        ("prefill_time_ms",  ctypes.c_float),
        ("prefill_tokens",   ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens",  ctypes.c_int),
        ("memory_usage_mb",  ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    """
    typedef struct {
        const char* text;
        int32_t token_id;
        RKLLMResultLastHiddenLayer last_hidden_layer;
        RKLLMResultLogits logits;
        RKLLMPerfStat perf;
    } RKLLMResult;
    """
    _fields_ = [
        ("text",              ctypes.c_char_p),
        ("token_id",          ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits",            RKLLMResultLogits),
        ("perf",               RKLLMPerfStat),
    ]


# ── Callback-Typ ───────────────────────────────────────────────────────────────
LLMResultCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,                       # Rückgabewert: 0 = weiter, 1 = pausieren
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

_lib.rkllm_run_async.restype  = ctypes.c_int
_lib.rkllm_run_async.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(RKLLMInput),
    ctypes.POINTER(RKLLMInferParam),
    ctypes.c_void_p,
]

_lib.rkllm_destroy.restype  = ctypes.c_int
_lib.rkllm_destroy.argtypes = [ctypes.c_void_p]

_lib.rkllm_abort.restype  = ctypes.c_int
_lib.rkllm_abort.argtypes = [ctypes.c_void_p]

_lib.rkllm_is_running.restype  = ctypes.c_int
_lib.rkllm_is_running.argtypes = [ctypes.c_void_p]

_lib.rkllm_set_chat_template.restype  = ctypes.c_int
_lib.rkllm_set_chat_template.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
]


# ── Sentinel für Queue-Ende ───────────────────────────────────────────────────
_STREAM_DONE = object()
_STREAM_ERROR = object()


class RKLLMModel:
    """
    Thread-sicherer Wrapper um die rkllm C-Library mit Token-Streaming.

    Die NPU kann nur eine Inferenz gleichzeitig ausführen – paralleles Aufrufen
    von generate_stream() aus mehreren Threads wird über self._lock serialisiert.
    """

    def __init__(self, model_path: str, max_context_len: int = 4096,
                 max_new_tokens: int = 2048, temperature: float = 0.7,
                 top_k: int = 1, top_p: float = 0.9,
                 enabled_cpus_mask: int | None = None,
                 enabled_cpus_num: int | None = None,
                 system_prompt: str | None = None):
        self._handle = ctypes.c_void_p()
        self._lock = threading.Lock()

        # Pro-Aufruf State – wird in generate_stream() neu belegt
        self._queue: "queue.Queue" = queue.Queue()
        self._last_perf: dict = {}

        # Callback als Attribut halten, sonst sammelt der GC ihn ein
        # während die C-Lib noch Zeiger darauf hält -> Segfault.
        self._callback = LLMResultCallback(self._on_result)

        param = _lib.rkllm_createDefaultParam()
        param.model_path         = model_path.encode("utf-8")
        param.max_context_len    = max_context_len
        param.max_new_tokens     = max_new_tokens
        param.temperature        = temperature
        param.top_k              = top_k
        param.top_p              = top_p
        param.skip_special_token = True
        param.ignore_eos_token   = False
        param.is_async           = False

        if enabled_cpus_mask is not None:
            param.extend_param.enabled_cpus_mask = enabled_cpus_mask
        if enabled_cpus_num is not None:
            param.extend_param.enabled_cpus_num = enabled_cpus_num
        # Defensiv wie im offiziellen Demo explizit gesetzt, statt sich auf den
        # rkllm_createDefaultParam()-Default zu verlassen.
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1

        ret = _lib.rkllm_init(ctypes.byref(self._handle), ctypes.byref(param), self._callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init fehlgeschlagen (Code {ret}) für Modell: {model_path}")
        log.info(
            "rkllm Modell geladen: %s (ctx=%d, max_new_tokens=%d, cpus_mask=%s, cpus_num=%s)",
            model_path, max_context_len, max_new_tokens,
            hex(enabled_cpus_mask) if enabled_cpus_mask is not None else "default",
            enabled_cpus_num if enabled_cpus_num is not None else "default",
        )

        if system_prompt:
            self.set_chat_template(system_prompt)

    def set_chat_template(self, system_prompt: str, prefix: str = "", postfix: str = ""):
        ret = _lib.rkllm_set_chat_template(
            self._handle,
            system_prompt.encode("utf-8"),
            prefix.encode("utf-8"),
            postfix.encode("utf-8"),
        )
        if ret != 0:
            log.warning("rkllm_set_chat_template fehlgeschlagen (Code %d)", ret)

    # ── Callback aus der C-Lib (läuft im Inferenz-Thread der Library!) ────────
    def _on_result(self, result_ptr, userdata, state) -> int:
        try:
            if state == RKLLM_RUN_ERROR:
                self._queue.put(_STREAM_ERROR)
                return 0

            if result_ptr:
                res = result_ptr.contents
                if res.text:
                    token = res.text.decode("utf-8", errors="replace")
                    self._queue.put(token)

                if state == RKLLM_RUN_FINISH:
                    perf = res.perf
                    self._last_perf = {
                        "prefill_time_ms":  perf.prefill_time_ms,
                        "prefill_tokens":   perf.prefill_tokens,
                        "generate_time_ms": perf.generate_time_ms,
                        "generate_tokens":  perf.generate_tokens,
                        "memory_usage_mb":  perf.memory_usage_mb,
                    }
                    self._queue.put(_STREAM_DONE)
            return 0
        except Exception:
            log.exception("Fehler im rkllm result callback")
            self._queue.put(_STREAM_ERROR)
            return 0

    def generate_stream(self, prompt: str, role: str | None = None,
                         temperature: float | None = None, top_k: int | None = None,
                         top_p: float | None = None, max_new_tokens: int | None = None):
        """
        Generator – yieldet Text-Chunks sobald sie von der NPU kommen.
        Am Ende steht self.last_perf mit den Performance-Stats des letzten Laufs zur Verfügung.

        temperature/top_k/top_p/max_new_tokens überschreiben pro Aufruf die Init-Werte
        (analog zum offiziellen flask_server.py Beispiel), ohne self._handle neu zu
        initialisieren. Werden sie nicht angegeben, gelten die beim Laden gesetzten Defaults.
        """
        with self._lock:
            # Queue für diesen Aufruf leeren (Sicherheitsnetz falls noch alte Reste drin sind)
            while not self._queue.empty():
                self._queue.get_nowait()
            self._last_perf = {}

            # Komplett nullen statt nur einzelne Felder zu setzen – analog zum offiziellen
            # Demo-Code (memset(&rkllm_input, 0, sizeof(RKLLMInput))), um sicherzustellen
            # dass kein uninitialisierter Stack-/Heap-Speicher in der Struct landet.
            rkllm_input = RKLLMInput()
            ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
            rkllm_input.role = role.encode("utf-8") if role else None
            rkllm_input.enable_thinking = False
            rkllm_input.input_type = RKLLM_INPUT_PROMPT
            rkllm_input.prompt_input = prompt.encode("utf-8")

            infer_param = RKLLMInferParam()
            ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
            infer_param.mode = RKLLM_INFER_GENERATE
            infer_param.lora_params = None
            infer_param.prompt_cache_params = None
            infer_param.keep_history = 0

            # Pro-Request-Sampling-Parameter optional setzen (Pointer muss am Leben
            # bleiben, solange rkllm_run() läuft -> lokale Variable im selben Scope).
            sampling_param = None
            if temperature is not None or top_k is not None or top_p is not None:
                sampling_param = RKLLMSamplingParam()
                ctypes.memset(ctypes.byref(sampling_param), 0, ctypes.sizeof(RKLLMSamplingParam))
                sampling_param.top_k = top_k if top_k is not None else 1
                sampling_param.top_p = top_p if top_p is not None else 0.9
                sampling_param.temperature = temperature if temperature is not None else 0.7
                sampling_param.repeat_penalty = 1.1
                sampling_param.frequency_penalty = 0.0
                sampling_param.presence_penalty = 0.0
                sampling_param.mirostat = 0
                sampling_param.mirostat_tau = 5.0
                sampling_param.mirostat_eta = 0.1
                infer_param.sampling_params = ctypes.pointer(sampling_param)

            if max_new_tokens is not None:
                infer_param.max_new_tokens = max_new_tokens

            t_start = time.monotonic()
            ret = _lib.rkllm_run(
                self._handle,
                ctypes.byref(rkllm_input),
                ctypes.byref(infer_param),
                None,
            )
            if ret != 0:
                raise RuntimeError(f"rkllm_run fehlgeschlagen (Code {ret})")

            while True:
                item = self._queue.get()
                if item is _STREAM_DONE:
                    perf = self._last_perf
                    wall_s = time.monotonic() - t_start
                    log.info(
                        "rkllm fertig: prefill=%.0fms/%dtok generate=%.0fms/%dtok "
                        "(%.2f tok/s) wall=%.1fs mem=%.0fMiB",
                        perf.get("prefill_time_ms", 0), perf.get("prefill_tokens", 0),
                        perf.get("generate_time_ms", 0), perf.get("generate_tokens", 0),
                        (perf.get("generate_tokens", 0) / (perf.get("generate_time_ms", 1) / 1000))
                            if perf.get("generate_time_ms") else 0.0,
                        wall_s,
                        perf.get("memory_usage_mb", 0),
                    )
                    return
                if item is _STREAM_ERROR:
                    raise RuntimeError("rkllm Inferenz-Fehler (Callback meldete RKLLM_RUN_ERROR)")
                yield item

    def generate(self, prompt: str, role: str | None = None,
                 temperature: float | None = None, top_k: int | None = None,
                 top_p: float | None = None, max_new_tokens: int | None = None) -> str:
        """Blockierende Inferenz – sammelt den Stream und gibt den vollständigen Text zurück."""
        return "".join(self.generate_stream(
            prompt, role=role, temperature=temperature,
            top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens,
        ))

    @property
    def last_perf(self) -> dict:
        return dict(self._last_perf)

    def abort(self):
        if self._handle:
            _lib.rkllm_abort(self._handle)

    def destroy(self):
        if self._handle:
            _lib.rkllm_destroy(self._handle)
            self._handle = None
