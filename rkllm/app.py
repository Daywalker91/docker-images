"""
RKLLM Wrapper – OpenAI-kompatibler API-Server für die echte Rockchip RKLLM-Runtime (librkllmrt.so).
Im Gegensatz zu rk-llama.cpp (GGUF, Requantisierung zur Laufzeit) läuft hier ein für die NPU
vorquantisiertes .rkllm Modell direkt über die offizielle RKLLM C-API.

Modelle werden – analog zu rk-llama – dynamisch per "hf:<owner>/<repo>/<filename>.rkllm" von
HuggingFace geladen und im PVC gecacht. Kein Pod-Neustart für einen Modellwechsel nötig.
"""

import json
import logging
import os
import resource
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)

# File-Descriptor-Limit erhöhen, analog zum offiziellen flask_server.py Beispiel
# (resource.setrlimit(RLIMIT_NOFILE, (102400, 102400)) vor rkllm_init()). Testweise
# ergänzt für Issue airockchip/rknn-llm#509 – falls release-v1.3.0 intern mehr
# File-Handles braucht als das Default-Limit erlaubt, könnte das den SIGSEGV erklären.
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    log.info("RLIMIT_NOFILE auf 102400 gesetzt")
except (ValueError, OSError) as e:
    log.warning("RLIMIT_NOFILE konnte nicht gesetzt werden: %s", e)

app = Flask(__name__)

MODELS_DIR        = Path(os.environ.get("MODELS_DIR", "/models"))
MAX_CONTEXT_LEN   = int(os.environ.get("RKLLM_MAX_CONTEXT", "4096"))
MAX_NEW_TOKENS    = int(os.environ.get("RKLLM_MAX_NEW_TOKENS", "2048"))
TEMPERATURE       = float(os.environ.get("RKLLM_TEMPERATURE", "0.7"))
TOP_K             = int(os.environ.get("RKLLM_TOP_K", "1"))
TOP_P             = float(os.environ.get("RKLLM_TOP_P", "0.9"))

# CPU-Affinität – optional, per ENV testbar (siehe rkllm_wrapper.CPU_BIG_CORES_RK3588).
# Leer lassen = RKLLM-Default (alle Kerne) verwenden.
_cpus_mask_env = os.environ.get("RKLLM_ENABLED_CPUS_MASK", "").strip()
_cpus_num_env  = os.environ.get("RKLLM_ENABLED_CPUS_NUM", "").strip()
ENABLED_CPUS_MASK = int(_cpus_mask_env, 0) if _cpus_mask_env else None
ENABLED_CPUS_NUM  = int(_cpus_num_env, 0) if _cpus_num_env else None

# Globaler State – ein Modell ist zur gleichen Zeit geladen (NPU = exklusive Ressource)
_model = None          # rkllm_wrapper.RKLLMModel
_current_model_name = None
_model_lock = threading.Lock()


def _resolve_model_path(model_name: str) -> Path:
    """
    Löst einen Modellnamen zu einem lokalen Pfad auf – analog zu rk-llama/app.py.
    Formate:
      - hf:<owner>/<repo>/<filename>.rkllm  -> HuggingFace Download (gecacht in MODELS_DIR)
      - <filename>.rkllm                    -> direkt aus MODELS_DIR
      - /absoluter/pfad.rkllm                -> direkt
    """
    if model_name.startswith("hf:"):
        parts = model_name[3:].split("/", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Ungültiges HuggingFace Format: {model_name} – erwartet hf:<owner>/<repo>/<filename>"
            )
        owner, repo, filename = parts
        local_path = MODELS_DIR / filename
        if not local_path.exists():
            log.info("Lade RKLLM Modell von HuggingFace: %s/%s/%s", owner, repo, filename)
            t0 = time.monotonic()
            hf_hub_download(
                repo_id=f"{owner}/{repo}",
                filename=filename,
                local_dir=str(MODELS_DIR),
            )
            log.info("Download abgeschlossen in %.1fs: %s", time.monotonic() - t0, local_path)
        return local_path

    if model_name.startswith("/"):
        return Path(model_name)

    local_path = MODELS_DIR / model_name
    if not local_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {local_path}")
    return local_path


def _ensure_model(model_name: str):
    """Stellt sicher, dass das angeforderte Modell geladen ist. Lädt bei Bedarf neu."""
    global _model, _current_model_name

    model_path = _resolve_model_path(model_name)
    model_key = str(model_path)

    with _model_lock:
        if _current_model_name == model_key and _model is not None:
            return  # bereits geladen

        if _model is not None:
            log.info("Entlade vorheriges Modell: %s", _current_model_name)
            try:
                _model.destroy()
            except Exception:
                log.exception("Fehler beim Entladen des alten Modells (ignoriert)")
            _model = None
            _current_model_name = None

        log.info("Lade RKLLM Modell: %s", model_path)
        t0 = time.monotonic()
        from rkllm_wrapper import RKLLMModel  # Import hier, damit Flask auch ohne Lib startet

        _model = RKLLMModel(
            model_path=model_key,
            max_context_len=MAX_CONTEXT_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            enabled_cpus_mask=ENABLED_CPUS_MASK,
            enabled_cpus_num=ENABLED_CPUS_NUM,
        )
        _current_model_name = model_key
        log.info("Modell bereit in %.1fs: %s", time.monotonic() - t0, model_path)


def _messages_to_prompt(messages: list) -> tuple[str, str]:
    """
    Reduziert eine OpenAI-style messages-Liste auf (system_prompt, user_prompt).
    RKLLM nutzt rkllm_set_chat_template() für den System-Prompt und übergibt
    den Rest als reinen Prompt-Text – kein eigenes Multi-Turn-Templating hier,
    das letzte 'user'-Message wird als aktueller Prompt verwendet.
    """
    system_prompt = ""
    user_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_prompt = content
        else:
            user_parts.append(content)
    return system_prompt, "\n".join(user_parts)


def _sse_chunk(content: str = "", finish_reason: str | None = None, model: str = "") -> str:
    """Baut einen OpenAI-kompatiblen SSE-Chunk im chat.completion.chunk Format."""
    payload = {
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": _current_model_name,
        "model_loaded": _model is not None,
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Listet verfügbare .rkllm Dateien im Models-Verzeichnis."""
    models = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.rkllm"):
            models.append({"id": f.name, "object": "model", "owned_by": "local"})
    return jsonify({"object": "list", "data": models})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Kein JSON Body"}), 400

    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Kein Modell angegeben"}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "Keine messages angegeben"}), 400

    stream = bool(data.get("stream", False))

    try:
        _ensure_model(model_name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error("Modell konnte nicht geladen werden: %s", e)
        return jsonify({"error": str(e)}), 400

    system_prompt, user_prompt = _messages_to_prompt(messages)
    if system_prompt:
        _model.set_chat_template(system_prompt)

    log.info(
        "Chat-Completion: model=%s stream=%s prompt_len=%d",
        model_name, stream, len(user_prompt),
    )

    if stream:
        def generate():
            t0 = time.monotonic()
            n_tokens = 0
            try:
                for token in _model.generate_stream(user_prompt):
                    n_tokens += 1
                    yield _sse_chunk(content=token, model=model_name)
                yield _sse_chunk(finish_reason="stop", model=model_name)
                yield "data: [DONE]\n\n"
                log.info(
                    "Stream beendet: %d Chunks gesendet in %.1fs",
                    n_tokens, time.monotonic() - t0,
                )
            except Exception as e:
                log.exception("Fehler während des Streams")
                err_payload = json.dumps({"error": str(e)})
                yield f"data: {err_payload}\n\n"
                yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Nicht-Streaming: kompletten Text sammeln und als Standard-Response zurückgeben
    try:
        t0 = time.monotonic()
        full_text = _model.generate(user_prompt)
        perf = _model.last_perf
        log.info(
            "Completion fertig in %.1fs (generate=%.0fms/%dtok)",
            time.monotonic() - t0, perf.get("generate_time_ms", 0), perf.get("generate_tokens", 0),
        )
    except Exception as e:
        log.exception("Fehler bei der Inferenz")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": _model.last_perf.get("prefill_tokens", 0),
            "completion_tokens": _model.last_perf.get("generate_tokens", 0),
        },
    })


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
