"""
rk-llama.cpp Wrapper
OpenAI-kompatibler API-Server mit dynamischem Modell-Loading.
Modelle werden von HuggingFace als GGUF heruntergeladen und gecacht.
llama-server wird per Request-Modell gestartet/neugestartet.
"""

import os
import re
import subprocess
import threading
import time
import logging
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from pathlib import Path
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "/usr/local/bin/llama-server")
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", "4096"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))

# Globaler State
_server_process = None
_current_model = None
_server_lock = threading.Lock()
_server_ready = False


def _wait_for_server(timeout=300):
    """Wartet bis llama-server bereit ist."""
    global _server_ready
    url = f"http://localhost:{LLAMA_SERVER_PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                _server_ready = True
                log.info("llama-server ist bereit")
                return True
        except Exception:
            pass
        time.sleep(1)
    log.error("llama-server Timeout beim Starten")
    return False


def _stop_server():
    """Stoppt den laufenden llama-server."""
    global _server_process, _server_ready, _current_model
    if _server_process and _server_process.poll() is None:
        log.info("Stoppe llama-server...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None
    _server_ready = False
    _current_model = None


def _start_server(model_path: str):
    """Startet llama-server mit dem angegebenen Modell."""
    global _server_process, _current_model, _server_ready

    _stop_server()

    log.info(f"Starte llama-server mit Modell: {model_path}")
    cmd = [
        LLAMA_SERVER_BIN,
        "--model", model_path,
        "--ctx-size", str(CONTEXT_SIZE),
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_SERVER_PORT),
        "--parallel", "1",
       # "--no-mmap",           # Wichtig für NPU – kein Memory-Mapping
        "--no-warmup",         # Kein Warmup-Pass
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
    ]

    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _current_model = model_path

    # Log-Output in separatem Thread
    def _log_output():
        for line in _server_process.stdout:
            log.info(f"[llama-server] {line.decode().rstrip()}")
    threading.Thread(target=_log_output, daemon=True).start()

    return _wait_for_server()


def _resolve_model_path(model_name: str) -> str:
    """
    Löst einen Modellnamen zu einem lokalen Pfad auf.
    Formate:
      - hf:<owner>/<repo>/<filename>   → HuggingFace Download
      - filename.gguf                  → direkt aus MODELS_DIR
      - /absoluter/pfad.gguf           → direkt
    """
    if model_name.startswith("hf:"):
        # hf:Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q8_0.gguf
        parts = model_name[3:].split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"Ungültiges HuggingFace Format: {model_name} – erwartet hf:<owner>/<repo>/<filename>")
        owner, repo, filename = parts
        local_path = MODELS_DIR / filename
        if not local_path.exists():
            log.info(f"Lade Modell von HuggingFace: {owner}/{repo}/{filename}")
            hf_hub_download(
                repo_id=f"{owner}/{repo}",
                filename=filename,
                local_dir=str(MODELS_DIR),
            )
        return str(local_path)

    elif model_name.startswith("/"):
        return model_name

    else:
        # Dateiname direkt
        local_path = MODELS_DIR / model_name
        if not local_path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {local_path}")
        return str(local_path)


def _ensure_model(model_name: str):
    """Stellt sicher dass der Server mit dem richtigen Modell läuft."""
    global _current_model, _server_ready

    model_path = _resolve_model_path(model_name)

    with _server_lock:
        if _current_model == model_path and _server_ready:
            return  # Bereits geladen
        if not _start_server(model_path):
            raise RuntimeError(f"llama-server konnte nicht gestartet werden für: {model_path}")


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": _current_model, "server_ready": _server_ready})


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Listet verfügbare GGUF-Modelle im Models-Verzeichnis."""
    models = []
    for f in MODELS_DIR.glob("*.gguf"):
        models.append({
            "id": f.name,
            "object": "model",
            "owned_by": "local",
        })
    return jsonify({"object": "list", "data": models})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-kompatibler Chat-Completions Endpoint."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Kein JSON Body"}), 400

    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Kein Modell angegeben"}), 400

    try:
        _ensure_model(model_name)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

    # Request an llama-server weiterleiten
    try:
        stream = data.get("stream", False)
        resp = requests.post(
            f"http://localhost:{LLAMA_SERVER_PORT}/v1/chat/completions",
            json=data,
            stream=stream,
            timeout=None,
        )

        if stream:
            def generate():
                for chunk in resp.iter_content(chunk_size=None):
                    yield chunk
            return Response(
                stream_with_context(generate()),
                content_type=resp.headers.get("Content-Type", "text/event-stream"),
            )
        else:
            return jsonify(resp.json()), resp.status_code

    except Exception as e:
        log.error(f"Fehler beim Weiterleiten an llama-server: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-kompatibler Completions Endpoint."""
    data = request.get_json()
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Kein Modell angegeben"}), 400

    try:
        _ensure_model(model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        resp = requests.post(
            f"http://localhost:{LLAMA_SERVER_PORT}/v1/completions",
            json=data,
            timeout=None,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5002, debug=False)
