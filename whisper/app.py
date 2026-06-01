from flask import Flask, request, jsonify
import subprocess
import os
import uuid
import logging
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Konfiguration via Env-Variablen (Defaults)
WHISPER_MODEL     = os.environ.get("WHISPER_MODEL", "base")
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models")
WHISPER_BINARY    = os.environ.get("WHISPER_BINARY", "/usr/local/bin/whisper-cli")
WHISPER_BACKEND   = os.environ.get("WHISPER_BACKEND", "cpu")   # cpu | rknn
WHISPER_THREADS   = os.environ.get("WHISPER_THREADS", "4")
RKNN_MODEL_DIR    = os.environ.get("RKNN_MODEL_DIR", "/models/rknn")

# Jobs-Store
jobs = {}
jobs_lock = threading.Lock()

# RKNN Modell (einmalig geladen)
rknn_model = None


def get_cpu_model_path(model: str) -> str:
    return os.path.join(WHISPER_MODEL_DIR, f"ggml-{model}.bin")


def get_rknn_paths(model: str) -> tuple:
    encoder = os.path.join(RKNN_MODEL_DIR, f"whisper-{model}-encoder.rknn")
    decoder = os.path.join(RKNN_MODEL_DIR, f"whisper-{model}-decoder.rknn")
    return encoder, decoder


def ensure_cpu_model(model: str):
    """CPU-Modell herunterladen falls nicht vorhanden."""
    model_path = get_cpu_model_path(model)
    if not os.path.exists(model_path):
        log.info(f"Modell nicht gefunden, starte Download: {model_path}")
        subprocess.run(["/usr/local/bin/download_model.sh"], env={
            **os.environ,
            "WHISPER_MODEL": model,
            "WHISPER_MODEL_DIR": WHISPER_MODEL_DIR,
        }, check=True)


def ensure_rknn_models(model: str):
    """RKNN-Modelle herunterladen falls nicht vorhanden."""
    encoder, decoder = get_rknn_paths(model)
    if not os.path.exists(encoder) or not os.path.exists(decoder):
        log.info(f"RKNN-Modelle nicht gefunden, starte Download...")
        subprocess.run(["/usr/local/bin/download_rknn_models.sh"], env={
            **os.environ,
            "WHISPER_MODEL": model,
            "RKNN_MODEL_DIR": RKNN_MODEL_DIR,
        }, check=True)


def load_rknn_model(model: str):
    """RKNN-Modell laden (einmalig)."""
    global rknn_model
    if rknn_model is None:
        from rknn_inference import WhisperRKNN
        encoder, decoder = get_rknn_paths(model)
        rknn_model = WhisperRKNN(encoder, decoder, model_size=model)
    return rknn_model


def worker_transcribe(job_id: str, audio_path: str, backend: str, model: str):
    """Hintergrund-Worker für Transkription."""
    with jobs_lock:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10

    try:
        if backend == "rknn":
            # RKNN-Backend (NPU)
            ensure_rknn_models(model)
            with jobs_lock:
                jobs[job_id]["progress"] = 30
            rknn = load_rknn_model(model)
            with jobs_lock:
                jobs[job_id]["progress"] = 50
            transcript = rknn.transcribe(audio_path)
        else:
            # CPU-Backend (whisper.cpp)
            ensure_cpu_model(model)
            with jobs_lock:
                jobs[job_id]["progress"] = 30
            result = subprocess.run([
                WHISPER_BINARY,
                "-m", get_cpu_model_path(model),
                "-f", audio_path,
                "-t", WHISPER_THREADS,
                "--output-txt",
                "--no-timestamps",
                "-l", "auto",
            ], capture_output=True, text=True, check=True)
            transcript = result.stdout.strip()

        with jobs_lock:
            jobs[job_id]["status"]     = "done"
            jobs[job_id]["progress"]   = 100
            jobs[job_id]["transcript"] = transcript

        log.info(f"Job {job_id} abgeschlossen ({backend})")

    except Exception as e:
        log.error(f"Job {job_id} fehlgeschlagen: {e}")
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = str(e)


# ─── POST /transcribe ─────────────────────────────────────────────────────────
# Parameter:
#   audio_path  (required) – Pfad zur WAV-Datei auf dem Shared Volume
#   backend     (optional) – "cpu" oder "rknn" – überschreibt Env-Variable
#   model       (optional) – "tiny|base|small|medium" – überschreibt Env-Variable
@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(silent=True) or {}

    audio_path = data.get("audio_path")
    if not audio_path:
        return jsonify({"error": "audio_path fehlt"}), 400
    if not os.path.exists(audio_path):
        return jsonify({"error": f"Datei nicht gefunden: {audio_path}"}), 400

    # Request-Parameter haben Vorrang vor Env-Variable
    backend = data.get("backend", WHISPER_BACKEND)
    model   = data.get("model",   WHISPER_MODEL)

    if backend not in ("cpu", "rknn"):
        return jsonify({"error": f"Ungültiges Backend: {backend}"}), 400
    if model not in ("tiny", "base", "small", "medium"):
        return jsonify({"error": f"Ungültiges Modell: {model}"}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status":   "queued",
            "progress": 0,
            "backend":  backend,
            "model":    model,
        }

    thread = threading.Thread(
        target=worker_transcribe,
        args=(job_id, audio_path, backend, model),
        daemon=True,
    )
    thread.start()

    log.info(f"Job {job_id} gestartet – backend={backend}, model={model}")
    return jsonify({"job_id": job_id, "status": "queued"})


# ─── GET /job/<job_id> ────────────────────────────────────────────────────────
# Gibt zurück: { "status": "queued|processing|done|error",
#                "progress": 0-100,
#                "transcript": "...",  ← nur wenn done
#                "backend": "cpu|rknn",
#                "error": "..." }      ← nur wenn error
@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job nicht gefunden"}), 404
    return jsonify(job)


# ─── GET /health ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    cpu_ok  = os.path.exists(get_cpu_model_path(WHISPER_MODEL))
    rknn_ok = rknn_model is not None

    return jsonify({
        "status":        "ok",
        "default_backend": WHISPER_BACKEND,
        "default_model":   WHISPER_MODEL,
        "cpu_model_ready": cpu_ok,
        "rknn_model_ready": rknn_ok,
    })


if __name__ == "__main__":
    # Beim Start: Default-Modell vorladen
    if WHISPER_BACKEND == "cpu":
        ensure_cpu_model(WHISPER_MODEL)
    elif WHISPER_BACKEND == "rknn":
        ensure_rknn_models(WHISPER_MODEL)
        load_rknn_model(WHISPER_MODEL)

    app.run(host="0.0.0.0", port=5001)
