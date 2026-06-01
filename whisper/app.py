import os
import threading
import subprocess
import uuid
import logging

from flask import Flask, request, jsonify

app = Flask(__name__)
log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="INFO:%(name)s:%(message)s")

# ── Konfiguration via Env ──────────────────────────────────────────────────────
WHISPER_BACKEND   = os.environ.get("WHISPER_BACKEND",   "cpu")
WHISPER_MODEL     = os.environ.get("WHISPER_MODEL",     "base")
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models")
RKNN_MODEL_DIR    = os.environ.get("RKNN_MODEL_DIR",    "/models/rknn")
WHISPER_BINARY    = os.environ.get("WHISPER_BINARY",    "/usr/local/bin/whisper-cli")
WHISPER_THREADS   = os.environ.get("WHISPER_THREADS",   "4")

# ── Job-Store ──────────────────────────────────────────────────────────────────
jobs: dict = {}
jobs_lock = threading.Lock()
rknn_model = None

def set_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)

# ── Pfad-Helfer ────────────────────────────────────────────────────────────────
def get_cpu_model_path(model: str) -> str:
    return os.path.join(WHISPER_MODEL_DIR, f"ggml-{model}.bin")

def get_rknn_paths(model: str) -> tuple[str, str]:
    encoder = os.path.join(RKNN_MODEL_DIR, f"whisper-{model}-encoder.rknn")
    decoder = os.path.join(RKNN_MODEL_DIR, f"whisper-{model}-decoder.rknn")
    return encoder, decoder

# ── Modell-Vorbereitung ────────────────────────────────────────────────────────
def ensure_cpu_model(model: str):
    model_path = get_cpu_model_path(model)
    if not os.path.exists(model_path):
        log.info(f"Modell nicht gefunden, starte Download: {model_path}")
        subprocess.run(["/app/download_model.sh"], env={
            **os.environ,
            "WHISPER_MODEL":     model,
            "WHISPER_MODEL_DIR": WHISPER_MODEL_DIR,
        }, check=True)

def ensure_rknn_models(model: str):
    encoder, decoder = get_rknn_paths(model)
    if not os.path.exists(encoder) or not os.path.exists(decoder):
        log.info("RKNN-Modelle nicht gefunden, starte Download...")
        subprocess.run(["/app/download_rknn_models.sh"], env={
            **os.environ,
            "WHISPER_MODEL":  model,
            "RKNN_MODEL_DIR": RKNN_MODEL_DIR,
        }, check=True)

def load_rknn_model(model: str):
    global rknn_model
    if rknn_model is None:
        from rknn_inference import WhisperRKNN
        encoder, decoder = get_rknn_paths(model)
        rknn_model = WhisperRKNN(encoder, decoder, model_size=model)
    return rknn_model

# ── Worker ─────────────────────────────────────────────────────────────────────
def worker_transcribe(job_id: str, audio_path: str, backend: str, model: str):
    set_job(job_id, status="processing", progress=10)
    try:
        if backend == "rknn":
            ensure_rknn_models(model)
            set_job(job_id, progress=30)
            rknn = load_rknn_model(model)
            set_job(job_id, progress=50)
            transcript = rknn.transcribe(audio_path)
        else:
            ensure_cpu_model(model)
            set_job(job_id, progress=30)
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

        set_job(job_id, status="done", progress=100, transcript=transcript,
                model=model, backend=backend)
        log.info(f"Job {job_id} abgeschlossen ({backend}/{model})")

    except Exception as e:
        log.error(f"Job {job_id} fehlgeschlagen: {e}")
        set_job(job_id, status="error", error=str(e))

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(silent=True) or {}

    audio_path = data.get("audio_path")
    if not audio_path:
        return jsonify({"error": "audio_path fehlt"}), 400
    if not os.path.exists(audio_path):
        return jsonify({"error": f"Datei nicht gefunden: {audio_path}"}), 400

    backend = data.get("backend", WHISPER_BACKEND)
    model   = data.get("model",   WHISPER_MODEL)

    if backend not in ("cpu", "rknn"):
        return jsonify({"error": f"Ungültiges Backend: {backend}"}), 400
    if model not in ("tiny", "base", "small", "medium"):
        return jsonify({"error": f"Ungültiges Modell: {model}"}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "backend": backend, "model": model}

    threading.Thread(target=worker_transcribe,
                     args=(job_id, audio_path, backend, model),
                     daemon=True).start()

    log.info(f"Job {job_id} gestartet – backend={backend}, model={model}")
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job nicht gefunden"}), 404
    return jsonify(job)


@app.route("/health", methods=["GET"])
def health():
    if WHISPER_BACKEND == "rknn":
        try:
            encoder, _ = get_rknn_paths(WHISPER_MODEL)
            model_ok = os.path.exists(encoder)
        except Exception:
            model_ok = False
    else:
        model_ok = os.path.exists(get_cpu_model_path(WHISPER_MODEL))

    return jsonify({
        "status":       "ok" if model_ok else "degraded",
        "model":        WHISPER_MODEL,
        "model_loaded": model_ok,
        "backend":      WHISPER_BACKEND,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
