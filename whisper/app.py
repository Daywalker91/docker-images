from flask import Flask, request, jsonify
import subprocess
import os
import uuid
import logging
import threading

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Konfiguration via Env-Variablen
WHISPER_MODEL     = os.environ.get("WHISPER_MODEL", "base")
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models")
WHISPER_BINARY    = os.environ.get("WHISPER_BINARY", "/usr/local/bin/whisper-cli")
WHISPER_BACKEND   = os.environ.get("WHISPER_BACKEND", "cpu")   # cpu | rknn
WHISPER_THREADS   = os.environ.get("WHISPER_THREADS", "4")
RKNN_MODEL_DIR    = os.environ.get("RKNN_MODEL_DIR", "/models/rknn")

# Pfade
CPU_MODEL_PATH    = os.path.join(WHISPER_MODEL_DIR, f"ggml-{WHISPER_MODEL}.bin")
RKNN_ENCODER_PATH = os.path.join(RKNN_MODEL_DIR, f"whisper_encoder_{WHISPER_MODEL}_rk3588.rknn")
RKNN_DECODER_PATH = os.path.join(RKNN_MODEL_DIR, f"whisper_decoder_{WHISPER_MODEL}_rk3588.rknn")

# In-Memory Job-Store { job_id: { status, progress, transcript, language, model, backend, error } }
jobs = {}
jobs_lock = threading.Lock()

# RKNN Modell-Instanz
rknn_model = None


def set_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def ensure_cpu_model():
    if not os.path.exists(CPU_MODEL_PATH):
        app.logger.info(f"CPU Modell nicht gefunden, starte Download: {CPU_MODEL_PATH}")
        result = subprocess.run(["/app/download_model.sh"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CPU Modell-Download fehlgeschlagen: {result.stderr}")


def ensure_rknn_models():
    if not os.path.exists(RKNN_ENCODER_PATH) or not os.path.exists(RKNN_DECODER_PATH):
        app.logger.info("RKNN Modelle nicht gefunden, starte Download...")
        result = subprocess.run(
            ["/app/download_rknn_models.sh"],
            capture_output=True, text=True,
            env={**os.environ, "WHISPER_MODEL": WHISPER_MODEL, "RKNN_MODEL_DIR": RKNN_MODEL_DIR}
        )
        if result.returncode != 0:
            raise RuntimeError(f"RKNN Modell-Download fehlgeschlagen: {result.stderr}")


def load_rknn_model():
    global rknn_model
    from rknn_inference import WhisperRKNN
    ensure_rknn_models()
    rknn_model = WhisperRKNN(
        encoder_path=RKNN_ENCODER_PATH,
        decoder_path=RKNN_DECODER_PATH,
        model_size=WHISPER_MODEL,
    )


def worker_transcribe(job_id: str, audio_path: str, language: str,
                      model: str, backend: str):
    """Background-Worker für Transkription (CPU oder RKNN)."""
    try:
        set_job(job_id, status="processing", progress=10)

        if backend == "rknn":
            # ── RKNN Backend (NPU) ─────────────────────────────────────────
            if rknn_model is None:
                set_job(job_id, status="error", progress=0, error="RKNN Modell nicht geladen")
                return
            transcript = rknn_model.transcribe(audio_path, language=language)

        else:
            # ── CPU Backend (whisper-cli) ───────────────────────────────────
            output_base = audio_path.replace(".wav", "")
            cmd = [
                WHISPER_BINARY,
                "-m", CPU_MODEL_PATH,
                "-f", audio_path,
                "-t", WHISPER_THREADS,
                "--output-txt",
                "--output-file", output_base,
                "--no-prints",
            ]
            if language != "auto":
                cmd += ["-l", language]

            set_job(job_id, status="processing", progress=30)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                set_job(job_id, status="error", progress=0,
                        error=f"whisper fehlgeschlagen: {result.stderr}")
                return

            transcript_path = output_base + ".txt"
            if not os.path.exists(transcript_path):
                set_job(job_id, status="error", progress=0, error="Keine Ausgabedatei erzeugt")
                return

            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            os.remove(transcript_path)

        set_job(job_id, status="done", progress=100,
                transcript=transcript,
                language=language,
                model=model,
                backend=backend)
        app.logger.info(f"[{job_id}] transcribe done ({backend}): {len(transcript)} Zeichen")

    except subprocess.TimeoutExpired:
        set_job(job_id, status="error", progress=0, error="Timeout (>10 Min.)")
    except Exception as e:
        set_job(job_id, status="error", progress=0, error=str(e))


# ── Backend beim Start initialisieren ─────────────────────────────────────────
app.logger.info(f"Backend: {WHISPER_BACKEND} | Modell: {WHISPER_MODEL}")
try:
    if WHISPER_BACKEND == "rknn":
        load_rknn_model()
    else:
        ensure_cpu_model()
except Exception as e:
    app.logger.error(f"Fehler beim Start: {e}")


# ─── POST /transcribe ─────────────────────────────────────────────────────────
# Erwartet: { "audio_path": "/shared/audio/abc123.wav", "language": "de",
#             "model": "base", "backend": "whisper_cpp" }
# Gibt zurück: { "job_id": "xyz789", "status": "queued" }
@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()
    if not data or "audio_path" not in data:
        return jsonify({"error": "Parameter 'audio_path' fehlt"}), 400

    audio_path = data["audio_path"]
    language   = data.get("language", "auto")
    model      = data.get("model", WHISPER_MODEL)
    backend    = data.get("backend", WHISPER_BACKEND)

    if not os.path.exists(audio_path):
        return jsonify({"error": f"Audio-Datei nicht gefunden: {audio_path}"}), 404

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0}

    threading.Thread(
        target=worker_transcribe,
        args=(job_id, audio_path, language, model, backend),
        daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "queued"})


# ─── GET /job/<job_id> ────────────────────────────────────────────────────────
# Gibt zurück: { "status": "queued|processing|done|error",
#                "progress": 0-100,
#                "transcript": "...",  ← nur wenn done
#                "language": "de",
#                "model": "base",
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
    if WHISPER_BACKEND == "rknn":
        model_ok = rknn_model is not None
    else:
        model_ok = os.path.exists(CPU_MODEL_PATH)

    return jsonify({
        "status":       "ok" if model_ok else "degraded",
        "model":        WHISPER_MODEL,
        "model_loaded": model_ok,
        "backend":      WHISPER_BACKEND,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
