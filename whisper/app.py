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
# RK3588: CPU4-7 = Cortex-A76 (schnell), CPU0-3 = Cortex-A55 – whisper.cpp auf
# die großen Kerne legen. Leer = keine Bindung (z.B. amd64).
WHISPER_CPU_AFFINITY = os.environ.get("WHISPER_CPU_AFFINITY", "4-7" if os.uname().machine == "aarch64" else "")
# Beam-Search-Breite für whisper.cpp: 5 = Standard (genauer), 1 = greedy (~30 % schneller)
WHISPER_BEAM_SIZE = os.environ.get("WHISPER_BEAM_SIZE", "5")

CPU_MODELS  = ("tiny", "base", "small", "medium")
RKNN_MODELS = ("tiny", "base")     # vorkonvertiert verfügbar, siehe download_rknn_models.sh

# ── Job-Store ──────────────────────────────────────────────────────────────────
jobs: dict = {}
jobs_lock = threading.Lock()
rknn_model = None
rknn_model_name = None

def set_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)

# ── Pfad-Helfer ────────────────────────────────────────────────────────────────
def get_cpu_model_path(model: str) -> str:
    return os.path.join(WHISPER_MODEL_DIR, f"ggml-{model}.bin")

def valid_language(lang: str) -> bool:
    from whisper_audio import LANGUAGES
    return lang == "auto" or lang in LANGUAGES

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
    global rknn_model, rknn_model_name
    if rknn_model is not None and rknn_model_name != model:
        rknn_model.release()
        rknn_model = None
    if rknn_model is None:
        from rknn_inference import WhisperRKNN
        encoder, decoder = get_rknn_paths(model)
        rknn_model = WhisperRKNN(encoder, decoder, model_size=model)
        rknn_model_name = model
    return rknn_model

def resolve_backend(backend: str, model: str) -> str:
    """NPU-Modelle gibt es nur für tiny/base – größere laufen auf der CPU."""
    if backend == "rknn" and model not in RKNN_MODELS:
        log.warning(f"RKNN hat kein Modell '{model}' – nutze CPU")
        return "cpu"
    return backend

# ── Worker ─────────────────────────────────────────────────────────────────────
def worker_transcribe(job_id: str, audio_path: str, backend: str, model: str, language: str):
    set_job(job_id, status="processing", progress=10)
    try:
        if backend == "rknn":
            ensure_rknn_models(model)
            set_job(job_id, progress=30)
            rknn = load_rknn_model(model)
            set_job(job_id, progress=50)
            transcript = rknn.transcribe(audio_path, language)
        else:
            ensure_cpu_model(model)
            set_job(job_id, progress=30)
            cmd = ["taskset", "-c", WHISPER_CPU_AFFINITY] if WHISPER_CPU_AFFINITY else []
            cmd += [
                WHISPER_BINARY,
                "-m", get_cpu_model_path(model),
                "-f", audio_path,
                "-t", WHISPER_THREADS,
                "-bs", WHISPER_BEAM_SIZE,
                "--no-timestamps",
                "-l", language,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            transcript = result.stdout.strip()

        set_job(job_id, status="done", progress=100, transcript=transcript,
                model=model, backend=backend)
        log.info(f"Job {job_id} abgeschlossen ({backend}/{model})")

    except Exception as e:
        log.error(f"Job {job_id} fehlgeschlagen: {e}")
        set_job(job_id, status="error", error=str(e))

def worker_preload(job_id: str, backend: str, model: str):
    backend = resolve_backend(backend, model)
    set_job(job_id, status="processing", progress=10)
    try:
        if backend == "rknn":
            ensure_rknn_models(model)
            set_job(job_id, progress=50)
            load_rknn_model(model)
        else:
            ensure_cpu_model(model)
        set_job(job_id, status="done", progress=100, model=model, backend=backend)
        log.info(f"Preload-Job {job_id} abgeschlossen ({backend}/{model})")
    except Exception as e:
        log.error(f"Preload-Job {job_id} fehlgeschlagen: {e}")
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

    backend  = data.get("backend", WHISPER_BACKEND)
    model    = data.get("model",   WHISPER_MODEL)
    language = (data.get("language") or "auto").strip().lower()

    if backend not in ("cpu", "rknn"):
        return jsonify({"error": f"Ungültiges Backend: {backend}"}), 400
    if model not in CPU_MODELS:
        return jsonify({"error": f"Ungültiges Modell: {model}"}), 400
    if not valid_language(language):
        return jsonify({"error": f"Ungültige Sprache: {language}"}), 400
    backend = resolve_backend(backend, model)

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "backend": backend, "model": model,
                        "language": language}

    threading.Thread(target=worker_transcribe,
                     args=(job_id, audio_path, backend, model, language),
                     daemon=True).start()

    log.info(f"Job {job_id} gestartet – backend={backend}, model={model}, language={language}")
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
        "backend":      WHISPER_BACKEND})


@app.route("/preload", methods=["POST"])
def preload():
    """
    Lädt das Modell proaktiv herunter/bereit, ohne eine echte
    Transkription auszuführen. Nützlich für einen "Modell laden"-
    Button in der UI, da /health nur prüft ob die Datei existiert,
    aber sie nie selbst herunterlädt.
    """
    data    = request.get_json(silent=True) or {}
    backend = data.get("backend", WHISPER_BACKEND)
    model   = data.get("model",   WHISPER_MODEL)

    if backend not in ("cpu", "rknn"):
        return jsonify({"error": f"Ungültiges Backend: {backend}"}), 400
    if model not in CPU_MODELS:
        return jsonify({"error": f"Ungültiges Modell: {model}"}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "backend": resolve_backend(backend, model),
                         "model": model, "type": "preload"}

    threading.Thread(target=worker_preload,
                     args=(job_id, backend, model),
                     daemon=True).start()

    log.info(f"Preload-Job {job_id} gestartet – backend={backend}, model={model}")
    return jsonify({"job_id": job_id, "status": "queued"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
