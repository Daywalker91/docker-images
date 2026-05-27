from flask import Flask, request, jsonify
import os
import logging
import subprocess

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Konfiguration via Env-Variablen
RKLLM_MODEL     = os.environ.get("RKLLM_MODEL", "1.5B")      # 1.5B | 3B | 7B
RKLLM_MODEL_DIR = os.environ.get("RKLLM_MODEL_DIR", "/models")
MAX_CONTEXT_LEN = int(os.environ.get("RKLLM_MAX_CONTEXT", "4096"))
MAX_NEW_TOKENS  = int(os.environ.get("RKLLM_MAX_NEW_TOKENS", "2048"))
TEMPERATURE     = float(os.environ.get("RKLLM_TEMPERATURE", "0.7"))

# Modell-Dateiname anhand der Größe
MODEL_FILES = {
    "1.5B": "Qwen2.5-1.5B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm",
    "3B":   "Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm",
    "7B":   "Qwen2.5-7B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm",
}

MODEL_PATH = os.path.join(RKLLM_MODEL_DIR, MODEL_FILES.get(RKLLM_MODEL, ""))
model = None


def ensure_model():
    """Modell herunterladen falls nicht vorhanden."""
    if not os.path.exists(MODEL_PATH):
        app.logger.info(f"Modell nicht gefunden, starte Download: {MODEL_PATH}")
        result = subprocess.run(
            ["/app/download_model.sh"],
            capture_output=True, text=True,
            env={**os.environ, "RKLLM_MODEL": RKLLM_MODEL, "RKLLM_MODEL_DIR": RKLLM_MODEL_DIR}
        )
        if result.returncode != 0:
            raise RuntimeError(f"Modell-Download fehlgeschlagen:\n{result.stderr}")
        app.logger.info("Modell erfolgreich heruntergeladen")


def load_model():
    global model
    ensure_model()
    from rkllm_wrapper import RKLLMModel
    app.logger.info(f"Lade RKLLM Modell: {MODEL_PATH}")
    model = RKLLMModel(
        model_path=MODEL_PATH,
        max_context_len=MAX_CONTEXT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    app.logger.info("✅ Modell bereit")


# Modell beim Start laden
try:
    load_model()
except Exception as e:
    app.logger.error(f"Fehler beim Laden des Modells: {e}")


# ─── POST /generate ───────────────────────────────────────────────────────────
# Erwartet: { "prompt": "Extrahiere das Rezept aus folgendem Text: ..." }
# Gibt zurück: { "response": "...", "model": "1.5B" }
@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Modell nicht geladen"}), 503

    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Parameter 'prompt' fehlt"}), 400

    prompt = data["prompt"]

    try:
        response = model.generate(prompt)
        app.logger.info(f"Inferenz OK → {len(response)} Zeichen")
        return jsonify({
            "response": response,
            "model": f"Qwen2.5-{RKLLM_MODEL}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── GET /health ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    model_ok = model is not None and os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "ok" if model_ok else "degraded",
        "model":  f"Qwen2.5-{RKLLM_MODEL}",
        "model_loaded": model_ok,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
