from flask import Flask, request, jsonify
import subprocess
import os
import logging
import tempfile

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Konfiguration via Env-Variablen
WHISPER_MODEL     = os.environ.get("WHISPER_MODEL", "base")
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models")
WHISPER_BINARY    = os.environ.get("WHISPER_BINARY", "/usr/local/bin/whisper-cli")
WHISPER_BACKEND   = os.environ.get("WHISPER_BACKEND", "cpu")   # cpu | rknn (experimentell)
WHISPER_THREADS   = os.environ.get("WHISPER_THREADS", "4")

MODEL_PATH = os.path.join(WHISPER_MODEL_DIR, f"ggml-{WHISPER_MODEL}.bin")


def ensure_model():
    """Modell herunterladen falls nicht vorhanden."""
    if not os.path.exists(MODEL_PATH):
        app.logger.info(f"Modell nicht gefunden, starte Download: {MODEL_PATH}")
        result = subprocess.run(
            ["/app/download_model.sh"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Modell-Download fehlgeschlagen: {result.stderr}")
        app.logger.info("Modell erfolgreich heruntergeladen")


# Modell beim Start sicherstellen
try:
    ensure_model()
except Exception as e:
    app.logger.error(f"Warnung beim Start: {e}")


# ─── POST /transcribe ─────────────────────────────────────────────────────────
# Erwartet: { "audio": "/shared/audio/abc123.wav", "language": "de" }
# Gibt zurück: { "transcript": "...", "language": "de", "model": "base" }
@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json()

    if not data or "audio" not in data:
        return jsonify({"error": "Parameter 'audio' fehlt"}), 400

    audio_path = data["audio"]
    language   = data.get("language", "auto")

    if not os.path.exists(audio_path):
        return jsonify({"error": f"Audio-Datei nicht gefunden: {audio_path}"}), 404

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": f"Modell nicht gefunden: {MODEL_PATH}"}), 500

    # Ausgabe-Datei im selben Verzeichnis wie die Audio-Datei
    output_base = audio_path.replace(".wav", "")

    cmd = [
        WHISPER_BINARY,
        "-m", MODEL_PATH,
        "-f", audio_path,
        "-t", WHISPER_THREADS,
        "--output-txt",
        "--output-file", output_base,
        "--no-prints",
    ]

    # Sprache setzen falls nicht auto-detect
    if language != "auto":
        cmd += ["-l", language]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            return jsonify({"error": f"whisper fehlgeschlagen: {result.stderr}"}), 500

        # Transkript aus der .txt Ausgabedatei lesen
        transcript_path = output_base + ".txt"
        if not os.path.exists(transcript_path):
            return jsonify({"error": "Keine Ausgabedatei erzeugt"}), 500

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        # Aufräumen
        os.remove(transcript_path)

        app.logger.info(f"Transkription OK: {audio_path} → {len(transcript)} Zeichen")
        return jsonify({
            "transcript": transcript,
            "language": language,
            "model": WHISPER_MODEL,
            "backend": WHISPER_BACKEND
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Transkription Timeout (>10 Min.)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── GET /health ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    model_ok = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "ok" if model_ok else "degraded",
        "model": WHISPER_MODEL,
        "model_loaded": model_ok,
        "backend": WHISPER_BACKEND
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
