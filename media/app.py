from flask import Flask, request, jsonify
import subprocess
import os
import uuid
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Shared Volume Pfad (überschreibbar per Env-Variable)
AUDIO_OUTPUT_DIR = os.environ.get("AUDIO_OUTPUT_DIR", "/shared/audio")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def to_wav_16k_mono(input_path: str, output_path: str):
    """Konvertiert beliebige Audio/Video-Datei zu WAV 16kHz mono via ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "ffmpeg Timeout (>5 Min.)"
    except Exception as e:
        return False, str(e)


# ─── POST /extract/local ──────────────────────────────────────────────────────
# Erwartet: { "path": "/shared/uploads/video.mp4" }
# Gibt zurück: { "output": "/shared/audio/abc123.wav", "filename": "abc123.wav" }
@app.route("/extract/local", methods=["POST"])
def extract_local():
    data = request.get_json()

    if not data or "path" not in data:
        return jsonify({"error": "Parameter 'path' fehlt"}), 400

    input_path = data["path"]

    if not os.path.exists(input_path):
        return jsonify({"error": f"Datei nicht gefunden: {input_path}"}), 404

    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

    success, error = to_wav_16k_mono(input_path, output_path)
    if not success:
        return jsonify({"error": f"ffmpeg fehlgeschlagen: {error}"}), 500

    app.logger.info(f"local extract OK → {output_filename}")
    return jsonify({"output": output_path, "filename": output_filename})


# ─── POST /extract/url ────────────────────────────────────────────────────────
# Erwartet: { "url": "https://youtube.com/watch?v=..." }
# Gibt zurück: { "output": "...", "filename": "...", "title": "...", "description": "...", "has_description": true/false }
@app.route("/extract/url", methods=["POST"])
def extract_url():
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "Parameter 'url' fehlt"}), 400

    url = data["url"]
    temp_prefix = os.path.join(AUDIO_OUTPUT_DIR, f"tmp_{uuid.uuid4()}")
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename)

    try:
        # Metadaten holen (Titel, Beschreibung) ohne Download
        meta_result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True, text=True, timeout=60,
        )
        title = ""
        description = ""
        if meta_result.returncode == 0:
            import json
            meta = json.loads(meta_result.stdout)
            title = meta.get("title", "")
            description = meta.get("description", "")

        # Nur Audio herunterladen (-x Flag, kein Video)
        result = subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format", "best",
                "--no-playlist",
                "-o", temp_prefix + ".%(ext)s",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            return jsonify({"error": f"yt-dlp fehlgeschlagen: {result.stderr}"}), 500

        # Heruntergeladene Temp-Datei finden
        temp_files = [
            f for f in os.listdir(AUDIO_OUTPUT_DIR)
            if f.startswith(os.path.basename(temp_prefix))
        ]
        if not temp_files:
            return jsonify({"error": "yt-dlp hat keine Ausgabedatei erzeugt"}), 500

        temp_file_path = os.path.join(AUDIO_OUTPUT_DIR, temp_files[0])

        # Zu WAV 16kHz mono konvertieren
        success, error = to_wav_16k_mono(temp_file_path, output_path)
        os.remove(temp_file_path)  # Temp-Datei immer aufräumen

        if not success:
            return jsonify({"error": f"ffmpeg Konvertierung fehlgeschlagen: {error}"}), 500

        app.logger.info(f"url extract OK → {output_filename}")
        return jsonify({
            "output": output_path,
            "filename": output_filename,
            "title": title,
            "description": description,
            "has_description": len(description) > 100  # >100 Zeichen = vermutlich nützlich
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Download Timeout (>10 Min.)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── GET /health ────────────────────────────────────────────────────────────── 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
