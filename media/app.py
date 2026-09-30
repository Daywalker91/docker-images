from flask import Flask, request, jsonify, send_file
import subprocess
import os
import uuid
import logging
import threading
import json
import glob
import shutil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Shared Volume Pfad (überschreibbar per Env-Variable)
AUDIO_OUTPUT_DIR = os.environ.get("AUDIO_OUTPUT_DIR", "/shared/audio")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# In-Memory Job-Store { job_id: { status, progress, audio_path, title, description, has_description, error } }
jobs = {}
jobs_lock = threading.Lock()


def set_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def to_wav_16k_mono(input_path: str, output_path: str):
    """Konvertiert beliebige Audio/Video-Datei zu WAV 16kHz mono via ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "ffmpeg Timeout (>5 Min.)"
    except Exception as e:
        return False, str(e)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def thumbnail_path(job_id: str) -> str:
    return os.path.join(AUDIO_OUTPUT_DIR, f"{job_id}.jpg")


def frame_from_video(video_path: str, out_path: str) -> bool:
    """Standbild aus einem Video als JPG (bei ~30 % der Laufzeit – der Anfang
    ist bei Reels oft ein Titelbild oder schwarz). Liefert True bei Erfolg."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip() or 0)
    except Exception:
        duration = 0
    seek = f"{duration * 0.3:.2f}" if duration > 1 else "0"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", seek, "-i", video_path, "-frames:v", "1",
             "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", out_path],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def image_to_jpg(src: str, out_path: str) -> bool:
    """Heruntergeladenes Thumbnail (webp/png/…) einheitlich als JPG speichern."""
    if src.lower().endswith((".jpg", ".jpeg")):
        shutil.move(src, out_path)
        return True
    try:
        result = subprocess.run(["ffmpeg", "-y", "-i", src, "-q:v", "3", out_path],
                                capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and os.path.exists(out_path)
    finally:
        if os.path.exists(src):
            os.remove(src)


def worker_local(job_id: str, input_path: str, output_path: str):
    """Background-Worker für lokale Datei-Extraktion."""
    try:
        set_job(job_id, status="processing", progress=10)
        success, error = to_wav_16k_mono(input_path, output_path)
        if not success:
            set_job(job_id, status="error", progress=0, error=f"ffmpeg fehlgeschlagen: {error}")
            return
        # Vorschaubild aus dem Video (bei reinen Audiodateien schlägt das still fehl)
        thumb = thumbnail_path(job_id)
        has_thumb = frame_from_video(input_path, thumb)

        set_job(job_id, status="done", progress=100, audio_path=output_path,
                filename=os.path.basename(output_path),
                thumbnail_file=os.path.basename(thumb) if has_thumb else "")
        app.logger.info(f"[{job_id}] local extract done")
    except Exception as e:
        set_job(job_id, status="error", progress=0, error=str(e))


def worker_url(job_id: str, url: str, temp_prefix: str, output_path: str):
    """Background-Worker für URL-Extraktion via yt-dlp."""
    try:
        # Schritt 1: Metadaten holen (Titel, Beschreibung)
        set_job(job_id, status="downloading", progress=5)
        meta_result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True, text=True, timeout=60,
        )
        title = ""
        description = ""
        thumbnail = ""
        if meta_result.returncode == 0:
            meta = json.loads(meta_result.stdout)
            title       = meta.get("title", "")
            description = meta.get("description", "")
            thumbnail   = meta.get("thumbnail", "")

        # Schritt 2: Audio herunterladen
        set_job(job_id, status="downloading", progress=20)
        # --write-thumbnail: Vorschaubild sofort mitnehmen – Social-Media-
        # Thumbnail-Links sind signiert und laufen schnell ab
        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "best", "--no-playlist",
             "--write-thumbnail",
             "-o", temp_prefix + ".%(ext)s", url],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            set_job(job_id, status="error", progress=0, error=f"yt-dlp fehlgeschlagen: {result.stderr}")
            return

        # Vorschaubild (falls geschrieben) von der Audiodatei trennen
        prefix = os.path.basename(temp_prefix)
        thumb = thumbnail_path(job_id)
        has_thumb = False
        for f in os.listdir(AUDIO_OUTPUT_DIR):
            if f.startswith(prefix) and f.lower().endswith(IMAGE_EXTS):
                if not has_thumb:
                    has_thumb = image_to_jpg(os.path.join(AUDIO_OUTPUT_DIR, f), thumb)
                elif os.path.exists(os.path.join(AUDIO_OUTPUT_DIR, f)):
                    os.remove(os.path.join(AUDIO_OUTPUT_DIR, f))

        # Kein Thumbnail (z.B. manche Facebook-Reels): kleinste Videoversion
        # laden und ein Standbild herausschneiden
        if not has_thumb:
            set_job(job_id, status="downloading", progress=60)
            vid = subprocess.run(
                ["yt-dlp", "-f", "worst[vcodec!=none]/worst", "--no-playlist",
                 "-o", temp_prefix + "_video.%(ext)s", url],
                capture_output=True, text=True, timeout=300,
            )
            for f in glob.glob(temp_prefix + "_video.*"):
                if vid.returncode == 0 and not has_thumb:
                    has_thumb = frame_from_video(f, thumb)
                os.remove(f)

        # Temp-Audiodatei finden
        temp_files = [f for f in os.listdir(AUDIO_OUTPUT_DIR)
                      if f.startswith(prefix) and not f.lower().endswith(IMAGE_EXTS)
                      and "_video." not in f]
        if not temp_files:
            set_job(job_id, status="error", progress=0, error="yt-dlp hat keine Ausgabedatei erzeugt")
            return

        temp_file_path = os.path.join(AUDIO_OUTPUT_DIR, temp_files[0])

        # Schritt 3: Zu WAV 16kHz mono konvertieren
        set_job(job_id, status="processing", progress=70)
        success, error = to_wav_16k_mono(temp_file_path, output_path)
        os.remove(temp_file_path)

        if not success:
            set_job(job_id, status="error", progress=0, error=f"ffmpeg fehlgeschlagen: {error}")
            return

        set_job(job_id, status="done", progress=100,
                audio_path=output_path,
                filename=os.path.basename(output_path),
                title=title,
                description=description,
                has_description=len(description) > 100,
                thumbnail_url=thumbnail,
                thumbnail_file=os.path.basename(thumb) if has_thumb else "")
        app.logger.info(f"[{job_id}] url extract done")

    except subprocess.TimeoutExpired:
        set_job(job_id, status="error", progress=0, error="Timeout (>10 Min.)")
    except Exception as e:
        set_job(job_id, status="error", progress=0, error=str(e))


# ─── POST /extract/local ──────────────────────────────────────────────────────
# Erwartet: { "path": "/shared/uploads/video.mp4" }
# Gibt zurück: { "job_id": "abc123", "status": "queued" }
@app.route("/extract/local", methods=["POST"])
def extract_local():
    data = request.get_json()
    if not data or "path" not in data:
        return jsonify({"error": "Parameter 'path' fehlt"}), 400

    input_path = data["path"]
    if not os.path.exists(input_path):
        return jsonify({"error": f"Datei nicht gefunden: {input_path}"}), 404

    job_id      = str(uuid.uuid4())
    output_path = os.path.join(AUDIO_OUTPUT_DIR, f"{job_id}.wav")

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0}

    threading.Thread(
        target=worker_local, args=(job_id, input_path, output_path), daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "queued"})


# ─── POST /extract/url ────────────────────────────────────────────────────────
# Erwartet: { "url": "https://youtube.com/watch?v=..." }
# Gibt zurück: { "job_id": "abc123", "status": "queued" }
@app.route("/extract/url", methods=["POST"])
def extract_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Parameter 'url' fehlt"}), 400

    url         = data["url"]
    job_id      = str(uuid.uuid4())
    temp_prefix = os.path.join(AUDIO_OUTPUT_DIR, f"tmp_{job_id}")
    output_path = os.path.join(AUDIO_OUTPUT_DIR, f"{job_id}.wav")

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0}

    threading.Thread(
        target=worker_url, args=(job_id, url, temp_prefix, output_path), daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "queued"})


# ─── GET /job/<job_id> ────────────────────────────────────────────────────────
# Gibt zurück: { "status": "queued|downloading|processing|done|error",
#                "progress": 0-100,
#                "audio_path": "...",   ← nur wenn done
#                "title": "...",         ← nur bei URL
#                "error": "..." }        ← nur wenn error
@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job nicht gefunden"}), 404
    return jsonify(job)


# ─── GET /thumbnail/<job_id> ──────────────────────────────────────────────────
# Liefert das beim Job gesicherte Vorschaubild (JPG) oder 404.
@app.route("/thumbnail/<job_id>", methods=["GET"])
def thumbnail(job_id):
    path = thumbnail_path(os.path.basename(job_id))
    if not os.path.exists(path):
        return jsonify({"error": "Kein Vorschaubild"}), 404
    return send_file(path, mimetype="image/jpeg")


# ─── DELETE /file/<filename> ──────────────────────────────────────────────────
# Räumt nach dem Import auf: Audiodatei + Vorschaubild desselben Jobs.
# (Die App ruft das nach dem Speichern auf – mediaDeleteFile() in PHP.)
@app.route("/file/<filename>", methods=["DELETE"])
def delete_file(filename):
    name = os.path.basename(filename)
    job_id = os.path.splitext(name)[0]
    removed = []
    for path in (os.path.join(AUDIO_OUTPUT_DIR, name), thumbnail_path(job_id)):
        if os.path.isfile(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    return jsonify({"ok": True, "removed": removed})


# ─── GET /health ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
