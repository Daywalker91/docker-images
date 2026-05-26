# Base: official python:slim (Debian)
FROM python:slim

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="Media container – ffmpeg + yt-dlp REST API (Audio-Extraktion)"
LABEL org.opencontainers.image.licenses="MIT"

# ffmpeg installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Shared Volume Mountpoint
RUN mkdir -p /shared/audio

EXPOSE 5000

CMD ["python", "app.py"]
