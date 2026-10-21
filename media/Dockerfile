# Base: official python:3.13-slim (Debian)
# Feste Python-Version statt python:slim: das wandert sonst auf jede neue
# Version (2026-10: 3.15), für die es anfangs noch keine fertigen Pakete gibt –
# z.B. brotli ließ sich dann ohne Compiler nicht installieren.
FROM python:3.13-slim

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

# YouTube braucht seit yt-dlp 2025.11 eine JavaScript-Runtime (deno) plus die
# EJS-Skripte, sonst nur eingeschränkt ("deprecated", Formate fehlen).
# curl-cffi: Browser-Imitation für Seiten, die einfache Anfragen abweisen.
# Die Pakete gibt es nur für amd64/arm64 – arm/v7 bleibt beim schlichten yt-dlp.
RUN case "$(uname -m)" in \
        x86_64|aarch64) pip install --no-cache-dir "yt-dlp[default,deno,curl-cffi]" ;; \
        *) echo "Keine deno/curl-cffi-Pakete für $(uname -m) – YouTube eingeschränkt" ;; \
    esac

COPY app.py .

# Shared Volume Mountpoint
RUN mkdir -p /shared/audio

EXPOSE 5000

CMD ["python", "app.py"]
