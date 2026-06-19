# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth=1 https://github.com/ggerganov/whisper.cpp /whisper.cpp
WORKDIR /whisper.cpp
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc)

RUN cp build/bin/whisper-cli /usr/local/bin/whisper-cli 2>/dev/null || \
    cp build/bin/main /usr/local/bin/whisper-cli 2>/dev/null || \
    (echo "FEHLER: whisper-cli Binary nicht gefunden!" && find /whisper.cpp/build -type f -name "whisper*" -o -name "main" && exit 1)

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="whisper.cpp container – Audio-Transkription REST API (CPU + RKNN)"
LABEL org.opencontainers.image.licenses="MIT"

# ffmpeg: wird von whisper_audio.load_audio() für den RKNN-Pfad gebraucht
# (Audio dekodieren/resamplen, exakt wie es openai-whisper intern auch via
# Subprocess tut). Für den CPU-Pfad (whisper-cli) nicht zwingend nötig, da
# der Media-Service WAV-Dateien bereits vorher extrahiert, schadet aber nicht.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli
RUN chmod +x /usr/local/bin/whisper-cli && whisper-cli --help || echo "FEHLER: Binary nicht funktionsfähig"

# rknnlite nur auf arm64 – separates Script um Shell-Expansion-Probleme zu vermeiden
COPY install_rknnlite.sh /tmp/install_rknnlite.sh
RUN chmod +x /tmp/install_rknnlite.sh && /tmp/install_rknnlite.sh && rm /tmp/install_rknnlite.sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py download_model.sh download_rknn_models.sh rknn_inference.py whisper_audio.py ./
RUN chmod +x download_model.sh download_rknn_models.sh
RUN mkdir -p /models/rknn /shared/audio
EXPOSE 5001
CMD ["python", "app.py"]
