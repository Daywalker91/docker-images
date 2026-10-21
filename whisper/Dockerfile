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
# arm64: ARMv8.2 mit dotprod + fp16 (RK3588, Cortex-A76/A55). Ohne diese Flags
# baut ggml nur generisches ARMv8 – auf dem RK1 gemessen ~3x langsamer
# (base: 89 s statt 31 s für 198 s deutsches Audio).
ARG TARGETARCH
RUN ARM_FLAGS=""; \
    if [ "$TARGETARCH" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; then \
        ARM_FLAGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; \
    fi; \
    cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    $ARM_FLAGS \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc) --target whisper-cli

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

# RKNN Runtime nur auf arm64 (NPU-Zugriff über die C-API, siehe rknn_zc.py)
COPY install_rknnrt.sh /tmp/install_rknnrt.sh
RUN chmod +x /tmp/install_rknnrt.sh && /tmp/install_rknnrt.sh && rm /tmp/install_rknnrt.sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py download_model.sh download_rknn_models.sh rknn_inference.py rknn_zc.py whisper_audio.py ./
RUN chmod +x download_model.sh download_rknn_models.sh
RUN mkdir -p /models/rknn /shared/audio
EXPOSE 5001
CMD ["python", "app.py"]
