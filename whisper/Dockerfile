# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# whisper.cpp für CPU-Backend kompilieren
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
    && cmake --build build --config Release -j$(nproc)

RUN cp build/bin/whisper-cli /usr/local/bin/whisper-cli 2>/dev/null || \
    cp build/bin/main /usr/local/bin/whisper-cli


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="whisper.cpp container – Audio-Transkription REST API (CPU + RKNN)"
LABEL org.opencontainers.image.licenses="MIT"

# System-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# whisper-cli Binary aus Stage 1
COPY --from=builder /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli
RUN chmod +x /usr/local/bin/whisper-cli

# rknnlite nur auf arm64 installieren (läuft nur auf RK3588)
ARG RKNN_LITE_VERSION=2.3.2
RUN arch=$(uname -m) && \
    if [ "$arch" = "aarch64" ]; then \
        echo "Installiere rknnlite fuer aarch64..." && \
        wget -q "https://github.com/airockchip/rknn-toolkit2/releases/download/v${RKNN_LITE_VERSION}/rknn_toolkit_lite2-${RKNN_LITE_VERSION}-cp312-cp312-linux_aarch64.whl" \
             -O /tmp/rknn_lite.whl && \
        pip install --no-cache-dir /tmp/rknn_lite.whl && \
        rm /tmp/rknn_lite.whl && \
        echo "rknnlite installiert"; \
    else \
        echo "rknnlite nicht verfuegbar auf $arch (nur arm64/RKNN-Backend)"; \
    fi

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py download_model.sh download_rknn_models.sh rknn_inference.py ./
RUN chmod +x download_model.sh download_rknn_models.sh

RUN mkdir -p /models/rknn /shared/audio

EXPOSE 5001

CMD ["python", "app.py"]
