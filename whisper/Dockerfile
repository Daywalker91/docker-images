# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# whisper.cpp kompilieren
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# whisper.cpp Version via Build-Arg steuerbar (wird vom Workflow gesetzt)
ARG WHISPER_VERSION=master
RUN if [ "${WHISPER_VERSION}" = "master" ]; then \
        git clone --depth=1 https://github.com/ggerganov/whisper.cpp /whisper.cpp; \
    else \
        git clone --depth=1 --branch ${WHISPER_VERSION} https://github.com/ggerganov/whisper.cpp /whisper.cpp; \
    fi

WORKDIR /whisper.cpp
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build build --config Release -j$(nproc)

# Binary-Name hat sich in neueren Versionen geändert (main → whisper-cli)
# Wir normalisieren auf 'whisper-cli'
RUN cp build/bin/whisper-cli /usr/local/bin/whisper-cli 2>/dev/null || \
    cp build/bin/main /usr/local/bin/whisper-cli


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:slim

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="whisper.cpp container – Audio-Transkription REST API"
LABEL org.opencontainers.image.licenses="MIT"

# Nur das fertige Binary kopieren – kein Build-Toolchain im Runtime-Image
COPY --from=builder /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli
RUN chmod +x /usr/local/bin/whisper-cli

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY download_model.sh .
RUN chmod +x download_model.sh

# Modell-Verzeichnis (als Volume mounten, nicht im Image)
RUN mkdir -p /models /shared/audio

EXPOSE 5001

CMD ["python", "app.py"]
