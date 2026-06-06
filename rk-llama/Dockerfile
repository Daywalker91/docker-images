# arm64 only – rk-llama.cpp läuft nur auf RK3588 NPU
# Multi-stage Build: Build-Stage + Runtime-Stage

# ── Build Stage ───────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    pkg-config \
    libgomp1 \
    libstdc++6 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# RKNN Runtime Library – aus rknn-llm tar.gz (gleicher Ansatz wie rkllm Container)
# Tag-Format: release-v1.2.3 → Ordner: rknn-llm-release-v1.2.3
ARG RKLLM_VERSION=release-v1.2.3
RUN wget -q "https://github.com/airockchip/rknn-llm/archive/refs/tags/${RKLLM_VERSION}.tar.gz" \
         -O /tmp/rknn-llm.tar.gz && \
    tar -xzf /tmp/rknn-llm.tar.gz -C /tmp && \
    cp /tmp/rknn-llm-${RKLLM_VERSION}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/ && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig && \
    rm -rf /tmp/rknn-llm.tar.gz /tmp/rknn-llm-${RKLLM_VERSION}

# rk-llama.cpp klonen (rknpu2 Branch) – rknn_api.h ist im Repo enthalten
RUN git clone --depth=1 --branch rknpu2 \
    https://github.com/invisiofficial/rk-llama.cpp.git /build

WORKDIR /build

# Bauen mit RKNN NPU Backend – nur llama-server Binary
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_RKNPU2=ON \
    -DRKNN_RT_LIB=/usr/lib/librknnrt.so \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --target llama-server -j$(nproc)

# ── Runtime Stage ─────────────────────────────────────────────────────────────
FROM python:slim

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="rk-llama.cpp – llama.cpp mit RK3588 NPU Backend, OpenAI-kompatibler API"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libgcc-s1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# RKNN Runtime Library – aus rknn-llm tar.gz
ARG RKLLM_VERSION=release-v1.2.3
RUN wget -q "https://github.com/airockchip/rknn-llm/archive/refs/tags/${RKLLM_VERSION}.tar.gz" \
         -O /tmp/rknn-llm.tar.gz && \
    tar -xzf /tmp/rknn-llm.tar.gz -C /tmp && \
    cp /tmp/rknn-llm-${RKLLM_VERSION}/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/ && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig && \
    rm -rf /tmp/rknn-llm.tar.gz /tmp/rknn-llm-${RKLLM_VERSION}

# llama-server Binary aus Build-Stage
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

RUN mkdir -p /models

EXPOSE 5002

CMD ["/app/entrypoint.sh"]
