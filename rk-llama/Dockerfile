FROM ubuntu:24.04

# RK3588 (Cortex-A76/A55) is ARMv8.2-A — does NOT support PAC (ARMv8.3-A).
# Debian Trixie ARM64 packages are compiled with PAC guards causing ENOEXEC on RK3588.
# Ubuntu 24.04 ARM64 does NOT use PAC and runs correctly on ARMv8.2-A hardware.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip python3.12-venv \
        libgomp1 wget curl git \
        build-essential cmake \
        lsb-release software-properties-common gnupg apt-utils && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Use venv to avoid Ubuntu 24.04 PEP 668 restriction
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask requests huggingface_hub

# Install Clang-21 (avoids PAC compilation, better ARM64 codegen for rknpu2)
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21 && \
    apt-get install -y clang-21 libomp-21-dev libomp5 && \
    rm llvm.sh

# RKNN Runtime Library – aus rknn-toolkit2 v2.3.0 (passend zu NPU Driver v0.9.8)
RUN wget -q "https://github.com/airockchip/rknn-toolkit2/raw/v2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
         -O /usr/lib/librknnrt.so && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig

# Build rk-llama.cpp mit RKNPU2 Support und Clang-21
RUN git clone --depth=1 -b rknpu2 https://github.com/invisiofficial/rk-llama.cpp.git /build/rk-llama.cpp

WORKDIR /build/rk-llama.cpp

RUN cmake -S . -B build \
        -DLLAMA_RKNPU2=ON \
        -DCMAKE_C_COMPILER=clang-21 \
        -DCMAKE_CXX_COMPILER=clang++-21 && \
    cmake --build build -j "$(nproc)" && \
    cp build/bin/llama-server /usr/local/bin/ && \
    rm -rf /build/rk-llama.cpp

WORKDIR /app

COPY app.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /models

EXPOSE 5002

ENTRYPOINT ["/entrypoint.sh"]
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

# RKNN Runtime Library – aus rknn-toolkit2 v2.3.0 (passend zu NPU Driver v0.9.8)
RUN wget -q "https://github.com/airockchip/rknn-toolkit2/raw/v2.3.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
         -O /usr/lib/librknnrt.so && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig

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
