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

# Install Clang-21 (same as rkllama reference build)
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21 && \
    apt-get install -y clang-21 libomp-21-dev libomp5 && \
    rm llvm.sh

# Build rk-llama.cpp mit RKNPU2 Support und Clang-21
# No -march flags, no extra cmake options — exactly like NotPunchnox rkllama reference build
RUN git clone --depth=1 -b rknpu2 https://github.com/invisiofficial/rk-llama.cpp.git /build/rk-llama.cpp

WORKDIR /build/rk-llama.cpp

# librknnrt.so aus Repo sichern, dann bauen, dann zurückschieben
RUN mv /build/rk-llama.cpp/ggml/src/ggml-rknpu2/libs/librknnrt.so /tmp/librknnrt.so && \
    cmake -S . -B build \
        -DLLAMA_RKNPU2=ON \
        -DCMAKE_C_COMPILER=clang-21 \
        -DCMAKE_CXX_COMPILER=clang++-21 && \
    cmake --build build -j "$(nproc)" && \
    cp build/bin/llama-server /usr/local/bin/ && \
    find build -name "*.so*" -exec cp {} /usr/lib/ \; && \
    ldconfig && \
    rm -rf /build/rk-llama.cpp && \
    mkdir -p /build/rk-llama.cpp/ggml/src/ggml-rknpu2/libs && \
    mv /tmp/librknnrt.so /build/rk-llama.cpp/ggml/src/ggml-rknpu2/libs/librknnrt.so

WORKDIR /app

COPY app.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /models

EXPOSE 5002

ENTRYPOINT ["/entrypoint.sh"]
