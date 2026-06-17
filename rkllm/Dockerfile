# arm64 only – rkllm läuft nur auf RK3588 NPU
FROM python:slim

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="RKLLM container – LLM Inferenz auf RK3588 NPU (echte RKLLM-Runtime, kein GGUF)"
LABEL org.opencontainers.image.licenses="MIT"

# System-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# rkllm Runtime Library von Rockchip GitHub holen
# ARG wird vom Workflow mit der aktuellen Version gesetzt
ARG RKLLM_VERSION=v1.2.1
RUN VERSION_CLEAN=$(echo ${RKLLM_VERSION} | sed 's/^v//') && \
    wget -q "https://github.com/airockchip/rknn-llm/archive/refs/tags/${RKLLM_VERSION}.tar.gz" \
         -O /tmp/rknn-llm.tar.gz && \
    tar -xzf /tmp/rknn-llm.tar.gz -C /tmp && \
    cp /tmp/rknn-llm-${VERSION_CLEAN}/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so /usr/lib/ && \
    rm -rf /tmp/rknn-llm* /tmp/rknn-llm.tar.gz

# RKNN Runtime Library – wird von librkllmrt.so intern für den NPU-Zugriff benötigt
RUN wget -q "https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
         -O /usr/lib/librknnrt.so && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py rkllm_wrapper.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

RUN mkdir -p /models

EXPOSE 5002

CMD ["/app/entrypoint.sh"]
