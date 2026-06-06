#!/bin/bash
# rknnlite nur auf arm64 installieren

ARCH=$(uname -m)

if [ "$ARCH" != "aarch64" ]; then
    echo "rknnlite nicht verfuegbar auf $ARCH (nur arm64) – wird uebersprungen"
    exit 0
fi

echo "Installiere rknn-toolkit-lite2 via PyPI..."
pip install --no-cache-dir rknn-toolkit-lite2
echo "rknnlite installiert"

echo "Installiere RKNN Runtime Library (librknnrt.so) via git sparse-checkout..."
git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/airockchip/rknn-toolkit2.git /tmp/rknn-tk2 && \
    cd /tmp/rknn-tk2 && \
    git sparse-checkout set rknpu2/runtime/Linux/librknn_api/aarch64 && \
    cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/ && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig && \
    rm -rf /tmp/rknn-tk2
echo "librknnrt.so installiert"
