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

echo "Installiere RKNN Runtime Library (librknnrt.so)..."
wget -q "https://github.com/rockchip-linux/rknpu2/raw/master/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
     -O /usr/lib/librknnrt.so && \
    chmod 755 /usr/lib/librknnrt.so && \
    ldconfig
echo "librknnrt.so installiert"
