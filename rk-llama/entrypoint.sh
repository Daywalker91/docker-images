#!/bin/bash
set -e

# Symlink /dev/rknpu → /dev/dri/renderD129
# Neuere Kernel exponieren die NPU über DRI statt /dev/rknpu
# librknnrt.so sucht aber intern nach /dev/rknpu
if [ ! -e /dev/rknpu ] && [ -e /dev/dri/renderD129 ]; then
    echo "INFO: Erstelle Symlink /dev/rknpu -> /dev/dri/renderD129"
    ln -s /dev/dri/renderD129 /dev/rknpu
elif [ -e /dev/rknpu ]; then
    echo "INFO: /dev/rknpu bereits vorhanden"
else
    echo "WARN: Weder /dev/rknpu noch /dev/dri/renderD129 gefunden – NPU möglicherweise nicht verfügbar"
fi

# NPU Umgebungsvariablen für rk-llama.cpp
# RKNPU_HYBRID: Quantisierungsmodus W8A8_HADAMARD ist optimal für RK3588
export RKNPU_HYBRID="${RKNPU_HYBRID:-W8A8_HADAMARD}"

echo "INFO: Starte rk-llama Wrapper auf Port 5002..."
echo "INFO: RKNPU_HYBRID=${RKNPU_HYBRID}"

exec python app.py
