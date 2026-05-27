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
