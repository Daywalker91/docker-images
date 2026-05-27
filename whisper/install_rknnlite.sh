#!/bin/bash
# rknnlite nur auf arm64 installieren

ARCH=$(uname -m)
RKNN_LITE_VERSION="2.3.2"
PYVER="cp312"

if [ "$ARCH" != "aarch64" ]; then
    echo "rknnlite nicht verfuegbar auf $ARCH (nur arm64) – wird uebersprungen"
    exit 0
fi

echo "Installiere rknnlite ${RKNN_LITE_VERSION} fuer aarch64..."

URL="https://github.com/airockchip/rknn-toolkit2/releases/download/v${RKNN_LITE_VERSION}/rknn_toolkit_lite2-${RKNN_LITE_VERSION}-${PYVER}-${PYVER}-linux_aarch64.whl"
WHEEL="/tmp/rknn_lite.whl"

wget -q "$URL" -O "$WHEEL"
if [ $? -ne 0 ]; then
    echo "Download fehlgeschlagen: $URL"
    exit 1
fi

pip install --no-cache-dir "$WHEEL"
rm "$WHEEL"
echo "rknnlite installiert"
