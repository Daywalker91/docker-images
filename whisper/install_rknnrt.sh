#!/bin/bash
# RKNN Runtime (librknnrt.so) nur auf arm64 installieren.
#
# Version 2.1.0 passend zu den RKNN-Modellen (mit rknn-toolkit2 2.1.0 konvertiert,
# siehe download_rknn_models.sh): Mit librknnrt 2.3.2/2.5.0 stürzt deren Decoder
# beim ersten Aufruf ab (SIGSEGV).
#
# Früher kam die Datei aus rockchip-linux/rknpu2 – das Repo gibt es nicht mehr,
# wget legte eine leere Datei an und der Build lief trotzdem durch. Darum hier
# set -e und eine Größenprüfung.
set -e

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "librknnrt nicht verfügbar auf $ARCH (nur arm64) – wird übersprungen"
    exit 0
fi

VERSION=v2.1.0
wget -q "https://github.com/airockchip/rknn-toolkit2/raw/${VERSION}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
     -O /usr/lib/librknnrt.so
SIZE=$(stat -c %s /usr/lib/librknnrt.so)
if [ "$SIZE" -lt 1000000 ]; then
    echo "❌ librknnrt.so ist nur ${SIZE} Bytes groß – Download kaputt"
    exit 1
fi
chmod 755 /usr/lib/librknnrt.so
ldconfig
echo "librknnrt.so ${VERSION} installiert (${SIZE} Bytes)"
