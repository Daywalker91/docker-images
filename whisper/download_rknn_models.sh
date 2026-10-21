#!/bin/bash
# RKNN Whisper-Modelle (Encoder + Decoder mit KV-Cache) herunterladen.
# Quelle: sherpa-onnx (k2-fsa), vorkonvertiert für RK3588 mit rknn-toolkit2 2.1.0
#   https://modelscope.cn/models/csukuangfj/2026-01-05-rknn
# Die Runtime im Image ist passend dazu librknnrt 2.1.0 (siehe install_rknnrt.sh).
# Verfügbar: tiny, base (jeweils mehrsprachig).
set -e

MODEL=${WHISPER_MODEL:-base}
RKNN_DIR=${RKNN_MODEL_DIR:-/models/rknn}
BASE_URL="https://modelscope.cn/models/csukuangfj/2026-01-05-rknn/resolve/master"

case "$MODEL" in
    tiny|base) ;;
    *) echo "❌ Kein RKNN-Modell für '${MODEL}' (nur tiny, base)"; exit 1 ;;
esac

mkdir -p "$RKNN_DIR"

download_if_missing() {
    local SRC=$1 TARGET="${RKNN_DIR}/$2"
    if [ -s "$TARGET" ]; then
        echo "✅ Bereits vorhanden: $TARGET"
        return 0
    fi
    echo "⬇️  Lade ${SRC}..."
    if ! wget -q --progress=dot:giga "${BASE_URL}/${SRC}" -O "${TARGET}.part" || [ ! -s "${TARGET}.part" ]; then
        echo "❌ Download fehlgeschlagen: $SRC"
        rm -f "${TARGET}.part"
        return 1
    fi
    mv "${TARGET}.part" "$TARGET"
    echo "✅ Gespeichert: $TARGET"
}

download_if_missing "${MODEL}-encoder.rknn" "whisper-${MODEL}-encoder.rknn"
download_if_missing "${MODEL}-decoder.rknn" "whisper-${MODEL}-decoder.rknn"

echo "✅ RKNN Modelle bereit (whisper-${MODEL})"
