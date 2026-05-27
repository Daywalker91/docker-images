#!/bin/bash
# RKNN Whisper Modelle vom GitHub Release herunterladen

MODEL=${WHISPER_MODEL:-base}
RKNN_DIR=${RKNN_MODEL_DIR:-/models/rknn}
GITHUB_REPO="Daywalker91/docker-images"
RELEASE_TAG="whisper-rknn-models"

ENCODER_FILE="whisper_encoder_${MODEL}_rk3588.rknn"
DECODER_FILE="whisper_decoder_${MODEL}_rk3588.rknn"
BASE_URL="https://github.com/${GITHUB_REPO}/releases/download/${RELEASE_TAG}"

mkdir -p "$RKNN_DIR"

download_if_missing() {
    local FILE=$1
    local TARGET="${RKNN_DIR}/${FILE}"

    if [ -f "$TARGET" ]; then
        echo "✅ Bereits vorhanden: $TARGET"
        return 0
    fi

    echo "⬇️  Lade ${FILE}..."
    wget -q --show-progress "${BASE_URL}/${FILE}" -O "$TARGET"

    if [ $? -ne 0 ]; then
        echo "❌ Download fehlgeschlagen: $FILE"
        rm -f "$TARGET"
        return 1
    fi

    echo "✅ Gespeichert: $TARGET"
}

download_if_missing "$ENCODER_FILE" || exit 1
download_if_missing "$DECODER_FILE" || exit 1

echo "✅ RKNN Modelle bereit (whisper-${MODEL})"
