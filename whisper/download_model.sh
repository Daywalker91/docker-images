#!/bin/bash
# Whisper-Modell herunterladen falls noch nicht vorhanden
# Wird beim Container-Start von app.py aufgerufen

MODEL=${WHISPER_MODEL:-base}
MODEL_DIR=${WHISPER_MODEL_DIR:-/models}
MODEL_FILE="${MODEL_DIR}/ggml-${MODEL}.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Modell bereits vorhanden: $MODEL_FILE"
    exit 0
fi

echo "⬇️  Lade Whisper-Modell: ${MODEL}"

BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
URL="${BASE_URL}/ggml-${MODEL}.bin"

mkdir -p "$MODEL_DIR"
wget -q --show-progress -O "$MODEL_FILE" "$URL"

if [ $? -eq 0 ]; then
    echo "✅ Modell gespeichert: $MODEL_FILE"
else
    echo "❌ Download fehlgeschlagen"
    rm -f "$MODEL_FILE"
    exit 1
fi
