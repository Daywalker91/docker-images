#!/bin/bash
# RKLLM Modell von HuggingFace herunterladen

MODEL_SIZE=${RKLLM_MODEL:-1.5B}
MODEL_DIR=${RKLLM_MODEL_DIR:-/models}

# HuggingFace Repos pro Modellgröße
case "$MODEL_SIZE" in
    "1.5B")
        HF_REPO="c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4"
        HF_FILE="Qwen2.5-1.5B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm"
        ;;
    "3B")
        HF_REPO="c01zaut/Qwen2.5-3B-Instruct-rk3588-1.1.1"
        HF_FILE="Qwen2.5-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm"
        ;;
    "7B")
        HF_REPO="c01zaut/Qwen2.5-7B-Instruct-RK3588-1.1.4"
        HF_FILE="Qwen2.5-7B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm"
        ;;
    *)
        echo "Unbekannte Modellgröße: $MODEL_SIZE (1.5B | 3B | 7B)"
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_DIR}/${HF_FILE}"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ Modell bereits vorhanden: $MODEL_PATH"
    exit 0
fi

echo "⬇️  Lade Qwen2.5-${MODEL_SIZE} RKLLM Modell..."
echo "   Repo: $HF_REPO"
echo "   File: $HF_FILE"

mkdir -p "$MODEL_DIR"

python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${HF_FILE}',
    cache_dir='/tmp/hf_cache'
)
shutil.copy(path, '${MODEL_PATH}')
print(f'✅ Modell gespeichert: ${MODEL_PATH}')
"

if [ $? -ne 0 ]; then
    echo "❌ Download fehlgeschlagen"
    rm -f "$MODEL_PATH"
    exit 1
fi
