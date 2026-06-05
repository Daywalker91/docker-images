#!/bin/bash
set -e

# Symlink /dev/rknpu → /dev/dri/renderD129
# Neuere Kernel exponieren die NPU über DRI statt /dev/rknpu
# librkllmrt.so sucht aber hardcoded nach /dev/rknpu
if [ ! -e /dev/rknpu ] && [ -e /dev/dri/renderD129 ]; then
    echo "INFO: Erstelle Symlink /dev/rknpu -> /dev/dri/renderD129"
    ln -s /dev/dri/renderD129 /dev/rknpu
elif [ -e /dev/rknpu ]; then
    echo "INFO: /dev/rknpu bereits vorhanden"
else
    echo "WARN: Weder /dev/rknpu noch /dev/dri/renderD129 gefunden – NPU möglicherweise nicht verfügbar"
fi

exec python app.py
