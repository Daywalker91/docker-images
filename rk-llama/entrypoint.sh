#!/bin/bash
set -e

# Kein Symlink /dev/rknpu mehr!
# librknnrt.so erkennt DRM-basierte NPU automatisch über /dev/dri/card* und /dev/dri/renderD*
# Ein Symlink /dev/rknpu -> /dev/dri/renderD129 zwingt die Library in den legacy ioctl Pfad
# und verursacht ENOTTY. Ohne Symlink nutzt sie DRM ioctls (magic 0x64).

# NPU DRI Device prüfen
if ls /dev/dri/ > /dev/null 2>&1; then
    echo "INFO: DRI Devices vorhanden:"
    ls -la /dev/dri/
else
    echo "WARN: Keine DRI Devices gefunden – NPU möglicherweise nicht verfügbar"
fi

# NPU Frequenz auf Maximum fixieren
if [ -f /sys/class/devfreq/fdab0000.npu/governor ]; then
    echo "INFO: Setze NPU Governor auf userspace und fixiere Frequenz auf 1GHz"
    echo userspace > /sys/class/devfreq/fdab0000.npu/governor
    echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
    echo "INFO: NPU Frequenz: $(cat /sys/class/devfreq/fdab0000.npu/cur_freq)"
else
    echo "WARN: NPU devfreq nicht gefunden – Frequenz nicht gesetzt"
fi

# NPU Umgebungsvariablen für rk-llama.cpp
# RKNPU_HYBRID: Quantisierungsmodus W8A8_HADAMARD ist optimal für RK3588
export RKNPU_HYBRID="${RKNPU_HYBRID:-W8A8_HADAMARD}"

echo "INFO: Starte rk-llama Wrapper auf Port 5002..."
echo "INFO: RKNPU_HYBRID=${RKNPU_HYBRID}"

exec python app.py
