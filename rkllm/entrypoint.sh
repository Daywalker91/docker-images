#!/bin/bash
set -e

# Kein /dev/rknpu Symlink mehr.
# Unbestätigt ob librkllmrt.so diesen überhaupt braucht (closed-source, kein
# öffentlicher Beweis gefunden) – wir lassen ihn bewusst weg und beobachten
# das Verhalten im Log. Falls rkllm_init mit einem Device-Fehler fehlschlägt,
# ist das hier der erste Verdächtige.

if ls /dev/dri/ > /dev/null 2>&1; then
    echo "INFO: DRI Devices vorhanden:"
    ls -la /dev/dri/
else
    echo "WARN: Keine DRI Devices gefunden – NPU möglicherweise nicht verfügbar"
fi

# NPU Frequenz: nicht fixiert, Governor regelt dynamisch (devfreq Standardverhalten).
# Falls die Performance nicht ausreicht, kann hier optional die Frequenz auf
# Maximum (1GHz) fixiert werden, wie beim rk-llama.cpp Container:
#
# if [ -f /sys/class/devfreq/fdab0000.npu/governor ]; then
#     echo "INFO: Setze NPU Governor auf userspace und fixiere Frequenz auf 1GHz"
#     echo userspace > /sys/class/devfreq/fdab0000.npu/governor
#     echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
#     echo "INFO: NPU Frequenz: $(cat /sys/class/devfreq/fdab0000.npu/cur_freq)"
# else
#     echo "WARN: NPU devfreq nicht gefunden – Frequenz nicht gesetzt"
# fi

echo "INFO: Starte RKLLM Wrapper auf Port 5002..."
echo "INFO: RKLLM_ENABLED_CPUS_MASK=${RKLLM_ENABLED_CPUS_MASK:-<default>}"
echo "INFO: RKLLM_ENABLED_CPUS_NUM=${RKLLM_ENABLED_CPUS_NUM:-<default>}"

exec python app.py
