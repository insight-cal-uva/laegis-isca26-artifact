#!/bin/bash

QEMU_LAUNCH="/p/confidentialgpu/hcc/tdx-tools/qemu_launch_cvmgpu.sh"
CHECK_INTERVAL=600  # 10 mins
WORKDIR="/p/confidentialgpu/hcc/tdx-tools"

cd "$WORKDIR" || exit 1

sudo -v
( while true; do sudo -v; sleep 60; done ) &

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

while true; do
    if ! pgrep -f "/usr/bin/qemu-system-x86_64" > /dev/null; then
        log "QEMU not running. Launching..."
        sudo tmux has-session -t qemu_vm 2>/dev/null && sudo tmux kill-session -t qemu_vm
        sudo tmux new-session -d -s qemu_vm "$QEMU_LAUNCH"
        sleep 300
    else
        log "QEMU VM is running."
    fi

    sleep $CHECK_INTERVAL
done
