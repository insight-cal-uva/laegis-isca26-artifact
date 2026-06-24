#!/bin/bash

SSH_CMD="ssh -p10026 root@localhost"
CHECK_INTERVAL_BOOT=30
VM_PORT=10026
WORKDIR="/p/confidentialgpu/hcc/tdx-tools"

cd "$WORKDIR" || exit 1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

while true; do
    if nc -z localhost $VM_PORT; then
        log "VM is reachable. Launching job..."

        $SSH_CMD "tmux new-session -d -s profile 'cd /asplos/uvm_bench/; python3 profiling_wrapper.py -g -a'" &

        log "Waiting for job to finish or VM to reboot..."
        
        sleep 600

        while true; do
            if ! nc -z localhost $VM_PORT; then
                log "VM reboot detected. Waiting for it to come back up..."
                break
            fi

            # Check if the profiling script is still running in the VM
            if $SSH_CMD "pgrep -f profiling_wrapper.py" > /dev/null 2>&1; then
                log "Job running"
            else
                log "Job process not found, need relaunch"
                break
            fi

            sleep 30
        done

    else
        log "Waiting for VM SSH port to open..."
        sleep $CHECK_INTERVAL_BOOT
    fi
done

