#!/bin/bash
# ============================================================
# Worker Monitor — shows status of parallel data collection
# ============================================================
# Usage: bash monitor.sh
# Exit:  Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_WORKERS=${1:-2}

while true; do
    clear
    echo "============================================"
    echo "  VLA Data Collection Monitor  $(date '+%H:%M:%S')"
    echo "============================================"
    echo ""

    # GPU status (single-line summary)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total,gpu-util --format=csv,noheader,nounits 2>/dev/null)
    echo "GPU: ${GPU_MEM} (used MiB, total MiB, util%)"
    echo ""

    # Per-worker status
    TOTAL_EP=0
    TOTAL_ALIVE=0
    printf "%-10s %-8s %-10s %-12s %s\n" "Worker" "Status" "Episodes" "Last Round" "Latest Log"
    echo "------------------------------------------------------------"

    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        WORKER_DIR="${SCRIPT_DIR}/episodes_worker${i}"
        LOG_FILE="${SCRIPT_DIR}/worker${i}.log"
        PID_FILE="/tmp/vla_worker_${i}.pid"

        # Check if process is alive
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            STATUS="RUNNING"
            TOTAL_ALIVE=$((TOTAL_ALIVE + 1))
        else
            if grep -q "GLOBAL SUMMARY" "$LOG_FILE" 2>/dev/null; then
                STATUS="DONE"
            else
                STATUS="DEAD"
            fi
        fi

        # Count episodes
        if [ -d "$WORKER_DIR" ]; then
            N_EP=$(ls -d "$WORKER_DIR"/episode_* 2>/dev/null | wc -l)
        else
            N_EP=0
        fi
        TOTAL_EP=$((TOTAL_EP + N_EP))

        # Latest round info
        LAST_ROUND=$(grep "ROUND" "$LOG_FILE" 2>/dev/null | tail -1 | head -c 60)

        # Latest log line (non-empty)
        LAST_LOG=$(grep -v "^$" "$LOG_FILE" 2>/dev/null | tail -1 | head -c 50)

        printf "%-10s %-8s %-10s %-12s %s\n" "W${i}" "$STATUS" "$N_EP" "$LAST_ROUND" "$LAST_LOG"
    done

    echo ""
    echo "Total: ${TOTAL_EP} episodes, ${TOTAL_ALIVE}/${NUM_WORKERS} workers alive"

    # Check for discarded episodes
    TOTAL_DISCARD=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        LOG_FILE="${SCRIPT_DIR}/worker${i}.log"
        N_DISCARD=$(grep -c "DISCARD" "$LOG_FILE" 2>/dev/null || echo 0)
        TOTAL_DISCARD=$((TOTAL_DISCARD + N_DISCARD))
    done
    if [ "$TOTAL_DISCARD" -gt 0 ]; then
        echo "Discarded: ${TOTAL_DISCARD} episodes (anomaly filtered)"
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 15
done
