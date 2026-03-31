#!/bin/bash
# ============================================================
# Parallel Data Collection Launcher (3-Camera, 15Hz)
# ============================================================
# Spawns multiple headless Isaac Sim workers, each running
# strawberry_pick_vla_collect.py with separate output dirs.
#
# Usage:
#   export ISAAC_SIM_DIR=/path/to/isaac-sim
#   cd data_collection
#   bash parallel_collect.sh 2
#
# Stop all workers:
#   kill $(cat /tmp/vla_worker_*.pid)
# ============================================================

set -e

NUM_WORKERS=${1:-2}
ROUNDS_PER_WORKER=50

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ISAAC_PYTHON="${ISAAC_SIM_DIR:-/path/to/isaac-sim}/python.sh"
BASE_EPISODE_DIR="${SCRIPT_DIR}/episodes"

# ── Source files ──
COLLECT_SCRIPT="${SCRIPT_DIR}/strawberry_pick_vla_collect.py"
COLLECTOR_MODULE="${SCRIPT_DIR}/vla_data_collector.py"

echo "============================================"
echo "Parallel VLA Data Collection (3-Camera, 15Hz)"
echo "  Workers:          $NUM_WORKERS"
echo "  Rounds per worker: $ROUNDS_PER_WORKER"
echo "  Total rounds:     $((NUM_WORKERS * ROUNDS_PER_WORKER))"
echo "  Source script:    $(basename $COLLECT_SCRIPT)"
echo "  Isaac Sim:        $ISAAC_PYTHON"
echo "============================================"

# ── Pre-flight checks ──
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "ERROR: Isaac Sim python not found at $ISAAC_PYTHON"
    echo "  Set ISAAC_SIM_DIR before running this script."
    exit 1
fi

if [ ! -f "$COLLECT_SCRIPT" ]; then
    echo "ERROR: $COLLECT_SCRIPT not found!"
    exit 1
fi

if [ ! -f "$COLLECTOR_MODULE" ]; then
    echo "ERROR: $COLLECTOR_MODULE not found!"
    exit 1
fi

# ── Create per-worker output directories ──
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    WORKER_DIR="${BASE_EPISODE_DIR}_worker${i}"
    rm -rf "$WORKER_DIR"
    mkdir -p "$WORKER_DIR"
    echo "Worker $i → $WORKER_DIR"
done

# ── Generate and launch per-worker scripts ──
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    WORKER_DIR="${BASE_EPISODE_DIR}_worker${i}"
    LOG_FILE="${SCRIPT_DIR}/worker${i}.log"
    WORKER_SCRIPT="${SCRIPT_DIR}/worker${i}_collect.py"

    # Copy collection script for this worker
    cp "$COLLECT_SCRIPT" "$WORKER_SCRIPT"

    # Patch output directory
    sed -i "s|VLA_EPISODE_DIR = .*|VLA_EPISODE_DIR = \"${WORKER_DIR}\"|" "$WORKER_SCRIPT"

    # Patch number of rounds
    sed -i "s|NUM_ROUNDS = .*|NUM_ROUNDS = ${ROUNDS_PER_WORKER}|" "$WORKER_SCRIPT"

    # Patch CSV log path (avoid conflicts between workers)
    sed -i "s|picking_log.csv|picking_log_worker${i}.csv|" "$WORKER_SCRIPT"

    # Patch fixed scene path (each worker needs its own copy)
    sed -i "s|scene_assembled_fixed_vla.usd|scene_assembled_fixed_worker${i}.usd|" "$WORKER_SCRIPT"

    # Launch
    echo "Starting worker $i → $LOG_FILE"
    nohup "$ISAAC_PYTHON" "$WORKER_SCRIPT" > "$LOG_FILE" 2>&1 &
    WORKER_PID=$!
    echo $WORKER_PID > "/tmp/vla_worker_${i}.pid"
    echo "  PID: $WORKER_PID"

    # Stagger launches to avoid GPU resource contention
    sleep 10
done

echo ""
echo "============================================"
echo "All workers started!"
echo ""
echo "Monitor progress:"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "  tail -f ${SCRIPT_DIR}/worker${i}.log"
done
echo ""
echo "Check running workers:"
echo "  ps aux | grep worker.*_collect.py"
echo ""
echo "Stop all workers:"
echo '  kill $(cat /tmp/vla_worker_*.pid)'
echo ""
echo "After all workers finish, merge:"
echo "  python3 merge_episodes.py --base_dir ${SCRIPT_DIR} --output_dir ${BASE_EPISODE_DIR}"
echo "============================================"
