#!/bin/bash
# Start several training runs on one node in parallel, one GPU each, plus the live
# dashboard. Run on the node from ~/runaway:
#
#   bash scripts/launch_runs.sh sweeps/round1.txt
#
# The sweep file has one run per line:  <run-name> <extra train.py arguments...>
#   r1_g088_s0  --sac.discount 0.88 --seed 0
#   r1_g092_s0  --sac.discount 0.92 --seed 0
# Blank lines and lines starting with # are ignored. Run i gets GPU i.
#
# Every run writes to runs/<name>/ and logs to runs/<name>/train.log. The dashboard
# rebuilds every 60 s and serves on port 8000 (expose it with `autor` from outside).
set -euo pipefail
cd "$(dirname "$0")/.."
SWEEP="$1"
NUM_GPUS=$(nvidia-smi -L | wc -l)
# Each run gets its own block of CPUs (taskset). Two reasons:
#  * JAX sizes its thread pools to the CPUs a process may use. Unpinned on a 224-CPU
#    node, one run made 940 threads and 8 runs hit the container's 4096 thread limit.
#  * The run's simulator workers inherit the block and stay off other runs' cores.
CPUS_PER_RUN=$(( $(nproc) / NUM_GPUS ))
COMMON_ARGS=${COMMON_ARGS:-}

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # several runs may share a GPU's memory

i=0
pids=()
while read -r name args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  gpu=$(( i % NUM_GPUS ))
  mkdir -p "runs/$name"
  cpus="$(( gpu * CPUS_PER_RUN ))-$(( gpu * CPUS_PER_RUN + CPUS_PER_RUN - 1 ))"
  echo "starting $name on GPU $gpu, CPUs $cpus: $args $COMMON_ARGS"
  # Append (>>) so a resumed run keeps its earlier log.
  # shellcheck disable=SC2086  # word-splitting of args is intended
  CUDA_VISIBLE_DEVICES=$gpu nohup taskset -c "$cpus" .venv/bin/python -m runaway.train --run-name "$name" \
    $args $COMMON_ARGS >> "runs/$name/train.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
done < "$SWEEP"

# Live dashboard on the node: curves and piano rolls (videos are rendered off-node,
# see scripts/sync_dashboard.sh, because software OpenGL takes ~6 min per video).
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa nohup nice -n 15 .venv/bin/python -m runaway.dashboard.build \
  --runs runs --out dashboard_site --watch --interval 60 --serve --port 8000 --no-videos \
  > dashboard.log 2>&1 &
dashboard_pid=$!

echo "launched $i runs + dashboard; waiting for training to finish"
wait "${pids[@]}" || true
# One last dashboard build so the final evaluations get their videos, then stop.
sleep 90
kill "$dashboard_pid" 2>/dev/null || true
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv/bin/python -m runaway.dashboard.build \
  --runs runs --out dashboard_site --no-videos >> dashboard.log 2>&1
echo "all runs finished"
