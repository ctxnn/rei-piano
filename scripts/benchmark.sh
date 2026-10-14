#!/bin/bash
# Measure training throughput (env steps/s) for a few settings on this machine.
# Each run is short: 5k random warm-up steps, then 45k steps of real training.
# Results land in runs/bench_*/metrics.jsonl, including time_ms/* per loop phase.
#
#   bash scripts/benchmark.sh                                  # default settings
#   BENCH_CONFIGS="13 0.5 256,13 1.0 1024" bash scripts/benchmark.sh
#
# Each config is "<num_envs> <updates_per_env_step> <batch_size>"; configs are
# separated by commas.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
IFS=',' read -ra CONFIGS <<< "${BENCH_CONFIGS:-13 0.5 256,13 1.0 256,13 0.25 1024,13 1.0 1024}"
for cfg in "${CONFIGS[@]}"; do
  read -r n utd batch <<< "$cfg"
  name="bench_n${n}_utd${utd}_b${batch}"
  .venv/bin/python -m runaway.train --run-name "$name" --num-envs "$n" \
    --updates-per-env-step "$utd" --sac.batch-size "$batch" \
    --total-env-steps 50000 --eval-every 1000000 --log-every 5000 2>&1 \
    | grep -E "event=train|event=eval|Error|error" | tail -4 | sed "s/^/[$name] /"
done
