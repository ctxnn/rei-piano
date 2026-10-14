#!/bin/bash
# Pull the light parts of training runs from a node and rebuild the dashboard locally,
# rendering any new performance videos on this machine (fast with a real GPU/Metal).
#
#   bash scripts/sync_dashboard.sh piano-r1 [run-glob]
#
# Pulled per run: config.json, metrics.jsonl, evals/*.npz (~1 MB each). Not pulled:
# checkpoints (they stay on the node and in the storage bucket).
# Result: outputs/dashboard_site/ (open index.html via a local web server, or publish).
set -euo pipefail
NODE="$1"
GLOB="${2:-*}"
cd "$(dirname "$0")/.."
MIRROR=outputs/runs_mirror
mkdir -p "$MIRROR"

autor run "$NODE" -- bash -c "cd ~/runaway/runs && tar czf ~/runs_light.tgz $GLOB/config.json $GLOB/metrics.jsonl $GLOB/evals 2>/dev/null; ls -la ~/runs_light.tgz | awk '{print \$5\" bytes\"}'"
autor cp "$NODE:~/runs_light.tgz" "$MIRROR/runs_light.tgz"
tar xzf "$MIRROR/runs_light.tgz" -C "$MIRROR"
rm "$MIRROR/runs_light.tgz"

.venv/bin/python -m runaway.dashboard.build --runs "$MIRROR" --out outputs/dashboard_site \
  --video-size 480x270 --max-videos-per-run "${MAX_VIDEOS:-6}" 2>&1 | grep -E "dashboard|wrote|skipping" || true
