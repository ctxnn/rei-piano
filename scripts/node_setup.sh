#!/bin/bash
# Set up (or re-set up after a wake) an Autoresearch H100 node for training.
#
# Run on the node from the project directory:   bash scripts/node_setup.sh
#
# Node image: jax-0.11-cuda12.9 (Ubuntu 24.04, CUDA 12.9, uv preinstalled).
# Only /home/dev survives a node stop, so:
#   * system packages (apt) are reinstalled on every wake (fast, ~20 s)
#   * the Python venv lives in the project dir on the persistent disk (kept)
set -euo pipefail
cd "$(dirname "$0")/.."

# System libraries:
#   libfluidsynth3  - audio synthesis (RoboPianist imports it at load time; also used
#                     to give the dashboard videos sound)
#   portaudio19-dev - needed to build PyAudio, a robopianist dependency
#   libosmesa6      - software OpenGL, so MuJoCo can render videos without GPU graphics
#                     (use MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa)
#   ffmpeg          - video encoding
sudo apt-get update -qq
sudo apt-get install -y -qq libfluidsynth3 portaudio19-dev libosmesa6 ffmpeg > /dev/null

# The Unitree G1 humanoid model (MuJoCo Menagerie), for --env.embodiment g1.
bash scripts/get_g1.sh

if [ ! -d .venv ]; then
  uv venv -q -p 3.12 .venv
fi
VIRTUAL_ENV=.venv uv pip install -q -e ".[cuda]" soundfile

.venv/bin/python - <<'EOF'
import os, jax
print("jax", jax.__version__, "devices:", jax.devices())
print("CPU budget (GMN_CPU_LIMIT):", os.environ.get("GMN_CPU_LIMIT"), " os.cpu_count():", os.cpu_count())
EOF
