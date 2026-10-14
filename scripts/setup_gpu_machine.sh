#!/bin/bash
# Set up a Linux GPU machine (Ubuntu + CUDA 12, with uv installed) for training.
#
# Run on the machine from the project directory:   bash scripts/setup_gpu_machine.sh
#
# Safe to re-run: system packages are (re)installed, the Python venv in .venv is kept.
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
print("os.cpu_count():", os.cpu_count())
EOF
