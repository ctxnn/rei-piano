#!/bin/bash
# Download the Unitree G1 model (and Shadow Hand) from MuJoCo Menagerie into
# third_party/mujoco_menagerie (sparse checkout: ~40 MB instead of the whole repo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone -q --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1 shadow_hand
echo "G1 model at third_party/mujoco_menagerie/unitree_g1"
