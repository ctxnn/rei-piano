#!/bin/bash
# Copy this project's code (not data/outputs/runs) to an Autoresearch node.
#
#   bash scripts/push_code.sh <node-name>
#
# The code lands in ~/runaway on the node's persistent disk. Existing runs/ and the
# .venv there are left alone, so this is safe to repeat after every code change.
set -euo pipefail
NODE="$1"
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# COPYFILE_DISABLE + --no-xattrs: keep macOS metadata out of the archive.
# Third-party models are not sent: node_setup.sh downloads them (scripts/get_g1.sh).
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$TMP/code.tgz" pyproject.toml README.md src scripts \
  tests sweeps data/midi
autor cp "$TMP/code.tgz" "$NODE:~/code.tgz"
autor run "$NODE" -- sh -c 'mkdir -p ~/runaway && tar xzf ~/code.tgz -C ~/runaway && echo "code updated in ~/runaway"'
