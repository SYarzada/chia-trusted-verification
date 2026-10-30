#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${1:?usage: scripts/run_certification_replay.sh RUNS_ROOT [--force]}"
FORCE_ARG="${2:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate chia_env
export PYTHONUNBUFFERED=1

args=(--runs "$RUNS_ROOT")
if [ "$FORCE_ARG" = "--force" ]; then
  args+=(--force)
elif [ -n "$FORCE_ARG" ]; then
  echo "Unknown second argument: $FORCE_ARG (expected --force)" >&2
  exit 2
fi

python -u analysis/replay_certification_backends.py "${args[@]}"
