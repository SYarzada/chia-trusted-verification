#!/usr/bin/env bash
set -euo pipefail

NUM_SHARDS="${1:?usage: scripts/run_experiment_shard.sh NUM_SHARDS SHARD_INDEX [CONFIG]}"
SHARD_INDEX="${2:?usage: scripts/run_experiment_shard.sh NUM_SHARDS SHARD_INDEX [CONFIG]}"
CONFIG="${3:-configs/teaal_experiments.yaml}"
SKIP_CERTIFICATION_REPLAY="${SKIP_CERTIFICATION_REPLAY:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

EXPERIMENT_ENV="${CHIA_EXPERIMENT_ENV:-${CHIA_CAMPAIGN_ENV:-$REPO/experiment.env}}"
if [ -f "$EXPERIMENT_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$EXPERIMENT_ENV"
  set +a
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate chia_env

export CHIA_PROJECT_ROOT="$REPO"
export CHIA_HEAD_IP="${CHIA_HEAD_IP:-$(hostname -I | awk '{print $1}')}"
if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
  export GOOGLE_CLOUD_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
export ALPHAEVOLVE_PROJECT_ID="${ALPHAEVOLVE_PROJECT_ID:-$GOOGLE_CLOUD_PROJECT}"
export ALPHAEVOLVE_LOCATION="${ALPHAEVOLVE_LOCATION:-global}"
export ALPHAEVOLVE_COLLECTION="${ALPHAEVOLVE_COLLECTION:-default_collection}"
export ALPHAEVOLVE_ASSISTANT="${ALPHAEVOLVE_ASSISTANT:-default_assistant}"

if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
  echo "GOOGLE_CLOUD_PROJECT is not set and gcloud has no active project." >&2
  exit 2
fi

cleanup_cluster() {
  timeout 90s chia down cluster/local_gemini.yaml >/dev/null 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
  docker ps -aq --filter "name=chia-trusted-project-${USER}-0" \
    | xargs -r docker rm -f >/dev/null 2>&1 || true
  docker ps -aq --filter "name=chia-ctv-opencode-${USER}-0" \
    | xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup_cluster EXIT

printf 'Starting local CHIA cluster on %s for shard %s/%s\n' "$CHIA_HEAD_IP" "$SHARD_INDEX" "$NUM_SHARDS"
cleanup_cluster
printf 'y\n' | chia up cluster/local_gemini.yaml
export RAY_ADDRESS="${CHIA_HEAD_IP}:6379"
export PYTHONUNBUFFERED=1

args=(
  --config "$CONFIG"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --resume
)
if [ "$SKIP_CERTIFICATION_REPLAY" = "1" ]; then
  args+=(--skip-certification-replay)
  echo "Certification replay: DEFERRED (search-only phase)"
else
  echo "Certification replay: ENABLED (after each search run)"
fi

python -u experiments/run_teaal_experiments.py "${args[@]}"
