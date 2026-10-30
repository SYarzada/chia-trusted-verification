#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi

ARCHIVE="/tmp/ctv-repo-update-$$.tgz"
trap 'rm -f "$ARCHIVE"' EXIT

tar \
  --exclude='./runs' \
  --exclude='./results' \
  --exclude='./.git' \
  --exclude='./experiment.env' \
  --exclude='*/__pycache__' \
  -czf "$ARCHIVE" -C "$REPO" .

for ((i=0; i<NUM_VMS; i++)); do
  vm="${VM_PREFIX}-${i}"
  echo "===== syncing $vm ====="
  gcloud compute scp "$ARCHIVE" "$vm:/tmp/ctv-repo-update.tgz" \
    --quiet --project="$PROJECT" --zone="$ZONE"
  gcloud compute ssh "$vm" \
    --quiet --project="$PROJECT" --zone="$ZONE" \
    --command="mkdir -p '$REMOTE_REPO' && tar -xzf /tmp/ctv-repo-update.tgz -C '$REMOTE_REPO' && rm -f /tmp/ctv-repo-update.tgz && source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate chia_env && cd '$REMOTE_REPO' && bash -n scripts/run_experiment_shard.sh && python -m compileall -q src experiments analysis && echo UPDATED"
done
