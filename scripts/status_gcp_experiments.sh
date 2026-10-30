#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"
LINES="${LINES:-12}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi

for ((i=0; i<NUM_VMS; i++)); do
  echo
  echo "================ SHARD $i ================"
  gcloud compute ssh "${VM_PREFIX}-${i}" --project="$PROJECT" --zone="$ZONE" --command \
    "tail -n '$LINES' '$REMOTE_REPO/experiment-shard-$i.log' 2>/dev/null || echo 'log not created yet'" || true
done
