#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"
CONFIG="${CONFIG:-configs/teaal_experiments.yaml}"
SKIP_CERTIFICATION_REPLAY="${SKIP_CERTIFICATION_REPLAY:-0}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi

for ((i=0; i<NUM_VMS; i++)); do
  vm="${VM_PREFIX}-${i}"
  echo "[$i/$NUM_VMS] launching $vm (skip_replay=$SKIP_CERTIFICATION_REPLAY)"
  gcloud compute ssh "$vm" --project="$PROJECT" --zone="$ZONE" --command \
    "cd '$REMOTE_REPO' && SKIP_CERTIFICATION_REPLAY='$SKIP_CERTIFICATION_REPLAY' setsid -f bash scripts/run_experiment_shard.sh '$NUM_VMS' '$i' '$CONFIG' > 'experiment-shard-$i.log' 2>&1 < /dev/null"
done

echo "Launched $NUM_VMS shards."
echo "Monitor search with: PHASE=search bash scripts/monitor_gcp_experiments.sh"
