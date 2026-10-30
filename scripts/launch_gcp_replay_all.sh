#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"
REMOTE_RESULTS="${REMOTE_RESULTS:-runs/teaal_chia_challenge_gemini35flash}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi

for ((i=0; i<NUM_VMS; i++)); do
  vm="${VM_PREFIX}-${i}"
  force_arg=""
  if [ "$FORCE_REPLAY" = "1" ]; then
    force_arg="--force"
  fi
  echo "[$i/$NUM_VMS] launching certification replay on $vm"
  gcloud compute ssh "$vm" --project="$PROJECT" --zone="$ZONE" --command \
    "cd '$REMOTE_REPO' && setsid -f bash scripts/run_certification_replay.sh '$REMOTE_RESULTS' '$force_arg' > 'certification-replay-shard-$i.log' 2>&1 < /dev/null"
done

echo "Launched replay on $NUM_VMS VMs."
echo "Monitor with: PHASE=replay bash scripts/monitor_gcp_experiments.sh"
