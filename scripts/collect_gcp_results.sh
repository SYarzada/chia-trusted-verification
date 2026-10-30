#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"

# Default to the Gemini 3.5 Flash results. For another
# experiment, set REMOTE_RESULTS directly or use MODEL_SLUG.
if [ -n "${REMOTE_RESULTS:-}" ]; then
  REMOTE_RESULTS="${REMOTE_RESULTS}"
elif [ -n "${MODEL_SLUG:-}" ]; then
  REMOTE_RESULTS="runs/teaal_chia_${MODEL_SLUG}"
else
  REMOTE_RESULTS="runs/teaal_chia_challenge_gemini35flash"
fi
LOCAL_ROOT="${LOCAL_ROOT:-$REMOTE_RESULTS}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi

mkdir -p "$LOCAL_ROOT"
for ((i=0; i<NUM_VMS; i++)); do
  vm="${VM_PREFIX}-${i}"
  echo "Collecting shard $i from $vm"
  tmp="/tmp/teaal-results-${i}"
  rm -rf "$tmp"
  mkdir -p "$tmp"
  gcloud compute scp --recurse \
    "$vm:$REMOTE_REPO/$REMOTE_RESULTS/." "$tmp/" \
    --quiet --project="$PROJECT" --zone="$ZONE"
  cp -a "$tmp/." "$LOCAL_ROOT/"
done

echo "Merged results into $LOCAL_ROOT"
