#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"
REMOTE_RESULTS="${REMOTE_RESULTS:-runs/teaal_chia_challenge_gemini35flash}"
EXPECTED_PER_SHARD="${EXPECTED_PER_SHARD:-40}"
PHASE="${PHASE:-search}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi
if [[ "$PHASE" != "search" && "$PHASE" != "replay" && "$PHASE" != "all" ]]; then
  echo "PHASE must be search, replay, or all" >&2
  exit 2
fi

for ((i=0; i<NUM_VMS; i++)); do
  vm="${VM_PREFIX}-${i}"
  printf 'shard %d: ' "$i"
  gcloud compute ssh "$vm" --quiet --project="$PROJECT" --zone="$ZONE" --command="
    cd '$REMOTE_REPO'
    ROOT='$REMOTE_RESULTS'
    completed=\$(grep -rl '\"status\": \"completed\"' \"\$ROOT\" --include=run_manifest.json 2>/dev/null | wc -l)
    errors=\$(grep -rl '\"status\": \"error\"' \"\$ROOT\" --include=run_manifest.json 2>/dev/null | wc -l)
    search=\$(find \"\$ROOT\" -name search_result.json 2>/dev/null | wc -l)
    replay=\$(find \"\$ROOT\" -name certification_replay.jsonl -size +0c 2>/dev/null | wc -l)
    if pgrep -f '[r]un_teaal_experiments.py|[r]un_experiment_shard.sh' >/dev/null; then search_state=RUNNING; else search_state=STOPPED; fi
    if pgrep -f '[r]eplay_certification_backends.py|[r]un_certification_replay.sh' >/dev/null; then replay_state=RUNNING; else replay_state=STOPPED; fi
    case '$PHASE' in
      search)
        printf 'completed=%s/$EXPECTED_PER_SHARD  search=%s/$EXPECTED_PER_SHARD  errors=%s  SEARCH_%s\\n' \"\$completed\" \"\$search\" \"\$errors\" \"\$search_state\"
        ;;
      replay)
        printf 'completed=%s/$EXPECTED_PER_SHARD  search=%s/$EXPECTED_PER_SHARD  replay=%s/$EXPECTED_PER_SHARD  errors=%s  REPLAY_%s\\n' \"\$completed\" \"\$search\" \"\$replay\" \"\$errors\" \"\$replay_state\"
        ;;
      all)
        printf 'completed=%s/$EXPECTED_PER_SHARD  search=%s/$EXPECTED_PER_SHARD  replay=%s/$EXPECTED_PER_SHARD  errors=%s  SEARCH_%s REPLAY_%s\\n' \"\$completed\" \"\$search\" \"\$replay\" \"\$errors\" \"\$search_state\" \"\$replay_state\"
        ;;
    esac
  "
done
