#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
ZONE="${ZONE:-us-central1-a}"
NUM_VMS="${NUM_VMS:-8}"
VM_PREFIX="${VM_PREFIX:-teaal-exp}"
MACHINE_TYPE="${MACHINE_TYPE:-c4-standard-8}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
ALPHAEVOLVE_ENGINE_ID="${ALPHAEVOLVE_ENGINE_ID:-alpha-evolve-experiment-engine}"
REMOTE_REPO="${REMOTE_REPO:-/home/$USER/chia-trusted-verification}"
ADC="$HOME/.config/gcloud/application_default_credentials.json"
CHIA_REPO="${CHIA_REPO:-$(dirname "$REPO")/chia}"

if [ -z "$PROJECT" ]; then
  echo "Set PROJECT (or GOOGLE_CLOUD_PROJECT) first." >&2
  exit 2
fi
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not on PATH." >&2
  exit 2
fi
if [ ! -r "$ADC" ]; then
  echo "ADC file not found at $ADC. Complete gcloud application-default authentication first." >&2
  exit 2
fi

cat > "$REPO/experiment.env" <<EOF
GOOGLE_CLOUD_PROJECT=$PROJECT
GOOGLE_CLOUD_LOCATION=global
VERTEX_LOCATION=global
ALPHAEVOLVE_PROJECT_ID=$PROJECT
ALPHAEVOLVE_ENGINE_ID=$ALPHAEVOLVE_ENGINE_ID
ALPHAEVOLVE_LOCATION=global
ALPHAEVOLVE_COLLECTION=default_collection
ALPHAEVOLVE_ASSISTANT=default_assistant
EOF
chmod 600 "$REPO/experiment.env"

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable compute.googleapis.com aiplatform.googleapis.com discoveryengine.googleapis.com \
  --project="$PROJECT"

NAMES=()
MISSING=()
for ((i=0; i<NUM_VMS; i++)); do
  name="${VM_PREFIX}-${i}"
  NAMES+=("$name")
  if ! gcloud compute instances describe "$name" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
    MISSING+=("$name")
  fi
done

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "Creating ${#MISSING[@]} VM(s): ${MISSING[*]}"
  gcloud compute instances create "${MISSING[@]}" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --scopes=cloud-platform
else
  echo "All $NUM_VMS experiment VMs already exist."
fi

echo "Packaging experiment repository..."
tar --exclude='./runs' --exclude='./results' --exclude='./.git' --exclude='*/__pycache__' \
  -czf /tmp/ctv-experiments.tgz -C "$REPO" .

HAVE_CHIA_TARBALL=0
if [ -d "$CHIA_REPO" ]; then
  echo "Packaging existing CHIA source from $CHIA_REPO"
  tar --exclude='./.git' --exclude='*/__pycache__' -czf /tmp/chia-source.tgz -C "$CHIA_REPO" .
  HAVE_CHIA_TARBALL=1
else
  echo "No sibling CHIA source found; VMs will clone CHIA from GitHub."
fi

for name in "${NAMES[@]}"; do
  echo "===== staging $name ====="
  files=(/tmp/ctv-experiments.tgz "$REPO/scripts/gcp_vm_setup.sh" "$ADC")
  if [ "$HAVE_CHIA_TARBALL" -eq 1 ]; then
    files+=(/tmp/chia-source.tgz)
  fi

  # Fresh VMs can take a short time before SSH is ready.
  success=0
  for attempt in {1..12}; do
    if gcloud compute scp "${files[@]}" "$name:/tmp/" \
      --project="$PROJECT" --zone="$ZONE"; then
      success=1
      break
    fi
    echo "SSH not ready on $name yet (attempt $attempt/12); retrying in 10 s..."
    sleep 10
  done
  if [ "$success" -ne 1 ]; then
    echo "Failed to copy setup files to $name" >&2
    exit 1
  fi
done

echo "Installing all VMs in parallel..."
pids=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  (
    gcloud compute ssh "$name" --project="$PROJECT" --zone="$ZONE" \
      --command="CTV_REMOTE_REPO='$REMOTE_REPO' bash /tmp/gcp_vm_setup.sh" \
      > "/tmp/teaal-setup-${i}.log" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Setup failed on ${NAMES[$i]}; see /tmp/teaal-setup-${i}.log" >&2
    failed=1
  fi
done

for i in "${!NAMES[@]}"; do
  echo "===== ${NAMES[$i]} ====="
  tail -n 8 "/tmp/teaal-setup-${i}.log" || true
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

echo
printf 'Provisioning complete. Remote repo: %s\n' "$REMOTE_REPO"
echo "Launch all shards with: bash scripts/launch_gcp_experiments.sh"
