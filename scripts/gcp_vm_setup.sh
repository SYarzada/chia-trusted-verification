
#!/usr/bin/env bash
set -euo pipefail

CTV="${CTV_REMOTE_REPO:-$HOME/chia-trusted-verification}"
CHIA="${CHIA_REMOTE_REPO:-$HOME/chia}"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  docker.io openssh-client openssh-server git curl ca-certificates build-essential
sudo systemctl enable --now docker
sudo systemctl enable --now ssh
sudo usermod -aG docker "$USER"

mkdir -p "$CTV"
tar -xzf /tmp/ctv-experiments.tgz -C "$CTV"

if [ -f /tmp/chia-source.tgz ]; then
  mkdir -p "$CHIA"
  tar -xzf /tmp/chia-source.tgz -C "$CHIA"
elif [ ! -d "$CHIA/.git" ]; then
  git clone https://github.com/ucb-bar/chia.git "$CHIA"
fi

mkdir -p "$HOME/.config/gcloud"
install -m 600 /tmp/application_default_credentials.json \
  "$HOME/.config/gcloud/application_default_credentials.json"

if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  curl -fsSLo /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx chia_env; then
  conda create -n chia_env --override-channels -c conda-forge python=3.10.19 git -y
fi
conda activate chia_env

# Use HTTPS for GitHub dependencies without SSH keys.
git config --global url."https://github.com/".insteadOf "git@github.com:"
git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/"

# Load Conda in non-interactive shells.
touch "$HOME/.bashrc"
if ! head -20 "$HOME/.bashrc" | grep -q 'miniconda3/etc/profile.d/conda.sh'; then
  {
    echo 'source "$HOME/miniconda3/etc/profile.d/conda.sh"'
    cat "$HOME/.bashrc"
  } > "$HOME/.bashrc.new"
  mv "$HOME/.bashrc.new" "$HOME/.bashrc"
fi

python -m pip install --upgrade pip
if ! python -m pip install -e "$CHIA"; then
  echo "Packaged CHIA source did not install cleanly; cloning fresh CHIA checkout."
  rm -rf "$CHIA"
  git clone https://github.com/ucb-bar/chia.git "$CHIA"
  python -m pip install -e "$CHIA"
fi

cd "$CTV"
conda install --override-channels -c conda-forge pandas=2.3.3 matplotlib z3-solver -y
python -m pip install -e ".[search,test,analysis]"
chmod 600 experiment.env

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  ssh-keygen -t ed25519 -N '' -f "$HOME/.ssh/id_ed25519"
fi
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"
PUBKEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
grep -qxF "$PUBKEY" "$HOME/.ssh/authorized_keys" || echo "$PUBKEY" >> "$HOME/.ssh/authorized_keys"
HOST_IP="$(hostname -I | awk '{print $1}')"
ssh-keyscan -H "$HOST_IP" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true

sudo docker pull ghcr.io/ucb-bar/chia-opencode:latest

# Check Docker access through a new SSH login.
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$USER@$HOST_IP" \
  'docker info >/dev/null && echo "Self-SSH and Docker OK"'

cd "$CTV"
python -m compileall -q src experiments analysis
python experiments/run_teaal_experiments.py --config configs/teaal_experiments.yaml --dry-run > /tmp/teaal-plan.txt
head -1 /tmp/teaal-plan.txt

echo "VM READY: $(hostname) ($HOST_IP)"
