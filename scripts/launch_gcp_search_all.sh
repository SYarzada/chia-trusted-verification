#!/usr/bin/env bash
set -euo pipefail
export SKIP_CERTIFICATION_REPLAY=1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_gcp_experiments.sh"
