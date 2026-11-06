#!/usr/bin/env bash
set -euo pipefail

adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout="$(cd "$adapter_dir/../.." && pwd)"
python_bin="${1:-python}"
"$python_bin" -m pip install -e "$checkout" Pillow
