#!/usr/bin/env bash
set -euo pipefail

adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout="$(cd "$adapter_dir/../.." && pwd)"
workspace="$(dirname "$checkout")"
source "$adapter_dir/runtime_env.sh"
justrua_activate_env "${8:?policy environment required}" "$checkout"
export PYTHONPATH="$workspace${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$7"
cd "$checkout"
exec "$JUSTRUA_PYTHON_BIN" "$checkout/setup_policy_server.py" --config_path "$adapter_dir/deploy.yml" \
    --overrides "bench_name=$1" "task_name=$2" "ckpt_name=$3" "env_cfg_type=$4" \
    "action_type=$5" "seed=$6" "port=$9" "host=${10:-localhost}"
