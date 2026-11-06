#!/usr/bin/env bash
set -euo pipefail

adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout="$(cd "$adapter_dir/../.." && pwd)"
workspace="$(dirname "$checkout")"
default_root="$workspace"
if [[ ! -f "$workspace/src/eval_client/main.py" ]]; then
    default_root="$workspace/RoboDojo-eval"
fi
export ROBODOJO_ROOT="${ROBODOJO_ROOT:-$default_root}"
if [[ -d "$ROBODOJO_ROOT" ]]; then
    ROBODOJO_ROOT="$(cd "$ROBODOJO_ROOT" && pwd)"
fi
source "$adapter_dir/runtime_env.sh"
justrua_activate_env "${8:?evaluation environment required}" "$ROBODOJO_ROOT"
# RoboDojo imports client_server as a top-level package. Keep its own utils
# namespace ahead of the checkout's utils once the native launcher changes cwd.
export PYTHONPATH="$workspace:$ROBODOJO_ROOT:$checkout${PYTHONPATH:+:$PYTHONPATH}"
# The robot client connects directly to the selected policy server even if its
# Python environment has a proxy configured for internet access.
export no_proxy="${no_proxy:-${NO_PROXY:-}},${11:-localhost}"
export NO_PROXY="$no_proxy"
cd "$checkout"

case "${EVAL_ENV_TYPE:-sim}" in
    debug)
        exec "$JUSTRUA_PYTHON_BIN" "$checkout/utils/debug_env_client.py" \
            --bench_name "$1" --task_name "$2" --env_cfg_type "$4" --policy_name JustRUA \
            --protocol ws --host "${11:-localhost}" --port "${10}" --eval_batch false --eval_episode_num 1
        ;;
    sim)
        main="$ROBODOJO_ROOT/src/eval_client/main.py"
        if [[ ! -f "$main" ]]; then
            echo "Point ROBODOJO_ROOT to the RoboDojo evaluation checkout." >&2
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES="$7"
        # CUDA renumbers the selected physical GPU as process-local device 0.
        # Isaac Lab uses --device for Kit/physics and RoboDojo logs --device_id.
        if [[ -e "$ROBODOJO_ROOT/.cuda-native/libcuda.so.1" ]]; then
            export LD_LIBRARY_PATH="$ROBODOJO_ROOT/.cuda-native${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        # Run the native client through the observer launcher so contact sensing
        # is in place when the first environment resets.
        exec "$JUSTRUA_PYTHON_BIN" -u -m XPolicyLab.policy.JustRUA.observations.launch "$main" \
            --task_name "$2" --env_cfg_type "$4" --num_envs 1 --enable_cameras --headless \
            --kit_args " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera" \
            --device cuda:0 --device_id 0 --policy_name JustRUA --host "${11:-localhost}" --port "${10}" \
            --policy_server_url "ws://${11:-localhost}:${10}" --protocol ws --additional_info "$9" --seed "$6"
        ;;
    *)
        echo "Supported EVAL_ENV_TYPE values are sim and debug." >&2
        exit 2
        ;;
esac
