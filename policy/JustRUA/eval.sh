#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 10 ]]; then
    echo "Usage: eval.sh BENCH TASK CKPT ROBOT ACTION SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV" >&2
    exit 2
fi
adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout="$(cd "$adapter_dir/../.." && pwd)"
if [[ "$1" != RoboDojo || "$4" != arx_x5 || "$5" != joint ]]; then
    echo "JustRUA requires bench=RoboDojo, env_cfg_type=arx_x5, action_type=joint." >&2
    exit 2
fi

port="$(bash "$checkout/utils/get_free_port.sh")"
host=localhost
bash "$adapter_dir/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "$port" "$host" &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bash "$checkout/utils/wait_for_policy_server.sh" "$host" "$port" "$server_pid" "Policy server" 120
bash "$adapter_dir/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
    "ckpt_name=$3,action_type=$5" "$port" "$host"
echo "[MAIN] eval finished"
