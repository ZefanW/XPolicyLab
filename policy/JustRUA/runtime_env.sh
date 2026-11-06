# Select an existing Python environment in this shell. Activation also selects
# the python3 executable used by RoboDojo's own evaluation script.
justrua_activate_env() {
    local requested=$1
    local project=$2
    if [[ "$requested" == uv ]]; then
        requested="$project/.venv"
    fi
    if [[ -d "$requested/.venv" ]]; then
        requested="$requested/.venv"
    fi
    if [[ -f "$requested/bin/activate" ]]; then
        requested="$(cd "$requested" && pwd)"
        source "$requested/bin/activate"
        JUSTRUA_PYTHON_BIN="$requested/bin/python"
    elif [[ -f "$requested" && -x "$requested" ]]; then
        JUSTRUA_PYTHON_BIN="$(cd "$(dirname "$requested")" && pwd)/$(basename "$requested")"
        export PATH="$(dirname "$JUSTRUA_PYTHON_BIN"):$PATH"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$requested"
        JUSTRUA_PYTHON_BIN="$(command -v python)"
    else
        echo "Python environment unavailable: $requested" >&2
        return 1
    fi
}
