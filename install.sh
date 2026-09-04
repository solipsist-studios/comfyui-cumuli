#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
#
# Install this pack's dependencies into ComfyUI's own environment.
#
#   ./install.sh                          # everything, into the detected interpreter
#   ./install.sh --dry-run                # print the commands, change nothing
#   ./install.sh --groups sfm             # just the rig solve
#   ./install.sh --python ~/miniconda3/envs/comfyenv/bin/python
#
# All this wrapper does is find ComfyUI's python; scripts/install.py does the work
# and takes the flags. Getting the interpreter right is the point: every package
# goes into that environment, so the wrong one installs a working set of
# dependencies somewhere ComfyUI will never look.

set -euo pipefail

# Resolve through the symlink in custom_nodes/ to the real checkout.
SCRIPT="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT" ]; do
    DIR="$(cd -P "$(dirname "$SCRIPT")" && pwd)"
    SCRIPT="$(readlink "$SCRIPT")"
    [[ "$SCRIPT" != /* ]] && SCRIPT="$DIR/$SCRIPT"
done
ROOT="$(cd -P "$(dirname "$SCRIPT")" && pwd)"

# --python is consumed here; everything else is forwarded untouched.
PYTHON=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            PYTHON="${2:-}"
            if [ -z "$PYTHON" ]; then
                echo "install.sh: --python needs a path" >&2
                exit 2
            fi
            shift 2
            ;;
        --python=*)
            PYTHON="${1#*=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Preference order, most explicit first. An activated venv or conda env is
# trusted over bare `python3`, which on most machines is the system interpreter.
if [ -z "$PYTHON" ]; then
    if [ -n "${COMFYUI_PYTHON:-}" ]; then
        PYTHON="$COMFYUI_PYTHON"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PYTHON="$CONDA_PREFIX/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    fi
fi

if [ -z "$PYTHON" ] || [ ! -x "$(command -v "$PYTHON" 2>/dev/null || echo "$PYTHON")" ]; then
    cat >&2 <<'EOF'
install.sh: could not find a python interpreter.

Point it at ComfyUI's own interpreter, which is the only one that matters here:

    ./install.sh --python /path/to/ComfyUI/venv/bin/python
    ./install.sh --python ~/miniconda3/envs/comfyenv/bin/python

or export COMFYUI_PYTHON, or activate the environment first.
EOF
    exit 1
fi

exec "$PYTHON" "$ROOT/scripts/install.py" ${ARGS[@]+"${ARGS[@]}"}
