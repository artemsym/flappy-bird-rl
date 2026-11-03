#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [[ -f "$VENV_PY" ]]; then
    PYTHON="$VENV_PY"
elif [[ -f "$HOME/.local/share/flappy-py/bin/python" ]]; then
    PYTHON="$HOME/.local/share/flappy-py/bin/python"
else
    PYTHON="python3"
fi

# Если переданы аргументы для DQN (например --play, --train, --list), запускаем flappy_dqn.py
if [[ "$#" -gt 0 ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/flappy_dqn.py" "$@"
else
    # Без аргументов запускается интерактивная игра для человека
    exec "$PYTHON" "$SCRIPT_DIR/flappy_bird.py"
fi
