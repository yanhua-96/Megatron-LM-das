#!/bin/bash
# Run inside the training container; serves all ChatGLM2 runs on port 6006.
set -eo pipefail

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Example:
# TENSORBOARD_LOGDIR=/root/Megatron-LM-das/examples/llama2/tensorboard \
#   ./run_tensorboard.sh
if [[ -z "${TENSORBOARD_LOGDIR:-}" ]]; then
    echo "TENSORBOARD_LOGDIR must be set." >&2
    echo "Example: TENSORBOARD_LOGDIR=/path/to/tensorboard ./run_tensorboard.sh" >&2
    exit 1
fi

exec tensorboard \
    --logdir "${TENSORBOARD_LOGDIR}" \
    --host 0.0.0.0 \
    --port 6006 \
    --reload_interval 5 \
    "$@"
