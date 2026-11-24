#!/bin/bash
# Launcher for ChatGLM2-6B training performance tests.
#   usage: bash run_chatglm2.sh <hostfile> <node_num>
#   e.g.   bash run_chatglm2.sh ./hostfile 2

set -eo pipefail

for para in "$@"
do
    if [[ $para == --profiling* ]]; then
        profiling=${para#*=}
    fi
done

# Those variables no need to modify
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
hostfile_input=${1:-${CURRENT_DIR}/hostfile}
node_num=${2:-1}

# Those variables need to modify
DTK_ENV=${DTK_ENV:-/opt/dtk/env.sh}
DATA_PATH=${DATA_PATH:-${CURRENT_DIR}/data/dataset/oscar-en-10k_text_document}
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-${CURRENT_DIR}/data/tokenizer/tokenizer.model}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${CURRENT_DIR}/tensorboard/$(date +%F-%H%M%S)}
LAUNCHER=${LAUNCHER:-torchrun}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_chatglm2_6B.sh}                       # script under this example directory to run
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

if [[ "${LAUNCHER}" != "mpirun" && "${LAUNCHER}" != "torchrun" ]]; then
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi

if ! [[ "${node_num}" =~ ^[1-9][0-9]*$ ]]; then
    echo "node_num must be a positive integer" >&2
    exit 1
fi
if [[ ! -f "${DTK_ENV}" ]]; then
    echo "DTK environment not found: ${DTK_ENV}. Run this script inside the training container." >&2
    exit 1
fi
source "${DTK_ENV}"

HOSTFILE="${hostfile_input}_slots"
rm -f "${HOSTFILE}"
if [[ "${LAUNCHER}" == "mpirun" ]]; then
    sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=8/' > "${HOSTFILE}"
else
    sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=1/' > "${HOSTFILE}"
fi

HOST="$(sed -n "1p" "${HOSTFILE}" | awk -F ' ' '{print $1}')"

if [[ "${LAUNCHER}" == "mpirun" ]]; then
    MPIRUN_NP=$((node_num * 8))
    PORT=${PORT:-25900}
else
    MPIRUN_NP=${node_num}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    MASTER_ADDR=${MASTER_ADDR:-${HOST}}
    MASTER_PORT=${MASTER_PORT:-25900}
fi

torchrun_args=""
if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun_args+="
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x PYTHONPATH \
        -x MASTER_ADDR=${MASTER_ADDR} \
        -x MASTER_PORT=${MASTER_PORT} \
        -x NNODES=${node_num} \
        -x GPUS_PER_NODE=${GPUS_PER_NODE} \
    "
fi

# optional: override train iters from the launcher, e.g. TRAIN_ITERS=50 bash run_chatglm2.sh ./hostfile 2
TRAIN_ITERS_ARG=""
if [[ -n "${TRAIN_ITERS}" ]]; then
    TRAIN_ITERS_ARG="--train_iters=${TRAIN_ITERS}"
fi

# Runs ChatGLM2 model
source "${NCCL_ENV}"

# A single node can launch torchrun directly, without an MPI/SSH hop.
if [[ "${LAUNCHER}" == "torchrun" && "${node_num}" == 1 ]]; then
    export MASTER_ADDR MASTER_PORT GPUS_PER_NODE
    export NNODES=1 NODE_RANK=0
    cd "${CURRENT_DIR}"
    train_args=(
        "${HOST}" "${MASTER_PORT}"
        "--data_path=${DATA_PATH}"
        "--launch_backend=${LAUNCHER}"
        "--tokenizer_path=${TOKENIZER_MODEL_PATH}"
        "--tensorboard_dir=${TENSORBOARD_DIR}"
        "--launch_with_binding=${LAUNCH_WITH_BINDING}"
        "--profiling=${profiling}"
    )
    if [[ -n "${TRAIN_ITERS}" ]]; then
        train_args+=( "--train_iters=${TRAIN_ITERS}" )
    fi
    bash "${TRAIN_SCRIPT}" "${train_args[@]}" 2>&1 |
        tee "log-${node_num}nodes-$(date +%F-%H%M%S).log"
    exit "${PIPESTATUS[0]}"
fi

CMD="mpirun -np ${MPIRUN_NP} --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    ${torchrun_args} \
    bash -c '
    source ${DTK_ENV} && \
    source ${NCCL_ENV} && \
    cd ${CURRENT_DIR} && \
    bash ${TRAIN_SCRIPT} \
    ${HOST} \
    ${PORT:-${MASTER_PORT}} \
    --data_path=${DATA_PATH} \
    --launch_backend=${LAUNCHER} \
    --tokenizer_path=${TOKENIZER_MODEL_PATH} \
    --tensorboard_dir=${TENSORBOARD_DIR} \
    --launch_with_binding=${LAUNCH_WITH_BINDING} \
    ${TRAIN_ITERS_ARG} \
    --profiling=${profiling}' 2>&1 | tee log-${node_num}nodes-$(date +%F-%H%M).log
"

eval ${CMD}
