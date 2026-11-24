#!/bin/bash

set -eo pipefail

for para in "$@"
do
    if [[ $para == --model=* ]]; then
        model=${para#*=}
    elif [[ $para == --profiling* ]]; then
        profiling=${para#*=}
    elif [[ $para == --load_checkpoint_path=* ]]; then
        load_checkpoint_path=${para#*=}
    elif [[ $para == --save_checkpoint_path=* ]]; then
        save_checkpoint_path=${para#*=}
    elif [[ $para == --save_ckpt=* ]]; then
        save_ckpt=${para#*=}
    elif [[ $para == --train_iters=* ]]; then
        train_iters=${para#*=}
    elif [[ $para == --save_interval=* ]]; then
        save_interval=${para#*=}
    fi
done

# Those variables no need to modify
hostfile_input=${CURRENT_DIR}/hostfile
node_num=1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
DTK_ENV=${DTK_ENV:-/opt/dtk/env.sh}
DATA_PATH=${DATA_PATH:-${CURRENT_DIR}/data/dataset/oscar-en-10k_text_document}
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-${CURRENT_DIR}/data/tokenizer}
LAUNCHER=${LAUNCHER:-torchrun}
SAVE_CKPT=${save_ckpt:-false}
if [[ "${SAVE_CKPT}" != "true" && "${SAVE_CKPT}" != "false" ]]; then
    echo "--save_ckpt must be true or false." >&2
    exit 1
fi
SAVE_CHECKPOINT_PATH=${save_checkpoint_path:-./ckpt_save}
TRAIN_ITERS=${train_iters:-10}
SAVE_INTERVAL=${save_interval:-${TRAIN_ITERS}}
MODEL=${model:-7B}
case "${MODEL}" in
    7B|13B|70B) ;;
    *) echo "--model must be 7B, 13B or 70B." >&2; exit 1 ;;
esac
TRAIN_SCRIPT=train_llama2_${MODEL}.sh                                   # script under this example directory to run
NCCL_ENV=${NCCL_ENV:-${MEGATRON_PATH}/requirements/env.sh}                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used
SSH_PORT=${SSH_PORT:-}                                                   # mpirun 跨机 ssh 端口；容器 sshd 不是默认 22 时必须设置（hcu-bench 跑时由 container.ssh_port 自动注入）

if [[ "${LAUNCHER}" != "mpirun" && "${LAUNCHER}" != "torchrun" ]]; then
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi

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

# mpirun 默认走 22 端口 ssh 到远端节点拉起进程；容器内 sshd 常不是 22，跨机必须用
# plm_rsh_args 传实际端口，否则会卡在连接超时/被拒（root@<node>: Permission denied）。
MPIRUN_SSH_ARGS=""
if [[ "${LAUNCHER}" == "mpirun" && -n "${SSH_PORT}" ]]; then
    MPIRUN_SSH_ARGS="--mca plm_rsh_args '-p ${SSH_PORT}'"
fi

torchrun_args=""
if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun_args+="
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x PYTHONPATH \
        -x HIP_VISIBLE_DEVICES \
        -x MASTER_ADDR=${MASTER_ADDR} \
        -x MASTER_PORT=${MASTER_PORT} \
        -x NNODES=${node_num} \
        -x GPUS_PER_NODE=${GPUS_PER_NODE} \
    "
fi

# Runs Llama2 model
source "${NCCL_ENV}"

CMD="mpirun -np ${MPIRUN_NP} --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    ${MPIRUN_SSH_ARGS} \
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
    --load_checkpoint_path=${load_checkpoint_path:-} \
    --save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
    --save_ckpt=${SAVE_CKPT} \
    --train_iters=${TRAIN_ITERS} \
    --save_interval=${SAVE_INTERVAL} \
    --launch_with_binding=${LAUNCH_WITH_BINDING} \
    --profiling=${profiling}' 2>&1 | tee BW1100-log-${node_num}nodes-$(date +%F-%H%M).log
"

eval ${CMD}
wait
