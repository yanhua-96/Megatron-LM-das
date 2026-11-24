#!/bin/bash

INITIALIZATION_ARGS=( --num-workers 2)

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --load_checkpoint_path* ]];then
        load_checkpoint_path=${para#*=}
    elif [[ $para == --save_checkpoint_path* ]];then
        save_checkpoint_path=${para#*=}
    elif [[ $para == --save_ckpt* ]];then
        save_ckpt=${para#*=}
    elif [[ $para == --train_iters* ]];then
        train_iters=${para#*=}
    elif [[ $para == --save_interval* ]];then
        save_interval=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --launch_backend* ]];then
        launch_backend=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    elif [[ $para == --reproduce* ]];then
        INITIALIZATION_ARGS=( --reproduce --num-workers 0)
        export MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC=1  # miopen 确定算法打开
        export ROCBLAS_ATOMICS_MOD=0                     # rocblas 关闭原子操作
        # 关闭miopen中的atomic操作算法, 只保留gemm算法
        export MIOPEN_DEBUG_CONV_FFT=0
        export MIOPEN_DEBUG_CONV_DIRECT=0
        export MIOPEN_DEBUG_CONV_GEMM=1
        export MIOPEN_DEBUG_CONV_WINOGRAD=0
        export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    fi
done

# data path
DATA_PATH=${data_path}
TOKENIZER_MODEL_PATH=${tokenizer_path}
LOAD_CHECKPOINT_PATH=${load_checkpoint_path:-}
SAVE_CKPT=${save_ckpt:-false}
SAVE_CHECKPOINT_PATH=${save_checkpoint_path:-}

# 运行环境参数
DIST_URL=${1}
DIST_PORT=${2}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MEGATRON_LAUNCH_BACKEND=${launch_backend:-"mpirun"}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
TENSORBOARD_DIR=${tensorboard_dir:-${TENSORBOARD_DIR:-${CURRENT_DIR}/tensorboard/$(date +%F-%H%M%S)}}

# default env
export GPU_MAX_HW_QUEUES=4

# split hyperparameters
TP=4
PP=8
CP=1

# batch hyperparameters
MBS=1
GBS=128

# seq hyperparameters
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=8192

# train iteration hyperparameters
# train_iters is the total, including iterations already done in the loaded checkpoint.
TRAIN_ITERS=${train_iters:-10}
SAVE_INTERVAL=${save_interval:-${TRAIN_ITERS}}
LR_WARMUP_ITERS=1

CHECKPOINT_ARGS=""
if [[ "${SAVE_CKPT}" == "true" ]]; then
    if [[ -z "${SAVE_CHECKPOINT_PATH}" ]]; then
        echo "--save_checkpoint_path is required with --save_ckpt=true." >&2
        exit 1
    fi
    SAVE_CHECKPOINT_PATH=$(realpath -m -- "${SAVE_CHECKPOINT_PATH}") || exit 1
    CHECKPOINT_ARGS="--save ${SAVE_CHECKPOINT_PATH} "
fi
if [[ -n "${LOAD_CHECKPOINT_PATH}" ]]; then
    LOAD_CHECKPOINT_PATH=$(realpath -m -- "${LOAD_CHECKPOINT_PATH}") || exit 1
    if [[ "${SAVE_CKPT}" == "true" && "${LOAD_CHECKPOINT_PATH}" == "${SAVE_CHECKPOINT_PATH}" ]]; then
        echo "Load and save checkpoint directories must be different." >&2
        exit 1
    fi
    tracker="${LOAD_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"
    if [[ ! -r "${tracker}" ]]; then
        echo "Missing checkpoint tracker: ${tracker}" >&2
        exit 1
    fi
    checkpoint_iteration=$(<"${tracker}")
    if [[ ! "${checkpoint_iteration}" =~ ^[0-9]+$ ]]; then
        echo "Resume requires a numeric training iteration in ${tracker}." >&2
        exit 1
    fi
    checkpoint_iteration=$((10#${checkpoint_iteration}))
    if (( TRAIN_ITERS <= checkpoint_iteration )); then
        echo "train_iters (${TRAIN_ITERS}) must exceed checkpoint iteration (${checkpoint_iteration})." >&2
        exit 1
    fi
    # common.pt holds Python objects (e.g. signal.SIGTERM), not only weights;
    # PyTorch >= 2.6 otherwise loads without weights_only=False.
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    CHECKPOINT_ARGS+="--load ${LOAD_CHECKPOINT_PATH} --exit-on-missing-checkpoint --use-checkpoint-opt-param-scheduler "
    echo "Resume from iteration ${checkpoint_iteration}: ${LOAD_CHECKPOINT_PATH}"
fi

MPI_DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

TORCH_DISTRIBUTED_ARGS=(
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --nproc_per_node $GPUS_PER_NODE
)

GPT_MODEL_ARGS=(
    --seq-length ${SEQ_LEN}
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --group-query-attention
    --num-query-groups 8
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
)

TRAINING_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models 
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --swiglu
    --lr 3.0e-5 
    --lr-decay-style cosine 
    --min-lr 3.0e-6
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --ckpt-format torch_dist
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --use-distributed-optimizer 
    --sequence-parallel
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH} 
    --split 1,0,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --eval-iters 0
    --log-interval 1
    --save-interval ${SAVE_INTERVAL}
    --eval-interval 1000 
    ${CHECKPOINT_ARGS}
    --tensorboard-dir "${TENSORBOARD_DIR}"
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_llama2_70B_tp${TP}-pp${PP}-cp${CP}
    --use-pytorch-profiler
)

HIP_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 4
    --profile-step-end 5
    --use-hip-profiler
)

if [[ "$MEGATRON_LAUNCH_BACKEND" == "mpirun" ]]; then
    APP="python -u ${MEGATRON_PATH}/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${MPI_DISTRIBUTED_ARGS[@]} \
        ${INITIALIZATION_ARGS[@]} \
        "
elif [[ "$MEGATRON_LAUNCH_BACKEND" == "torchrun" ]]; then
    APP="torchrun ${TORCH_DISTRIBUTED_ARGS[@]} \
        ${MEGATRON_PATH}/pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${INITIALIZATION_ARGS[@]} \
        "
else
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi
if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS[@]}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

#for hygon cpu
if [[ "$MEGATRON_LAUNCH_BACKEND" == "mpirun" ]]; then
    ${launch_with_binding} ${LOCAL_RANK} ${APP}
elif [[ "$MEGATRON_LAUNCH_BACKEND" == "torchrun" ]]; then
    echo ${APP}
    ${APP}
else
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi
