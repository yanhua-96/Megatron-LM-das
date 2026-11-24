#!/bin/bash
# ============================================================================
# ChatGLM2-6B training performance test (no checkpoint I/O or evaluation)
#
# Reference architecture (THUDM/chatglm2-6b config.json):
#   num_layers                    28
#   hidden_size                   4096
#   ffn_hidden_size               13696          (SwiGLU)
#   num_attention_heads           32
#   kv_channels                   128            (= head_dim)
#   multi_query_group_num         2              (32 Q heads share 2 KV heads)
#   padded_vocab_size             65024
#   rmsnorm, layernorm_epsilon    1e-5
#   add_qkv_bias = true, add_bias_linear = false
#   tie_word_embeddings = false
#   original_rope = true  ->  partial RoPE: rotary_dim = kv_channels // 2 = 64
#                             (64/128 = 0.5), interleaved pairing (x[...,0],x[...,1])
#   attention_softmax_in_fp32 = true
#
# 未映射项说明：
#   apply_query_key_layer_scaling=true 仅对 fp16 数值稳定有意义，bf16 下不需要
#   开启（Megatron 侧对应 --apply-query-key-layer-scaling）。
#   src/prefix 相关的 P-Tuning 结构（pre_seq_len / prefix_projection）属于
#   ChatGLM2 的可选微调分支，本脚本不构建。
#
# NOTE: bridge (--use-bridge) does NOT cover ChatGLM (config architectures is
# "ChatGLMModel" and no ChatGLM bridge is registered), so this script uses the
# native MCore model path and trains from random initialization.
# ============================================================================

set -eo pipefail

INITIALIZATION_ARGS=( --num-workers 2)

for para in "$@"
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --tensorboard_dir* ]];then
        tensorboard_dir=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --launch_backend* ]];then
        launch_backend=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    elif [[ $para == --train_iters* ]];then
        train_iters=${para#*=}
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
DATA_PATH=${data_path:-${DATA_PATH:-}}
TOKENIZER_MODEL_PATH=${tokenizer_path:-${TOKENIZER_MODEL_PATH:-}}

if [[ -z "${DATA_PATH}" || -z "${TOKENIZER_MODEL_PATH}" ]]; then
    echo "DATA_PATH and TOKENIZER_MODEL_PATH must be set"
    exit 1
fi

# 运行环境参数
DIST_URL=${1:-${DIST_URL:-localhost}}
DIST_PORT=${2:-${DIST_PORT:-29500}}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MEGATRON_LAUNCH_BACKEND=${launch_backend:-"mpirun"}
MASTER_ADDR=${MASTER_ADDR:-${DIST_URL}}
MASTER_PORT=${MASTER_PORT:-${DIST_PORT}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
TENSORBOARD_DIR=${tensorboard_dir:-${TENSORBOARD_DIR:-${CURRENT_DIR}/tensorboard/$(date +%F-%H%M%S)}}

# default env
export GPU_MAX_HW_QUEUES=4

# tokenizer
#   HuggingFaceTokenizer  -> 指向 HF 模型目录（含 tokenizer 配置）
#   GPTSentencePieceTokenizer -> 指向 tokenizer.model；若 AutoTokenizer 因
#   trust_remote_code 不可用（ChatGLMTokenizer 依赖自定义代码），改用此项
TOKENIZER_TYPE=${TOKENIZER_TYPE:-GPTSentencePieceTokenizer}

# split hyperparameters
# ChatGLM2-6B 的 num_query_groups=2，MCore 要求 num_query_groups 能被 TP 整除，
# 因此 TP 只能取 1 或 2。8 卡默认 TP=1 / PP=1（DP=8）。
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
if (( TP > 2 )); then
    echo "TP=${TP} is invalid: ChatGLM2-6B has num_query_groups=2, TP must be <= 2"
    exit 1
fi

# batch hyperparameters
MBS=${MBS:-1}
GBS=${GBS:-256}

# seq hyperparameters
# seq_length=4096 用于训练；ChatGLM2 原生支持到 32768，roPE cache 按上限申请
SEQ_LEN=${SEQ_LEN:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-32768}

# train iteration hyperparameters
TRAIN_ITERS=${train_iters:-${TRAIN_ITERS:-10}}
LR=${LR:-1.5e-4}
MIN_LR=${MIN_LR:-1.5e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-1}

if ! [[ "${TRAIN_ITERS}" =~ ^[1-9][0-9]*$ && "${LR_WARMUP_ITERS}" =~ ^[0-9]+$ ]]; then
    echo "TRAIN_ITERS must be a positive integer and LR_WARMUP_ITERS a nonnegative integer" >&2
    exit 1
fi
if (( LR_WARMUP_ITERS >= TRAIN_ITERS )); then
    echo "LR_WARMUP_ITERS (${LR_WARMUP_ITERS}) must be less than TRAIN_ITERS (${TRAIN_ITERS})" >&2
    exit 1
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
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --num-layers 28
    --hidden-size 4096
    --ffn-hidden-size 13696
    --num-attention-heads 32
    --kv-channels 128
    --num-query-groups 2
    --group-query-attention
    --position-embedding-type rope
    --rotary-base 10000
    # rotary_dim = kv_channels // 2 = 64，占 head_dim(128) 的一半
    --rotary-percent 0.5
    # ChatGLM2 的 apply_rotary_pos_emb 按相邻维 (x[...,0], x[...,1]) 配对
    # 注意：若启动时报 "--rotary-interleaved does not work with rope_fusion"，
    # 说明 apply_rope_fusion 默认为 True，需追加 --no-apply-rope-fusion
    --rotary-interleaved
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    # add_qkv_bias: true
    --add-qkv-bias
    # add_bias_linear: false
    --disable-bias-linear
    # tie_word_embeddings: false
    --untie-embeddings-and-output-weights
    # 65024 可被 128 整除，padded_vocab_size 与 HF 保持一致
    --make-vocab-size-divisible-by 128
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
    --init-method-std 0.02
    --clip-grad 1.0
    --bf16
    --attention-dropout 0
    --hidden-dropout 0
    --attention-softmax-in-fp32
    --lr ${LR}
    --lr-decay-style cosine
    --min-lr ${MIN_LR}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --use-distributed-optimizer
)
if (( TP > 1 )); then
    MODEL_PARALLEL_ARGS+=( --sequence-parallel )
fi

DATA_ARGS=(
    --tokenizer-type ${TOKENIZER_TYPE}
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 1,0,0
)

# Performance tests only train; TensorBoard logs are independent of checkpoints.
LOGGING_ARGS=(
    --log-throughput
    --eval-iters 0
    # Dataset initialization divides by eval_interval even with eval_iters=0.
    --eval-interval 1000
    # The training loop uses save_interval in arithmetic even without --save.
    --save-interval 1000
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --tensorboard-log-interval 1
    --tensorboard-queue-size 1
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir torch_prof_chatglm2_6B_tp${TP}-pp${PP}-cp${CP}
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
        ${LOGGING_ARGS[@]} \
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
        ${LOGGING_ARGS[@]} \
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
