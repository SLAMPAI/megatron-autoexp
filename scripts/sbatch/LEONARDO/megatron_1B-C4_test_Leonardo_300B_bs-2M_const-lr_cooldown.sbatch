#!/bin/bash

#SBATCH --nodes=54
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --account=EUHPC_E03_068
#SBATCH --partition=boost_usr_prod
# #SBATCH --qos=boost_qos_bprod
# #SBATCH --qos=boost_qos_dbg
#SBATCH --threads-per-core=1
#SBATCH --output=/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/%x_%j.out

# Author: Jenia Jitsev, 23.12.2024
# USAGE: sbatch SCRIPT_NAME.sbatch MICRO_BATCH_SIZE GAS
# WARNING: ADAPT corresponding pathes carefully
# Reference run: sbatch megatron_1B-C4_test_Leonardo_300B_bs-2M_const-lr_cooldown.sbatch 6 1
# 54 nodes on LEONARDO (216 A100, 64 GB); 50B C4 reference run
# Gives global bs = 1296 (2.65M); approx. 725ms per iteration, 150 TFLOP/s/GPU
# 16948 Tokens/s/GPU, 3660976 Tokens/s for 216 GPUs; projected 3.8 hours for 50B tokens
# Total cost : ca. 820 A100 GPU hours, 1.7B model, 50B tokens; ca. 4916 A100 hours (300B)
# Const LR + Cooldown (0.2 of total training duration) schedule (WSD, trapezoid)

# For large node numbers (> 64 nodes): add #SBATCH --qos=boost_qos_bprod
# For quick tests on max 2 nodes: #SBATCH --qos=boost_qos_dbg

MICRO_BATCH_SIZE=$1
GAS=$2

export PROJECT_DIR="/leonardo_work/EUHPC_E03_068"
export RUN_DIR="/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference"
export SHARED_CONTAINERS="/leonardo_work/EUHPC_E03_068/shared/container_images"

# MEGATRON_CACHE_BASE="/leonardo_scratch/fast/EUHPC_E03_068"
MEGATRON_CACHE_BASE="/leonardo_scratch/large/userexternal"
MEGATRON_CACHE_FOLDER="${MEGATRON_CACHE_BASE}/${USER}"
mkdir -p ${MEGATRON_CACHE_FOLDER}

export MEGATRON_CACHE="${MEGATRON_CACHE_FOLDER}/MEGATRON_CACHEDIR"
mkdir -p $MEGATRON_CACHE
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p $TENSORBOARD_DIR

export APPTAINER_CACHEDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_TMPDIR"

# export APPTAINER_CACHEDIR="${SHARED_CONTAINERS}/APPTAINER_CACHEDIR"
# export APPTAINER_TMPDIR="${SHARED_CONTAINERS}/APPTAINER_TMPDIR"

mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

# IMAGE=${SHARED_CONTAINERS}/pytorch_24.09-py3_JUWELS.sif
IMAGE=${SHARED_CONTAINERS}/pytorch_24.09-py3_leonardo.sif


# necessary on JUWELS to handle GLOO CPU communication related errors; on JEDI, this is not required
# export GLOO_SOCKET_IFNAME=ib0
# export GLOO_SOCKET_IFNAME=eth0
# echo "GLOO_SOCKET_IFNAME: " ${GLOO_SOCKET_IFNAME} 

# On Leonardo, if experimenting with setting GLOO_SOCKET_IFNAME=ib0, NCCL_SOCKET_IFNAME has to be also set accordingly
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_DEBUG=INFO

# NCCL settings to improve distributed training stability (handling flipping links, irresponsive nodes, etc)
# waiting for 120s in case nodes become irresponsive giving a chance to recover
export NCCL_IB_TIMEOUT=120

# Training setup
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR="${MASTER_ADDR}"
MASTER_IP="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
echo $MASTER_IP
export MASTER_ADDR=$MASTER_IP
# export MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=12345
NNODES=$SLURM_NNODES
# NODE_RANK=$SLURM_PROCID
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=1
# export GLOO_SOCKET_IFNAME=ib0 # for NCCL


# Data args Starcoder test

# /p/data1/mmlaion/text-data/tokenized/c4

DATA="/leonardo_work/EUHPC_E03_068/shared/datasets/language"
DATA_PATH="$DATA/tokenized/c4/GPT-NeoX/merged"
TOKENIZER_MODEL="/leonardo_work/EUHPC_E03_068/shared/models/EleutherAI/gpt-neox-20b"
TOKENIZER_TYPE="HuggingFaceTokenizer"
# VOCAB_FILE="$DATA/vocab.json"
# MERGE_FILE="$DATA/merges.txt"

# Data args C4 reference
# DATA=/p/data1/mmlaion/text-data/RedPajama-Data-1T/c4

DATA_NUM_WORKERS=4
DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type $TOKENIZER_TYPE
    --split 989,10,1
    --num-workers $DATA_NUM_WORKERS
)

# Distributed args
# DISTRIBUTED_ARGS=(
#     --nproc-per-node $GPUS_PER_NODE 
#     --nnodes $NNODES
#     --master-addr $MASTER_ADDR 
#     --master-port $MASTER_PORT
#     --distributed-backend nccl 
# )


# # GPT model args 70b
# NUM_LAYERS=80
# HIDDEN_SIZE=8192
# NUM_ATTN_HEADS=64
# # GPT model args 34b

# NUM_LAYERS=60
# HIDDEN_SIZE=6656
# NUM_ATTN_HEADS=52
# MAX_POSITION_EMBEDDINGS=8192
# SEQ_LENGTH=8192

# GPT args 1.3b (NVIDIA reference table tells 1.7B as their vocabulary is 128K, compared to only 50K)
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# MAX_POSITION_EMBEDDINGS=2048
# SEQ_LENGTH=2048

# GPT args 1.7b
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
MAX_POSITION_EMBEDDINGS=2048
SEQ_LENGTH=2048


GPT_MODEL_ARGS=(
    --num-layers $NUM_LAYERS 
    --hidden-size $HIDDEN_SIZE 
    --num-attention-heads $NUM_ATTN_HEADS 
    --seq-length $SEQ_LENGTH 
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
)


# TP=$GPUS_PER_NODE
TP=1
PP=1

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP 
	--pipeline-model-parallel-size $PP
    --sequence-parallel
)

#     --sequence-parallel

# GLOBAL_BATCH_SIZE = ${MICRO_BATCH_SIZE} * SLURM_GPUS_PER_NODE
# GAS=1
# GAS=4
NUM_GPUS=$((SLURM_GPUS_PER_NODE*SLURM_JOB_NUM_NODES))

echo "SLURM_GPUS_PER_NODE: " $SLURM_GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES: " $SLURM_JOB_NUM_NODES 
echo "NUM_GPUS: " $NUM_GPUS
GLOBAL_BATCH_SIZE=$(((NUM_GPUS*MICRO_BATCH_SIZE*GAS)/TP))

echo "MICRO_BATCH_SIZE: " $MICRO_BATCH_SIZE
echo "GRADIENT_ACCUMULATION_STEPS: " $GAS
echo "GLOBAL_BATCH_SIZE: " $GLOBAL_BATCH_SIZE


CHECKPOINT_FORMAT="torch"

if (( TP > 1 || PP > 1 )); then 

    CHECKPOINT_FORMAT="torch_dist"

fi

TOTAL_TOKENS_NUM=300000000000 # 300B tokens
# TOTAL_TOKENS_NUM=50000000000 # 50B tokens
# TOTAL_TOKENS_NUM=20000000000 # 20B tokens
LR_WARMUP_ITERS=2000
# ceil for LR_DECAY_ITERS = TOTAL_TOKENS_NUM / SEQ_LENGTH / GLOBAL_BATCH_SIZE
# LR_DECAY_ITERS=$(((${TOTAL_TOKENS_NUM} + (${SEQ_LENGTH} * ${GLOBAL_BATCH_SIZE}) - 1)/(${SEQ_LENGTH}*${GLOBAL_BATCH_SIZE})))
# TRAIN_ITERS=$((${LR_WARMUP_ITERS} + ${LR_DECAY_ITERS}))
COOLDOWN_FRACTION=1/5
# ceil for total train iterations, TRAIN_ITERS = TOTAL_TOKENS_NUM / SEQ_LENGTH / GLOBAL_BATCH_SIZE
TRAIN_ITERS=$(((${TOTAL_TOKENS_NUM} + (${SEQ_LENGTH} * ${GLOBAL_BATCH_SIZE}) - 1)/(${SEQ_LENGTH}*${GLOBAL_BATCH_SIZE})))
# LR_DECAY_ITERS=$((${TRAIN_ITERS}-${LR_WARMUP_ITERS}))
LR_DECAY_ITERS=$TRAIN_ITERS
# LR_WSD_DECAY_ITERS=$((${TRAIN_ITERS}-${TRAIN_ITERS} * ${COOLDOWN_FRACTION}))
LR_WSD_DECAY_ITERS=$((${TRAIN_ITERS} * ${COOLDOWN_FRACTION}))


SAVE_INTERVAL=2000
# SAVE_INTERVAL=60
EVAL_INTERVAL=${TRAIN_ITERS}
LOG_INTERVAL=50
EVAL_ITERS=1

echo "TOTAL TOKENS: " $TOTAL_TOKENS_NUM
echo "TRAIN_ITERS: " $TRAIN_ITERS
echo "LR_WARMUP_ITERS: " $LR_WARMUP_ITERS
echo "LR_DECAY_ITERS: " $LR_DECAY_ITERS
echo "LR_WSD_DECAY_ITERS: " $LR_WSD_DECAY_ITERS

LR_DECAY_STYLE="WSD"
LR_WSD_DECAY_STYLE="linear"

ROTARY_BASE=10000
# ROTARY_PERCENT=0.25
# DEFAULT;
ROTARY_PERCENT=1.0

NORM_EPSILON=1e-5
INIT_METHOD_STD=0.02

# Training args
TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.05 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0 
    --lr-decay-style ${LR_DECAY_STYLE}
    --lr-wsd-decay-style ${LR_WSD_DECAY_STYLE}
    --lr 1e-3    
    --min-lr 0.0
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS}
    --data-cache-path $MEGATRON_CACHE
    --use-flash-attn
    --bf16
    --qk-layernorm  
    --tensorboard-dir $TENSORBOARD_DIR
    --ckpt-format $CHECKPOINT_FORMAT
    --position-embedding-type rope
    --rotary-base ${ROTARY_BASE}
    --rotary-percent ${ROTARY_PERCENT}
    --normalization RMSNorm
    --norm-epsilon ${NORM_EPSILON}
    --init-method-std ${INIT_METHOD_STD}
    --swiglu
    --distributed-backend nccl 
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
)

# USE --ckpt-format torch if not using TP, PP
# USE --ckpt-format torch_dist if using TP, PP

# These args substantially improve TFLOP/s/GPU (1.7B model on 54 nodes, 160 vs 140 with vs without)
# --overlap-param-gather
# --overlap-grad-reduce

# --overlap-param-gather-with-optimizer-step with interleaved pipeline parallelism

CHECKPOINT_PATH="${RUN_DIR}/checkpoints"
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
EXP_LABEL="open-sci-ref-1.7b-const-lr-cooldown-C4-300B-LEONARDO"
CHECKPOINT_PATH="$CHECKPOINT_PATH/$TIMESTAMP-${EXP_LABEL}"

mkdir -p $CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="$CHECKPOINT_PATH/tensorboard"
mkdir -p $TENSORBOARD_LOGS_PATH


# Eval and logging args
EVAL_AND_LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --save-interval ${SAVE_INTERVAL} 
    --eval-interval ${EVAL_INTERVAL} 
    --log-throughput
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters ${EVAL_ITERS}
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Command
CMD="pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    "

# Distributed args
DISTRIBUTED_ARGS=(
    --nproc-per-node $GPUS_PER_NODE 
    --nnodes $NNODES
)


LAUNCHER="singularity exec \
    --nv \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $MEGATRON_CACHE_FOLDER:$MEGATRON_CACHE_FOLDER \
    $IMAGE \
   python -u -m torch.distributed.run \
    ${DISTRIBUTED_ARGS[@]} \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend static \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD


SRUN_ARGS=" \
    --wait=60 --cpus-per-task=32 --threads-per-core=1 \
    --kill-on-bad-exit=1"

# MEGATRON_PATH="/p/project1/laionize/marianna/megatron/Megatron-LM"

MEGATRON_PATH="/leonardo_work/EUHPC_E03_068/shared/repos/Megatron-LM"
cd $MEGATRON_PATH

srun $SRUN_ARGS \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"