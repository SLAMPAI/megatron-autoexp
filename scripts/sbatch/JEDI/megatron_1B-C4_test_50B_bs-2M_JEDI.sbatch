#!/bin/bash

#SBATCH --nodes=16
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=280
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=all
#SBATCH --account=jureap59
#SBATCH --threads-per-core=1
#SBATCH --output=/p/project1/laionize/jitsev1_juwelsbooster/megatron_lm_reference/slurm_output/%x_%j.out

# Author: Jenia Jitsev, 23.12.2024
# USAGE: sbatch SCRIPT_NAME.sbatch MICRO_BATCH_SIZE GAS
# WARNING: ADAPT corresponding pathes carefully
# Reference run: sbatch megatron_1B-C4_test_50B_bs-2M_JEDI.sbatch 16 1
# 16 nodes on JEDI (64 H100, 96 GB); 50B C4 reference run
# Gives global bs = 1024; approx. 942ms per iteration, 315 TFLOP/s/GPU
# 34785 Tokens/s/GPU, 2226276 Tokens/s for 64 GPUs; projected 6.5 hours for 50B tokens
# Total cost : ca. 400 H100 GPU hours, 1.7B model, 50B tokens

MICRO_BATCH_SIZE=$1
GAS=$2

export RUN_DIR="/p/project1/laionize/jitsev1_juwelsbooster/megatron_lm_reference"
export SHARED_CONTAINERS=/p/data1/mmlaion/shared/containers/
export APPTAINER_CACHEDIR="${SHARED_CONTAINERS}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${SHARED_CONTAINERS}/APPTAINER_TMPDIR"
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

IMAGE=${SHARED_CONTAINERS}/pytorch_24.09-py3_ARM.sif


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

export MEGATRON_CACHE="${SHARED_CONTAINERS}/MEGATRON_CACHEDIR"
mkdir -p $MEGATRON_CACHE
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p $TENSORBOARD_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=1
# export GLOO_SOCKET_IFNAME=ib0 # for NCCL


# Data args Starcoder test

# /p/data1/mmlaion/text-data/tokenized/c4

DATA=/p/data1/mmlaion/text-data
DATA_PATH="$DATA/tokenized/c4/merged"
TOKENIZER_MODEL="/p/data1/mmlaion/marianna/models/EleutherAI/gpt-neox-20b"
TOKENIZER_TYPE="HuggingFaceTokenizer"
# VOCAB_FILE="$DATA/vocab.json"
# MERGE_FILE="$DATA/merges.txt"

# Data args C4 reference
# DATA=/p/data1/mmlaion/text-data/RedPajama-Data-1T/c4

DATA_NUM_WORKERS=8
DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --tokenizer-type $TOKENIZER_TYPE
    --split 989,10,1
    --num-workers $DATA_NUM_WORKERS
)

# --split 969,30,1
# --split 949,50,1
# --split 1000,0,0



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

# GPT args 1.3b
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# MAX_POSITION_EMBEDDINGS=2048
# SEQ_LENGTH=2048

# GPT args 1.7b (NVIDIA reference table)
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


# GLOBAL_BATCH_SIZE = ${MICRO_SBATCH_SIZE} * SLURM_GPUS_PER_NODE
# GAS=1
# GAS=2
# GAS=4
# GAS=16
NUM_GPUS=$((SLURM_GPUS_PER_NODE*SLURM_JOB_NUM_NODES))


echo "SLURM_GPUS_PER_NODE: " $SLURM_GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES: " $SLURM_JOB_NUM_NODES 
echo "NUM_GPUS: " $NUM_GPUS
GLOBAL_BATCH_SIZE=$(((NUM_GPUS*MICRO_BATCH_SIZE*GAS)/TP))

echo "MICRO_BATCH_SIZE: " $MICRO_BATCH_SIZE
echo "GRADIENT_ACCUMULATION_STEPS: " $GAS
echo "GLOBAL_BATCH_SIZE: " $GLOBAL_BATCH_SIZE


# Training args

CHECKPOINT_FORMAT="torch"

if (( TP > 1 || PP > 1 )); then 

    CHECKPOINT_FORMAT="torch_dist"

fi

TOTAL_TOKENS_NUM=50000000000 # 50B tokens
# TOTAL_TOKENS_NUM=20000000000 # 20B tokens
LR_WARMUP_ITERS=2000
# ceil for LR_DECAY_ITERS = TOTAL_TOKENS_NUM / SEQ_LENGTH / GLOBAL_BATCH_SIZE
# LR_DECAY_ITERS=$(((${TOTAL_TOKENS_NUM} + (${SEQ_LENGTH} * ${GLOBAL_BATCH_SIZE}) - 1)/(${SEQ_LENGTH}*${GLOBAL_BATCH_SIZE})))
# TRAIN_ITERS=$((${LR_WARMUP_ITERS} + ${LR_DECAY_ITERS}))

# ceil for total train iterations, TRAIN_ITERS = TOTAL_TOKENS_NUM / SEQ_LENGTH / GLOBAL_BATCH_SIZE
TRAIN_ITERS=$(((${TOTAL_TOKENS_NUM} + (${SEQ_LENGTH} * ${GLOBAL_BATCH_SIZE}) - 1)/(${SEQ_LENGTH}*${GLOBAL_BATCH_SIZE})))
# LR_DECAY_ITERS=$((${TRAIN_ITERS}-${LR_WARMUP_ITERS}))
LR_DECAY_ITERS=${TRAIN_ITERS}

SAVE_INTERVAL=2000
# SAVE_INTERVAL=60
EVAL_INTERVAL=${TRAIN_ITERS}
LOG_INTERVAL=50
EVAL_ITERS=1

echo "TOTAL TOKENS: " $TOTAL_TOKENS_NUM
echo "TRAIN_ITERS: " $TRAIN_ITERS
echo "LR_WARMUP_ITERS: " $LR_WARMUP_ITERS
echo "LR_DECAY_ITERS: " $LR_DECAY_ITERS


ROTARY_BASE=10000
# ROTARY_PERCENT=0.25
# DEFAULT;
ROTARY_PERCENT=1.0

NORM_EPSILON=1e-5
INIT_METHOD_STD=0.02

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.05 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02
    --clip-grad 1.0 
    --lr-decay-style cosine
    --lr 3.0e-3 
    --min-lr 3.0e-5
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --lr-decay-iters ${LR_DECAY_ITERS}
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

CHECKPOINT_PATH="${RUN_DIR}/checkpoints"
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
EXP_LABEL="open-sci-ref-1.7b-cosine-C4-50B-JEDI"
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
    --wait=60 --cpus-per-task=280 --threads-per-core=1 \
    --kill-on-bad-exit=1 \
    "

# MEGATRON_PATH="/p/project1/laionize/marianna/megatron/Megatron-LM"

MEGATRON_PATH="/p/data1/mmlaion/shared/repos/Megatron-LM"
cd $MEGATRON_PATH

srun $SRUN_ARGS \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"