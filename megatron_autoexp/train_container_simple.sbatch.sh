#!/bin/bash -x
#SBATCH --account={ACCOUNT}
#SBATCH --nodes={NODES}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --time={TIME}
#SBATCH --partition={PARTITION}
#SBATCH --threads-per-core=1
#SBATCH --output={output_file}
#SBATCH --job-name={name}

MICRO_BATCH_SIZE={MICRO_BATCH_SIZE}
GAS={GAS}

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
# Set directory paths
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export PROJECT_DIR="{PROJECT_DIR_PATH}"
RUN_DIR="{LOGS}/{EXP_NAME}"
MEGATRON_CACHE_BASE="{SHARED_CONTAINERS}"
MEGATRON_CACHE_FOLDER="${{MEGATRON_CACHE_BASE}}/${{USER}}"
mkdir -p ${{MEGATRON_CACHE_FOLDER}}
TENSORBOARD_DIR="${{RUN_DIR}}/tensorboard"
mkdir -p $TENSORBOARD_DIR
export MEGATRON_CACHE="${{MEGATRON_CACHE_FOLDER}}/MEGATRON_CACHEDIR"
mkdir -p $MEGATRON_CACHE
export APPTAINER_CACHEDIR="{SHARED_CONTAINERS}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="{SHARED_CONTAINERS}/APPTAINER_TMPDIR"
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

IMAGE="{SHARED_CONTAINERS}/{IMAGE_NAME}"

#------------------------------------------------------------------------------
# Distributed Training Setup
#------------------------------------------------------------------------------
# Get number of GPUs per node from SLURM
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export NCCL_IB_TIMEOUT=150

# 3. Get the master's IP address *on the high-speed interface*
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR="${{MASTER_ADDR}}i"
# 4. Create a dynamic port from the job ID
MASTER_IP="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

# 5. Set critical networking variables for PyTorch and the container
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_SOCKET_FAMILY=AF_INET

export GLOO_SOCKET_IFNAME=ib0

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=1
# export GLOO_SOCKET_IFNAME=ib0 # for NCCL

# Configure data loading parameters
DATA_ARGS=(
    --data-path {DATA_PATH}
    --tokenizer-model {TOKENIZER_MODEL}
    --tokenizer-type {TOKENIZER_TYPE}
    --split 989,10,1  # Default is 969,30,1
    --num-workers {DATA_NUM_WORKERS}  
)

GPT_MODEL_ARGS=(
    --num-layers {NUM_LAYERS}
    --hidden-size {HIDDEN_SIZE}
    --num-attention-heads {NUM_ATTN_HEADS} 
    --seq-length {SEQ_LENGTH}
    --max-position-embeddings {MAX_POSITION_EMBEDDINGS}
    --ffn-hidden-size {FFN_HIDDEN_SIZE}
)

# Configure tensor and pipeline parallelism
TP={TP}  # Tensor parallelism degree
PP={PP}  # Pipeline parallelism degree

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP 
    --pipeline-model-parallel-size $PP
    --sequence-parallel  # Default is False
)

echo "NUM_GPUS: " {NUM_GPUS}
echo "MICRO_BATCH_SIZE: " {MICRO_BATCH_SIZE}
echo "GRADIENT_ACCUMULATION_STEPS: " {GAS}

# Do the entire calculation in one template expression,
# 4 * {NODES} * {MICRO_BATCH_SIZE} * {GAS} / TP
echo "GLOBAL_BATCH_SIZE: " {GLOBAL_BATCH_SIZE}

CHECKPOINT_FORMAT="torch"
if (( TP > 1 || PP > 1 )); then 
    CHECKPOINT_FORMAT="torch_dist"
fi

#------------------------------------------------------------------------------
# Training Schedule Configuration
#------------------------------------------------------------------------------
echo "TOTAL TOKENS: " {TOTAL_TOKENS_NUM} 
echo "TRAIN_ITERS: " {TRAIN_ITERS}
echo "LR_WARMUP_ITERS: " {LR_WARMUP_ITERS}  
echo "LR_DECAY_ITERS: " {LR_DECAY_ITERS}
echo "LR_WSD_DECAY_ITERS: " {LR_WSD_DECAY_ITERS}

#------------------------------------------------------------------------------
# Training Hyperparameters
#------------------------------------------------------------------------------
# Rotary parameters are set to default 
if [ "{PHASE}" == "COOLDOWN" ]; then
    WARMUP_ITERS=0
else
    WARMUP_ITERS={LR_WARMUP_ITERS}
fi
# Compile all training arguments
TRAINING_ARGS=(
    --micro-batch-size {MICRO_BATCH_SIZE}  # No default
    --global-batch-size {GLOBAL_BATCH_SIZE}  # No default
    --train-iters {TRAIN_ITERS}  # No default
    --weight-decay {WD}  # Default is 0.01
    --adam-beta1 {BETA1}  # Default is 0.9
    --adam-beta2 {BETA2}  # Default is 0.999
    --clip-grad {CLIP_GRAD}  # Default is 1.0
    --lr-decay-style {LR_DECAY_STYLE}
    --lr-wsd-decay-style {LR_WSD_DECAY_STYLE}
    --lr {LR}  # Default is 0.001
    --min-lr {MIN_LR}  # Default is 0.0
    --lr-warmup-iters $WARMUP_ITERS
    --lr-decay-iters {LR_DECAY_ITERS}
    --lr-wsd-decay-iters {LR_WSD_DECAY_ITERS}
    --data-cache-path $MEGATRON_CACHE  # No default
    --use-flash-attn  # Default is False
    --bf16  # Default is False (uses fp32)
    --qk-layernorm  # Default is False
    {DISABLE_BIAS_LINEAR}  # LLaMA-style: disable bias in linear layers
    --tensorboard-dir $TENSORBOARD_DIR  # No default
    --ckpt-format $CHECKPOINT_FORMAT  # Default is torch
    --position-embedding-type rope  # Default is learned
    --rotary-base 100000
    --rotary-percent 1.0
    --normalization RMSNorm
    --swiglu  # Default is False (uses GeLU)
    --distributed-backend nccl  # Default is nccl
    --use-distributed-optimizer  # Default is False
    --overlap-param-gather  # Default is False
    --overlap-grad-reduce  # Default is False
    --use-checkpoint-args  # Default is False
    --override-opt_param-scheduler  # Default is False
)

# USE --ckpt-format torch if not using TP, PP
# USE --ckpt-format torch_dist if using TP, PP

# These args substantially improve TFLOP/s/GPU (1.7B model on 54 nodes, 160 vs 140 with vs without)
# --overlap-param-gather
# --overlap-grad-reduce

CHECKPOINT_PATH="{RUN_DIR}/checkpoints"
# TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
# CHECKPOINT_PATH="$CHECKPOINT_PATH/{name}"

mkdir -p $CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="$CHECKPOINT_PATH/tensorboard"
mkdir -p $TENSORBOARD_LOGS_PATH

# Configure evaluation and logging arguments
EVAL_AND_LOGGING_ARGS=(
    --log-interval {LOG_INTERVAL}
    --save-interval {SAVE_INTERVAL}
    --eval-interval {TRAIN_ITERS} 
    --log-throughput  # Default is False
    --save $CHECKPOINT_PATH  # No default
    --load $CHECKPOINT_PATH  # No default
    --eval-iters {EVAL_ITERS}
    --tensorboard-dir $TENSORBOARD_LOGS_PATH  # No default
)

#------------------------------------------------------------------------------
# Command Construction and Execution
#------------------------------------------------------------------------------
# Construct the main training command
CMD="pretrain_gpt.py \
    ${{GPT_MODEL_ARGS[@]}} \
    ${{TRAINING_ARGS[@]}} \
    ${{MODEL_PARALLEL_ARGS[@]}} \
    ${{DATA_ARGS[@]}} \
    ${{EVAL_AND_LOGGING_ARGS[@]}}"

# Set up distributed training arguments
DISTRIBUTED_ARGS=(
    --nproc-per-node $GPUS_PER_NODE 
    --nnodes $NNODES
)
# Configure the container launcher
LAUNCHER="singularity exec \
    --nv \
    $IMAGE \
   python -u -m torch.distributed.run \
    ${{DISTRIBUTED_ARGS[@]}} \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend static \
    --max_restarts 0 \
    --tee 3 \
    "
SRUN_ARGS=" \
    --wait=60 --threads-per-core=1 \
    --kill-on-bad-exit=1"
# Set Megatron path
MEGATRON_PATH="{SHARED_MEGATRON}"
cd $MEGATRON_PATH

srun $SRUN_ARGS \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"