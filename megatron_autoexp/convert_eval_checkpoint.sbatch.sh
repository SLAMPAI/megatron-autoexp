#!/bin/bash
#SBATCH --account={ACCOUNT}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition={PARTITION}
#SBATCH --threads-per-core=1
#SBATCH --output={output_file}
#SBATCH --job-name={name}

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export PROJECT_DIR="{PROJECT_DIR_PATH}"
RUN_DIR="{RUN_DIR}"
MEGATRON_CACHE_BASE="{SHARED_CONTAINERS}"
MEGATRON_CACHE_FOLDER="${{MEGATRON_CACHE_BASE}}/${{USER}}"
mkdir -p ${{MEGATRON_CACHE_FOLDER}}

export MEGATRON_CACHE="${{MEGATRON_CACHE_FOLDER}}/MEGATRON_CACHEDIR"
mkdir -p $MEGATRON_CACHE
export APPTAINER_CACHEDIR="{SHARED_CONTAINERS}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="{SHARED_CONTAINERS}/APPTAINER_TMPDIR"
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

IMAGE="{SHARED_CONTAINERS}/{IMAGE_NAME}"

# Get master address for multi-node setup (even though using 1 node)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

#------------------------------------------------------------------------------
# Conversion Configuration
#------------------------------------------------------------------------------
# Paths
SOURCE_CHECKPOINT_PATH="{SOURCE_CHECKPOINT_PATH}"
TARGET_HF_PATH="{TARGET_HF_PATH}"
OPEN_SCI_HF_PATH="{OPEN_SCI_HF_PATH}"

# Model configuration parameters (use values from training config)
ARCHITECTURE="{ARCHITECTURE}"
MODEL_TYPE="{MODEL_TYPE}"
TOKENIZER_NAME="{TOKENIZER_NAME}"
NUM_KEY_VALUE_HEADS="{NUM_ATTN_HEADS}"
INTERMEDIATE_SIZE="{INTERMEDIATE_SIZE}"
VOCAB_SIZE="{VOCAB_SIZE}"

# Training architecture parameters - will be substituted from EXPERIMENTS section
HIDDEN_SIZE="{HIDDEN_SIZE}"
NUM_LAYERS="{NUM_LAYERS}"
NUM_ATTN_HEADS="{NUM_ATTN_HEADS}"
MAX_POSITION_EMBEDDINGS="{MAX_POSITION_EMBEDDINGS}"

# Parallelism configuration
TARGET_TP_SIZE="{TARGET_TP_SIZE}"
TARGET_PP_SIZE="{TARGET_PP_SIZE}"
WORLD_SIZE=$(($TARGET_TP_SIZE * $TARGET_PP_SIZE))

#------------------------------------------------------------------------------
# Find Latest Unconverted Checkpoint
#------------------------------------------------------------------------------
echo "Searching for unconverted checkpoints in: $SOURCE_CHECKPOINT_PATH"

# Create tracking directories
CONVERTED_TRACKING_DIR="{RUN_DIR}/converted_checkpoints"
IN_PROGRESS_TRACKING_DIR="{RUN_DIR}/in_progress_checkpoints"
mkdir -p "$CONVERTED_TRACKING_DIR"
mkdir -p "$IN_PROGRESS_TRACKING_DIR"

# Process ALL unconverted checkpoints
while true; do
    # Find all available checkpoint iterations (only directories, handle zero-padded numbers)
    AVAILABLE_ITERS=$(find "$SOURCE_CHECKPOINT_PATH" -maxdepth 1 -type d -name "iter_*" -printf "%f\n" 2>/dev/null | sed 's/iter_0*//' | sort -n)
    
    if [[ -z "$AVAILABLE_ITERS" ]]; then
        echo "ERROR: No checkpoints found in $SOURCE_CHECKPOINT_PATH"
        exit 1
    fi
    
    # Find the earliest unconverted checkpoint
    TARGET_ITER=""
    for iter in $AVAILABLE_ITERS; do
        # Skip if already converted or in progress
        if [[ ! -f "$CONVERTED_TRACKING_DIR/iter_$iter.done" && ! -f "$IN_PROGRESS_TRACKING_DIR/iter_$iter.lock" ]]; then
            TARGET_ITER=$iter
            break  # Take the first (earliest) unconverted checkpoint
        fi
    done
    
    # If no unconverted checkpoints found, we're done
    if [[ -z "$TARGET_ITER" ]]; then
        echo "INFO: All available checkpoints have been converted."
        break
    fi
    
    # Mark checkpoint as in progress
    touch "$IN_PROGRESS_TRACKING_DIR/iter_$TARGET_ITER.lock"
    
    # Handle zero-padded checkpoint directory names
    LOAD_CHECKPOINT_PATH="$SOURCE_CHECKPOINT_PATH/iter_$(printf "%07d" $TARGET_ITER)"
    echo "Converting checkpoint: $LOAD_CHECKPOINT_PATH (iteration $TARGET_ITER)"
    
    # Create checkpoint-specific target directory
    TARGET_HF_PATH_ITER="${{TARGET_HF_PATH}}/iter_$TARGET_ITER"
    mkdir -p "$TARGET_HF_PATH_ITER"

#------------------------------------------------------------------------------
# Create HuggingFace Configuration
#------------------------------------------------------------------------------
echo "Creating HuggingFace configuration..."

cat <<EOF > ${{TARGET_HF_PATH_ITER}}/config.json
{{
    "_name_or_path": "",
    "architectures": [
      "$ARCHITECTURE"
    ],
    "attention_bias": true,
    "attention_dropout": 0.0,
    "auto_map": {{
        "AutoConfig": "configuration_opensci.OpensciConfig",
        "AutoModel": "modeling_opensci.OpensciPreTrainedModel",
        "AutoModelForCausalLM": "modeling_opensci.OpensciForCausalLM"
      }},
    "bos_token_id": 0,
    "eos_token_id": 0,
    "head_dim": $(($HIDDEN_SIZE / $NUM_ATTN_HEADS)),
    "hidden_act": "silu",
    "hidden_size": $HIDDEN_SIZE,
    "initializer_range": 0.02,
    "intermediate_size": $INTERMEDIATE_SIZE,
    "max_position_embeddings": $MAX_POSITION_EMBEDDINGS,
    "mlp_bias": true,
    "model_type": "$MODEL_TYPE",
    "num_attention_heads": $NUM_ATTN_HEADS,
    "num_hidden_layers": $NUM_LAYERS,
    "num_key_value_heads": $NUM_KEY_VALUE_HEADS,
    "pretraining_tp": 1,
    "qk_layernorm": true,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.3",
    "use_cache": true,
    "vocab_size": $VOCAB_SIZE
  }}
EOF

#------------------------------------------------------------------------------
# Copy Model and Tokenizer Files
#------------------------------------------------------------------------------
echo "Copying model and tokenizer files..."

# Check if Open-Sci-hf path exists
if [[ ! -d "$OPEN_SCI_HF_PATH" ]]; then
    echo "ERROR: Open-Sci-hf path not found: $OPEN_SCI_HF_PATH"
    exit 1
fi

# Copy model files
cp -r "$OPEN_SCI_HF_PATH/sample/modeling_opensci.py" "${{TARGET_HF_PATH_ITER}}/"
cp -r "$OPEN_SCI_HF_PATH/sample/configuration_opensci.py" "${{TARGET_HF_PATH_ITER}}/"

# Copy tokenizer files
cp -r "$OPEN_SCI_HF_PATH/sample/tokenizer"* "${{TARGET_HF_PATH_ITER}}/"
cp -r "$OPEN_SCI_HF_PATH/sample/special_tokens_map.json" "${{TARGET_HF_PATH_ITER}}/"
cp -r "$OPEN_SCI_HF_PATH/sample/vocab.json" "${{TARGET_HF_PATH_ITER}}/"

#------------------------------------------------------------------------------
# Run Conversion
#------------------------------------------------------------------------------
echo "Starting checkpoint conversion..."

# Set up the conversion command
CONVERSION_CMD="scripts/ckpt/mcore_to_hf_opensci.py \
    --load_path $LOAD_CHECKPOINT_PATH \
    --save_path ${{TARGET_HF_PATH_ITER}} \
    --source_model ${{TARGET_HF_PATH_ITER}} \
    --target_tensor_model_parallel_size $TARGET_TP_SIZE \
    --target_pipeline_model_parallel_size $TARGET_PP_SIZE \
    --target_params_dtype 'bf16' \
    --world_size $WORLD_SIZE \
    --convert_checkpoint_from_megatron_to_transformers \
    --num_key_value_heads $NUM_KEY_VALUE_HEADS \
    --print-checkpoint-structure"

# Set up container launcher
LAUNCHER="singularity exec \
    --nv \
    $IMAGE \
    python -u"

# Set Megatron path and run conversion
MEGATRON_PATH="/p/data1/mmlaion/shared/repos/Megatron-LM-Open-Sci"

# Critical: Add Megatron to Python path (missing from original template!)
export PYTHONPATH="$MEGATRON_PATH:${{PYTHONPATH:-}}"

# Validate paths exist before proceeding
if [[ ! -d "$MEGATRON_PATH" ]]; then
    echo "ERROR: Megatron path not found: $MEGATRON_PATH"
    exit 1
fi

cd $MEGATRON_PATH

echo "Running conversion command:"
echo "$LAUNCHER $CONVERSION_CMD"

$LAUNCHER $CONVERSION_CMD

CONVERSION_EXIT_CODE=$?

if [[ $CONVERSION_EXIT_CODE -eq 0 ]]; then
    echo "Checkpoint conversion completed successfully!"
    echo "Converted model saved to: ${{TARGET_HF_PATH_ITER}}"
    
    #------------------------------------------------------------------------------
    # Run Evaluation
    #------------------------------------------------------------------------------
    echo "Starting evaluation..."
    
    EVAL_OUTPUT_DIR="{RUN_DIR}/eval_results"
    mkdir -p "$EVAL_OUTPUT_DIR"
    
    # Set up evaluation environment
    export HF_HOME="{HF_HOME}"
    export HF_HUB_OFFLINE="1"
    
    EVAL_CMD="accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${{TARGET_HF_PATH_ITER}},trust_remote_code=True \
    --include_path {PROJECT_DIR_PATH}/eval \
    --tasks oellm-core \
    --output_path $EVAL_OUTPUT_DIR/iter_$TARGET_ITER/results.json \
    --batch_size auto \
    --trust_remote_code"

    LAUNCHER_EVAL="singularity exec --nv $IMAGE"

    echo "Running evaluation command:"
    echo "$LAUNCHER_EVAL $EVAL_CMD"

    $LAUNCHER_EVAL $EVAL_CMD

    EVAL_EXIT_CODE=$?
    
    if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
        echo "Evaluation completed successfully!"
        echo "Results saved to: $EVAL_OUTPUT_DIR/iter_$TARGET_ITER/"
    else
        echo "WARNING: Evaluation failed with exit code $EVAL_EXIT_CODE"
        # Don't fail the whole job if eval fails
    fi
    
    # Mark checkpoint as successfully converted
    rm -f "$IN_PROGRESS_TRACKING_DIR/iter_$TARGET_ITER.lock"
    touch "$CONVERTED_TRACKING_DIR/iter_$TARGET_ITER.done"
    echo "Conversion and evaluation completed for iteration $TARGET_ITER"
    else
        echo "ERROR: Checkpoint conversion failed with exit code $CONVERSION_EXIT_CODE"
        # Remove in-progress lock on failure
        rm -f "$IN_PROGRESS_TRACKING_DIR/iter_$TARGET_ITER.lock"
        exit $CONVERSION_EXIT_CODE
    fi
done  # End of while loop for processing all checkpoints

echo "All checkpoint conversions completed"