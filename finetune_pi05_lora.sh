#!/bin/bash

# Fine-tune pi05_droid model on pick_up_four_cubes dataset using LoRA
# LoRA (Low-Rank Adaptation) allows efficient fine-tuning with much lower memory usage
# Memory requirement: ~22-30GB (vs ~60-80GB for full fine-tuning)

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
echo "Activating yam conda environment..."
source /home/prior/anaconda3/etc/profile.d/conda.sh
conda activate yam

# Configuration
PRETRAINED_MODEL="$SCRIPT_DIR/pi05_pytorch"
DATASET_PATH="$SCRIPT_DIR/datasets/pick_up_four_cubes_and_stack_them_in_the_middle-v3.0"
OUTPUT_DIR="$SCRIPT_DIR/outputs/finetune_pi05_lora_$(date +%Y%m%d_%H%M%S)"

# LoRA Configuration
LORA_RANK=16              # Rank of LoRA matrices (higher = more parameters, better performance)
LORA_ALPHA=32             # LoRA scaling factor (usually 2x rank)
LORA_DROPOUT=0.1          # Dropout for LoRA layers
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # Which layers to apply LoRA

# Training parameters - can use larger batch size with LoRA
BATCH_SIZE=4              # Can use 4-8 with LoRA (vs 1 for full fine-tuning)
STEPS=2000
EVAL_FREQ=500
LOG_FREQ=100
SAVE_FREQ=500

echo "========================================"
echo "Fine-tuning pi05 model with LoRA"
echo "========================================"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Dataset: $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "LoRA Config: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "Target modules: $LORA_TARGET_MODULES"
echo "========================================"

# Run training with LoRA
python src/lerobot/scripts/lerobot_train.py \
    --policy.path="$PRETRAINED_MODEL" \
    --policy.repo_id="finetune_pi05_lora" \
    --policy.use_lora=true \
    --policy.lora_rank=$LORA_RANK \
    --policy.lora_alpha=$LORA_ALPHA \
    --policy.lora_dropout=$LORA_DROPOUT \
    --policy.lora_target_modules="$LORA_TARGET_MODULES" \
    --dataset.repo_id="$DATASET_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --log_freq=$LOG_FREQ \
    --save_freq=$SAVE_FREQ \
    --save_checkpoint=true \
    --seed=1000 \
    --num_workers=4

echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
