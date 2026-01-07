#!/bin/bash

# Fine-tune pi05_droid model on pick_up_four_cubes dataset
# This script uses lerobot-train to fine-tune the pretrained pi05_droid model

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
DATASET_REPO_ID="thomas0829/pick_up_four_cubes_and_stack_them_in_the_middle-v3.0"
OUTPUT_DIR="$SCRIPT_DIR/outputs/finetune_pi05_$(date +%Y%m%d_%H%M%S)"

# Set CUDA memory optimization to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training parameters - using very small batch size to fit in GPU memory
BATCH_SIZE=1
STEPS=10000
EVAL_FREQ=500
LOG_FREQ=100
SAVE_FREQ=500

echo "========================================"
echo "Fine-tuning pi05 model"
echo "========================================"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Dataset: $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Run training with pretrained model and memory optimizations
python src/lerobot/scripts/lerobot_train.py \
    --policy.path="$PRETRAINED_MODEL" \
    --policy.repo_id="thomas0829/finetune_pi05" \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=$BATCH_SIZE \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --log_freq=$LOG_FREQ \
    --save_freq=$SAVE_FREQ \
    --save_checkpoint=true \
    --seed=1000 \
    --num_workers=4 \
    --wandb.enable=false

echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
