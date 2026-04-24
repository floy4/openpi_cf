#!/bin/bash
# Run CF Attention evaluation on LIBERO benchmark
# 
# Usage:
#   ./run_cf_attn_libero.sh                              # Run with default settings (CF mode E, libero_spatial)
#   ./run_cf_attn_libero.sh --cf_mode F                  # Run with CF mode F
#   ./run_cf_attn_libero.sh --task_suite_names libero_object  # Run on libero_object suite
#   ./run_cf_attn_libero.sh --task_suite_names "libero_10,libero_goal,libero_object"  # Run multiple suites
#
# Available CF modes:
#   BASE: No counterfactual reweighting
#   A: Reweight image tokens only
#   B: Reweight text tokens only  
#   C: Reweight state tokens only
#   D: Reweight image + text tokens
#   E: Reweight image + state tokens
#   F: Reweight text + state tokens
#
# Available task suites:
#   libero_spatial: Spatial reasoning tasks
#   libero_object: Object manipulation tasks
#   libero_goal: Goal-oriented tasks
#   libero_10: 10-task benchmark
#   libero_90: 90-task benchmark
#
# Environment variables:
#   CHECKPOINT_DIR: Model checkpoint directory
#   CF_MODE: Counterfactual mode (BASE, A-F)
#   TASK_SUITES: Comma-separated list of task suites
#   NUM_TRIALS: Number of trials per task
#   GPU_ID: GPU device ID
#   OUTPUT_DIR: Output directory for results

set -e

# Default configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data4/zhy/models/openpi-assets/checkpoints/pi05_libero}"
CF_MODE="${CF_MODE:-E}"
TASK_SUITES="${TASK_SUITES:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cf_attn_libero}"

echo "=========================================="
echo "CF Attention LIBERO Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "CF Mode: $CF_MODE"
echo "Task Suites: $TASK_SUITES"
echo "Num Trials: $NUM_TRIALS"
echo "GPU ID: $GPU_ID"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

python scripts/eval_cf_attn_libero.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cf_mode "$CF_MODE" \
    --task_suite_names "$TASK_SUITES" \
    --num_trials_per_task "$NUM_TRIALS" \
    --gpu_id "$GPU_ID" \
    --video_out_path "$OUTPUT_DIR" \
    "$@"
