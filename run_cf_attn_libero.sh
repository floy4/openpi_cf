#!/bin/bash
# Run CF Attention-Level Reweighting evaluation on LIBERO benchmark
#
# Attention-level intervention with reweighting:
# - Three forward passes: BASE, NO_IMAGE, NO_STATE
# - Compute effects: effect_image, effect_state
# - Reweight baseline: actions_final = baseline + guidance * delta_image + adaptive * delta_state
#
# This provides theoretically pure counterfactual analysis at the attention level.
#
# Usage:
#   ./run_cf_attn_libero.sh                              # Run with default settings
#   ./run_cf_attn_libero.sh --cf_guidance_scale 0.2      # Increase image guidance
#   ./run_cf_attn_libero.sh --state_weight_base 0.1      # Increase state weight
#   ./run_cf_attn_libero.sh --task_suite_names libero_object  # Run on libero_object
#
# Available task suites:
#   libero_spatial: Spatial reasoning tasks
#   libero_object: Object manipulation tasks
#   libero_goal: Goal-oriented tasks
#   libero_10: 10-task benchmark
#   libero_90: 90-task benchmark
#
# Reweighting parameters:
#   cf_guidance_scale: Weight for image effect delta (default 0.1)
#   state_weight_base: Base weight for state effect delta (default 0.05)
#   use_state_adaptive: Use adaptive state weight based on effect ratio (default True)
#   effect_threshold: Threshold to fallback to baseline (default 0.5)
#
# Environment variables:
#   CHECKPOINT_DIR: Model checkpoint directory
#   CF_GUIDANCE_SCALE: Image guidance scale
#   STATE_WEIGHT_BASE: State weight base
#   TASK_SUITES: Comma-separated list of task suites
#   NUM_TRIALS: Number of trials per task
#   GPU_ID: GPU device ID
#   OUTPUT_DIR: Output directory for results
#   SAMPLING_STEPS: Flow-matching steps per sampling call
#   CF_INTERVAL: Run CF once every N replans (1 means every replan)
#   LOG_REPLAN_TIMING: Whether to print replan latency logs (True/False)
#   MAX_TASKS: Max tasks per suite (0 means all tasks)

set -e

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Default configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data4/zhy/models/openpi-assets/checkpoints/pi05_libero}"
CF_GUIDANCE_SCALE="${CF_GUIDANCE_SCALE:-0.1}"
STATE_WEIGHT_BASE="${STATE_WEIGHT_BASE:-0.05}"
TASK_SUITES="${TASK_SUITES:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-50}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cf_attn_level_libero}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"
CF_INTERVAL="${CF_INTERVAL:-1}"
LOG_REPLAN_TIMING="${LOG_REPLAN_TIMING:-True}"
MAX_TASKS="${MAX_TASKS:-0}"

# Ensure device selection is applied before Python/JAX imports.
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=========================================="
echo "CF Attention-Level Reweighting LIBERO"
echo "=========================================="
echo "Intervention Type: Attention-level reweighting"
echo "  - Three passes: BASE, NO_IMAGE, NO_STATE"
echo "  - Reweight with effect deltas"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "CF Guidance Scale: $CF_GUIDANCE_SCALE"
echo "State Weight Base: $STATE_WEIGHT_BASE"
echo "Task Suites: $TASK_SUITES"
echo "Num Trials: $NUM_TRIALS"
echo "GPU ID: $GPU_ID"
echo "Output Dir: $OUTPUT_DIR"
echo "Sampling Steps: $SAMPLING_STEPS"
echo "CF Interval: $CF_INTERVAL"
echo "Log Replan Timing: $LOG_REPLAN_TIMING"
echo "Max Tasks: $MAX_TASKS"
echo "=========================================="

python scripts/eval_cf_attn_level_libero.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --cf_guidance_scale "$CF_GUIDANCE_SCALE" \
    --state_weight_base "$STATE_WEIGHT_BASE" \
    --task_suite_names "$TASK_SUITES" \
    --num_trials_per_task "$NUM_TRIALS" \
    --gpu_id "$GPU_ID" \
    --video_out_path "$OUTPUT_DIR" \
    --sampling_num_steps "$SAMPLING_STEPS" \
    --cf_inference_interval "$CF_INTERVAL" \
    $(if [[ "$LOG_REPLAN_TIMING" == "True" ]]; then echo "--log_replan_timing"; fi) \
    --max_tasks "$MAX_TASKS" \
    "$@"