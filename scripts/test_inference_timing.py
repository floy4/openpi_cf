#!/usr/bin/env python3
"""测试不同推理模式的单步推理时间。

测试三种模式:
1. 基础推理 (sample_actions) - 单次前向推理
2. CF Attention-Level 推理 (sample_with_cf_reweight) - 三次前向推理 (BASE/NO_IMAGE/NO_STATE)
3. CF Input-Level 推理 (如果有实现) - 多次前向推理

Usage:
    python scripts/test_inference_timing.py --gpu_id 0 --num_runs 10
"""

import logging
import os
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


def _set_cuda_visible_devices_from_argv() -> None:
    """Set CUDA_VISIBLE_DEVICES from --gpu_id before importing JAX."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--gpu_id" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
            return
        if arg.startswith("--gpu_id="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_set_cuda_visible_devices_from_argv()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.models.cf_attention import CfSampler, ModalityBounds
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


SIGLIP_IMAGE_TOKENS = 256  # SigLIP So400m/14 on 224x224 produces 256 tokens


@dataclasses.dataclass
class Args:
    """Arguments for inference timing test."""

    checkpoint_dir: str = "/data4/zhy/models/openpi-assets/checkpoints/pi05_libero"
    config_name: str = "pi05_libero"
    gpu_id: int = 0
    num_runs: int = 10  # Number of runs for each mode
    warmup_runs: int = 3  # Warmup runs (not counted in timing)
    sampling_steps: int = 10  # Flow matching steps
    seed: int = 42


def create_fake_observation(batch_size: int = 1) -> _model.Observation:
    """Create a fake observation for testing."""
    # Fake images (224x224 RGB)
    fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    fake_image_float = fake_image.astype(np.float32) / 255.0 * 2.0 - 1.0

    # Fake state (7D for robot)
    fake_state = np.random.randn(7).astype(np.float32)

    # Fake prompt tokens
    fake_tokens = np.random.randint(0, 1000, (100,), dtype=np.int32)
    fake_token_mask = np.ones((100,), dtype=np.bool_)

    # Build observation dict
    obs_dict = {
        "image": {
            "base_0_rgb": jnp.asarray(fake_image_float)[np.newaxis, ...],
            "left_wrist_0_rgb": jnp.asarray(fake_image_float)[np.newaxis, ...],
            "right_wrist_0_rgb": jnp.asarray(fake_image_float)[np.newaxis, ...],
        },
        "image_mask": {
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        },
        "state": jnp.asarray(fake_state)[np.newaxis, ...],
        "tokenized_prompt": jnp.asarray(fake_tokens)[np.newaxis, ...],
        "tokenized_prompt_mask": jnp.asarray(fake_token_mask)[np.newaxis, ...],
    }

    return _model.Observation.from_dict(obs_dict)


def create_modality_bounds(observation: _model.Observation, prompt_length: int = 20) -> ModalityBounds:
    """Compute modality bounds from observation."""
    image_bounds = {}
    current_pos = 0

    # Each camera produces 256 tokens (SigLIP)
    for name in observation.images:
        image_bounds[name] = (current_pos, current_pos + SIGLIP_IMAGE_TOKENS)
        current_pos += SIGLIP_IMAGE_TOKENS

    image_offset = current_pos

    # Language bounds (approximate)
    language_start = image_offset
    language_end = language_start + prompt_length
    current_pos = language_end

    # State bounds
    state_dim = observation.state.shape[-1]
    state_start = current_pos
    state_end = state_start + state_dim
    current_pos = state_end

    prefix_len = current_pos + 10  # Action suffix tokens

    return ModalityBounds(
        image_bounds=image_bounds,
        language_bounds=(language_start, language_end),
        state_bounds=(state_start, state_end),
        prefix_len=prefix_len,
        suffix_start=prefix_len,
    )


def run_timing_tests(args: Args) -> dict:
    """Run timing tests for different inference modes."""
    logging.info(f"Loading model config: {args.config_name}")
    train_config = _config.get_config(args.config_name)

    logging.info(f"Loading checkpoint from: {args.checkpoint_dir}")
    logging.info("This may take several minutes...")

    # Create policy without CF sampling
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        use_cf_sampling=False,
    )

    logging.info("Model loaded successfully!")

    # Get model and create CF sampler
    model = policy._model
    cf_sampler = CfSampler(
        model,
        default_num_steps=args.sampling_steps,
        default_action_dim=train_config.model.action_dim,
        default_action_horizon=train_config.model.action_horizon,
    )

    logging.info("CF sampler created.")

    # Create fake observation
    observation = create_fake_observation(batch_size=1)
    modality_bounds = create_modality_bounds(observation)

    rng = jax.random.PRNGKey(args.seed)

    results = {}

    # ========================================
    # Test 1: Baseline inference (single forward)
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 1: Baseline inference (sample_actions)")
    logging.info("=" * 60)

    # Warmup
    logging.info(f"Warmup ({args.warmup_runs} runs)...")
    for i in range(args.warmup_runs):
        rng, sub_rng = jax.random.split(rng)
        _ = model.sample_actions(sub_rng, observation, num_steps=args.sampling_steps)
        jax.block_until_ready(_)

    # Timing
    logging.info(f"Timing ({args.num_runs} runs)...")
    baseline_times = []
    for i in range(args.num_runs):
        rng, sub_rng = jax.random.split(rng)
        start = time.perf_counter()
        actions = model.sample_actions(sub_rng, observation, num_steps=args.sampling_steps)
        jax.block_until_ready(actions)
        elapsed = time.perf_counter() - start
        baseline_times.append(elapsed)
        logging.info(f"  Run {i+1}: {elapsed:.4f}s")

    baseline_mean = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    results["baseline"] = {
        "mean_s": baseline_mean,
        "std_s": baseline_std,
        "times": baseline_times,
        "forward_passes": 1,
        "description": "Single forward pass with flow matching",
    }
    logging.info(f"\nBaseline timing: {baseline_mean:.4f}s ± {baseline_std:.4f}s")

    # ========================================
    # Test 2: CF Attention-Level Reweight (three forwards)
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 2: CF Attention-Level Reweight (sample_with_cf_reweight)")
    logging.info("  - Three forward passes: BASE, NO_IMAGE, NO_STATE")
    logging.info("  - Prefix shared when possible")
    logging.info("=" * 60)

    # Warmup
    logging.info(f"Warmup ({args.warmup_runs} runs)...")
    for i in range(args.warmup_runs):
        rng, sub_rng = jax.random.split(rng)
        _ = cf_sampler.sample_with_cf_reweight(
            sub_rng,
            observation,
            num_steps=args.sampling_steps,
            modality_bounds=modality_bounds,
            cf_guidance_scale=0.1,
            state_weight_base=0.05,
        )
        jax.block_until_ready(_)

    # Timing
    logging.info(f"Timing ({args.num_runs} runs)...")
    cf_attn_times = []
    for i in range(args.num_runs):
        rng, sub_rng = jax.random.split(rng)
        start = time.perf_counter()
        actions, metrics = cf_sampler.sample_with_cf_reweight(
            sub_rng,
            observation,
            num_steps=args.sampling_steps,
            modality_bounds=modality_bounds,
            cf_guidance_scale=0.1,
            state_weight_base=0.05,
            return_metrics=True,
        )
        jax.block_until_ready(actions)
        elapsed = time.perf_counter() - start
        cf_attn_times.append(elapsed)
        logging.info(f"  Run {i+1}: {elapsed:.4f}s (effect_image={metrics['effect_image']:.4f})")

    cf_attn_mean = np.mean(cf_attn_times)
    cf_attn_std = np.std(cf_attn_times)
    results["cf_attn_reweight"] = {
        "mean_s": cf_attn_mean,
        "std_s": cf_attn_std,
        "times": cf_attn_times,
        "forward_passes": 3,
        "description": "Three forward passes (BASE/NO_IMAGE/NO_STATE) with shared prefix",
    }
    logging.info(f"\nCF Attention-Level timing: {cf_attn_mean:.4f}s ± {cf_attn_std:.4f}s")

    # ========================================
    # Test 3: CF Analysis (full modes - may be more forwards)
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("Test 3: Full CF Analysis (sample_with_cf)")
    logging.info("  - Multiple forward passes for all modes")
    logging.info("=" * 60)

    from openpi.models.cf_attention.attention_mask import get_cf_modes_for_analysis

    cf_modes = get_cf_modes_for_analysis()
    num_cf_modes = len(cf_modes)
    logging.info(f"  CF modes: {[m.value for m in cf_modes]} ({num_cf_modes} modes)")

    # Warmup
    logging.info(f"Warmup ({args.warmup_runs} runs)...")
    for i in range(args.warmup_runs):
        rng, sub_rng = jax.random.split(rng)
        _ = cf_sampler.sample_with_cf(
            sub_rng,
            observation,
            cf_modes=cf_modes,
            num_steps=args.sampling_steps,
            modality_bounds=modality_bounds,
        )
        jax.block_until_ready(_.get_baseline_actions())

    # Timing
    logging.info(f"Timing ({args.num_runs} runs)...")
    cf_full_times = []
    for i in range(args.num_runs):
        rng, sub_rng = jax.random.split(rng)
        start = time.perf_counter()
        result = cf_sampler.sample_with_cf(
            sub_rng,
            observation,
            cf_modes=cf_modes,
            num_steps=args.sampling_steps,
            modality_bounds=modality_bounds,
        )
        jax.block_until_ready(result.get_baseline_actions())
        elapsed = time.perf_counter() - start
        cf_full_times.append(elapsed)
        logging.info(f"  Run {i+1}: {elapsed:.4f}s")

    cf_full_mean = np.mean(cf_full_times)
    cf_full_std = np.std(cf_full_times)
    results["cf_full_analysis"] = {
        "mean_s": cf_full_mean,
        "std_s": cf_full_std,
        "times": cf_full_times,
        "forward_passes": num_cf_modes,
        "description": f"Full CF analysis with {num_cf_modes} forward passes",
    }
    logging.info(f"\nFull CF Analysis timing: {cf_full_mean:.4f}s ± {cf_full_std:.4f}s")

    # ========================================
    # Summary
    # ========================================
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)

    for mode, data in results.items():
        logging.info(f"\n{mode}:")
        logging.info(f"  Mean: {data['mean_s']:.4f}s")
        logging.info(f"  Std:  {data['std_s']:.4f}s")
        logging.info(f"  Forward passes: {data['forward_passes']}")
        logging.info(f"  Time per forward: {data['mean_s'] / data['forward_passes']:.4f}s")
        logging.info(f"  Description: {data['description']}")

    # Compare with baseline
    logging.info("\n" + "-" * 40)
    logging.info("Comparison (relative to baseline):")
    logging.info("-" * 40)
    baseline = results["baseline"]["mean_s"]
    for mode in ["cf_attn_reweight", "cf_full_analysis"]:
        ratio = results[mode]["mean_s"] / baseline
        logging.info(f"  {mode}: {ratio:.2f}x baseline ({results[mode]['forward_passes']} forwards)")

    return results


def main():
    args = tyro.cli(Args)
    logging.info(f"GPU ID: {args.gpu_id}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    logging.info(f"Num runs: {args.num_runs}")
    logging.info(f"Warmup runs: {args.warmup_runs}")
    logging.info(f"Sampling steps: {args.sampling_steps}")

    results = run_timing_tests(args)

    # Save results
    import json
    import pathlib

    output_path = pathlib.Path("data/inference_timing_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for mode, data in results.items():
        serializable[mode] = {
            "mean_s": float(data["mean_s"]),
            "std_s": float(data["std_s"]),
            "forward_passes": data["forward_passes"],
            "description": data["description"],
        }

    output_path.write_text(json.dumps(serializable, indent=2))
    logging.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()