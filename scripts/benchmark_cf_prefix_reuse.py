#!/usr/bin/env python3
"""Micro-benchmark old vs optimized CF reweight sampling paths.

This benchmark isolates inference-side sampling cost on one synthetic observation.
It compares:
1) old-like path: three mode samples without shared prefix context
2) optimized path: sample_with_cf_reweight with shared prefix context
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.cf_attention import CfSampler
from openpi.models.cf_attention.attention_mask import CfAttnMode
from openpi.training import config as _config


def _old_like_cf_path(
    cf_sampler: CfSampler,
    rng: jax.Array,
    observation: _model.Observation,
    num_steps: int,
) -> jax.Array:
    """Emulate prior CF path without shared prefix context."""
    modality_bounds = cf_sampler._compute_modality_bounds(observation)
    _, r1, r2, r3 = jax.random.split(rng, 4)

    actions_base = cf_sampler._sample_single_mode(
        r1,
        observation,
        CfAttnMode.BASE,
        modality_bounds,
        num_steps,
        prefix_context=None,
    )
    actions_no_image = cf_sampler._sample_single_mode(
        r2,
        observation,
        CfAttnMode.NO_IMAGE,
        modality_bounds,
        num_steps,
        prefix_context=None,
    )
    actions_no_state = cf_sampler._sample_single_mode(
        r3,
        observation,
        CfAttnMode.NO_STATE,
        modality_bounds,
        num_steps,
        prefix_context=None,
    )

    # Same algebraic shape as CF reweighting, only for fair timing output materialization.
    delta_image = actions_base - actions_no_image
    delta_state = actions_base - actions_no_state
    return actions_base + 0.1 * delta_image + 0.05 * jnp.clip(delta_state, -0.1, 0.1)


def _optimized_cf_path(
    cf_sampler: CfSampler,
    rng: jax.Array,
    observation: _model.Observation,
    num_steps: int,
) -> jax.Array:
    actions, _ = cf_sampler.sample_with_cf_reweight(
        rng,
        observation,
        num_steps=num_steps,
        return_metrics=True,
    )
    return actions


def _benchmark(
    fn,
    key: jax.Array,
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    keys = jax.random.split(key, warmup + iters)

    for i in range(warmup):
        out = fn(keys[i])
        jax.block_until_ready(out)

    timings_ms: list[float] = []
    for i in range(warmup, warmup + iters):
        t0 = time.perf_counter()
        out = fn(keys[i])
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    return timings_ms


def _summarize(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values) if values else float("nan"),
        "median_ms": statistics.median(values) if values else float("nan"),
        "min_ms": min(values) if values else float("nan"),
        "max_ms": max(values) if values else float("nan"),
        "all_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config_name", default="pi05_libero")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    model = train_config.model.load(
        _model.restore_params(pathlib.Path(args.checkpoint_dir) / "params", dtype=jnp.bfloat16)
    )

    cf_sampler = CfSampler(
        model,
        default_num_steps=args.num_steps,
        default_action_dim=train_config.model.action_dim,
        default_action_horizon=train_config.model.action_horizon,
    )

    observation = train_config.model.fake_obs(batch_size=1)

    old_timings = _benchmark(
        lambda key: _old_like_cf_path(cf_sampler, key, observation, args.num_steps),
        jax.random.PRNGKey(args.seed),
        warmup=args.warmup,
        iters=args.iters,
    )

    optimized_timings = _benchmark(
        lambda key: _optimized_cf_path(cf_sampler, key, observation, args.num_steps),
        jax.random.PRNGKey(args.seed + 1),
        warmup=args.warmup,
        iters=args.iters,
    )

    old_summary = _summarize(old_timings)
    optimized_summary = _summarize(optimized_timings)

    speedup = old_summary["mean_ms"] / optimized_summary["mean_ms"]

    result = {
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "num_steps": args.num_steps,
        "warmup": args.warmup,
        "iters": args.iters,
        "old_like": old_summary,
        "optimized": optimized_summary,
        "speedup_x": speedup,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = pathlib.Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
