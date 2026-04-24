#!/usr/bin/env python3
"""Evaluate CF Attention Level intervention with reweighting on LIBERO benchmark.

This script evaluates the attention-level counterfactual reweighting on LIBERO tasks:
- Three forward passes: BASE, NO_IMAGE, NO_STATE
- Compute effects: effect_image, effect_state
- Reweight baseline: actions_final = baseline + guidance * delta_image + adaptive * delta_state

Attention-level intervention blocks attention from action tokens to specific modalities,
providing a more theoretically pure counterfactual analysis compared to input-level intervention.

Usage:
    python scripts/eval_cf_attn_level_libero.py \
        --checkpoint_dir /data4/zhy/models/openpi-assets/checkpoints/pi05_libero \
        --task_suite_names libero_spatial

    # Adjust reweighting parameters
    python scripts/eval_cf_attn_level_libero.py \
        --cf_guidance_scale 0.2 \
        --state_weight_base 0.1

Output:
    - Success rate statistics
    - Rollout videos
    - CF effect analysis data
"""

# Setup logging BEFORE any other imports to capture all messages
import logging
import os
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
# Add file handler that flushes immediately
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
file_handler = FlushFileHandler("run.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)


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

import collections
import dataclasses
import json
import logging
import pathlib
import time

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import tqdm
import tyro

from openpi.models import model as _model
from openpi.models.cf_attention import CfSampler, ModalityBounds
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SIGLIP_IMAGE_TOKENS = 256  # SigLIP So400m/14 on 224x224 produces 256 tokens


@dataclasses.dataclass
class Args:
    """Arguments for CF Attention Level reweighting evaluation on LIBERO."""

    # Model configuration
    checkpoint_dir: str = "/data4/zhy/models/openpi-assets/checkpoints/pi05_libero"
    config_name: str = "pi05_libero"

    # CF Reweighting configuration
    cf_guidance_scale: float = 0.1  # Guidance scale for image effect
    state_weight_base: float = 0.05  # Base weight for state effect
    use_state_adaptive: bool = True  # Adaptive state weight based on effect ratio
    effect_threshold: float = 0.5  # Threshold to fallback to baseline
    sampling_num_steps: int = 10  # Flow-matching sampling steps per forward pass
    cf_inference_interval: int = 1  # Run CF every N replans; 1 means every replan
    log_replan_timing: bool = True  # Log end-to-end replanning latency

    # LIBERO environment parameters
    task_suite_name: str = ""
    task_suite_names: str = "libero_spatial"
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    max_tasks: int = 0  # 0 means evaluate all tasks in the selected suite

    # Output configuration
    video_out_path: str = "data/cf_attn_level_libero"
    save_videos: bool = True
    save_cf_effects: bool = True

    # Random seed
    seed: int = 7

    # GPU configuration
    gpu_id: int = 0

    def get_task_suites(self) -> list[str]:
        if self.task_suite_name:
            return [self.task_suite_name]
        return [s.strip() for s in self.task_suite_names.split(",") if s.strip()]


@dataclasses.dataclass
class EpisodeResult:
    """Result of a single episode."""
    task_id: int
    task_description: str
    episode_idx: int
    success: bool
    total_steps: int
    cf_effects: dict
    replay_images: list = dataclasses.field(default_factory=list)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "episode_idx": self.episode_idx,
            "success": self.success,
            "total_steps": self.total_steps,
            "cf_effects": self.cf_effects,
        }


def create_cf_sampler_policy(args: Args) -> tuple[_policy.Policy, CfSampler]:
    """Create a policy and wrap it with CfSampler for attention-level CF."""
    logging.info(f"Loading model config: {args.config_name}")
    train_config = _config.get_config(args.config_name)

    logging.info(f"Loading checkpoint from: {args.checkpoint_dir}")
    logging.info("This may take several minutes for large models...")

    # Create base policy without CF sampling
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        use_cf_sampling=False,
    )

    logging.info("Model loaded successfully!")
    logging.info("Creating CF sampler...")

    # Wrap model with CfSampler
    cf_sampler = CfSampler(
        policy._model,
        default_num_steps=args.sampling_num_steps,
        default_action_dim=train_config.model.action_dim,
        default_action_horizon=train_config.model.action_horizon,
    )

    logging.info("CF sampler ready. Starting evaluation...")
    return policy, cf_sampler


def compute_modality_bounds_for_obs(obs_dict: dict, prompt_length: int = 50) -> ModalityBounds:
    """Compute modality bounds from observation dictionary."""
    # Count cameras
    image_bounds = {}
    current_pos = 0

    # Agent view image
    if "observation/image" in obs_dict:
        image_bounds["base_0_rgb"] = (current_pos, current_pos + SIGLIP_IMAGE_TOKENS)
        current_pos += SIGLIP_IMAGE_TOKENS

    # Wrist image
    if "observation/wrist_image" in obs_dict:
        image_bounds["wrist_0_rgb"] = (current_pos, current_pos + SIGLIP_IMAGE_TOKENS)
        current_pos += SIGLIP_IMAGE_TOKENS

    image_offset = current_pos

    # Language bounds (approximate, based on prompt length)
    # For Pi05: "Task: {prompt}, State: {state};\nAction: "
    language_start = image_offset + 4  # "Task: " prefix
    language_end = language_start + prompt_length
    current_pos = language_end

    # State bounds (for Pi05, state is discretized in tokens)
    # Approximate: state_dim tokens after ", State: " prefix
    state_start = current_pos + 8  # ", State: " prefix
    state_dim = obs_dict.get("observation/state", np.zeros(7)).shape[0] if hasattr(obs_dict.get("observation/state", np.zeros(7)), 'shape') else 7
    state_end = state_start + state_dim
    current_pos = state_end

    # Prefix length includes action suffix tokens
    prefix_len = current_pos + 10  # ";\nAction: " suffix

    return ModalityBounds(
        image_bounds=image_bounds,
        language_bounds=(language_start, language_end),
        state_bounds=(state_start, state_end),
        prefix_len=prefix_len,
        suffix_start=prefix_len,
    )


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img, target_size):
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, target_size, target_size)
    )


def run_episode(
    env,
    task_description: str,
    policy: _policy.Policy,
    cf_sampler: CfSampler,
    args: Args,
    episode_idx: int,
    task_id: int,
    current_task_suite: str,
    rng: jax.Array,
) -> EpisodeResult:
    """Run a single episode with CF attention-level reweighting."""
    env.reset()
    action_plan = collections.deque()

    t = 0
    replan_count = 0
    cf_replan_count = 0
    base_only_replan_count = 0
    replay_images = []
    cf_effects_data = {"steps": [], "cumulative": {}}

    logging.info(f"Starting episode {episode_idx + 1} with CF attention-level reweighting")

    while t < get_max_steps(current_task_suite) + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = resize_image(img, args.resize_size)
            wrist_img = resize_image(wrist_img, args.resize_size)

            if args.save_videos:
                replay_images.append(img)

            if not action_plan:
                replan_count += 1
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                # Apply input transform to prepare observation for model
                inputs = policy._input_transform(element)
                # Make batch and convert to jax.Array
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
                observation = _model.Observation.from_dict(inputs)

                rng, sub_rng = jax.random.split(rng)
                use_cf_this_replan = (
                    args.cf_inference_interval <= 1
                    or ((replan_count - 1) % args.cf_inference_interval == 0)
                )

                replan_start = time.perf_counter()

                if use_cf_this_replan:
                    cf_replan_count += 1

                    # Compute modality bounds
                    modality_bounds = compute_modality_bounds_for_obs(
                        element,
                        prompt_length=len(task_description.split())
                    )

                    # Sample with CF attention-level reweighting
                    actions, metrics = cf_sampler.sample_with_cf_reweight(
                        sub_rng,
                        observation,
                        num_steps=args.sampling_num_steps,
                        modality_bounds=modality_bounds,
                        cf_guidance_scale=args.cf_guidance_scale,
                        state_weight_base=args.state_weight_base,
                        use_state_adaptive=args.use_state_adaptive,
                        effect_threshold=args.effect_threshold,
                        return_metrics=True,
                    )

                    # Print CF metrics for debugging
                    logging.info(f"CF metrics at step {t}:")
                    logging.info(f"  effect_image: {metrics['effect_image']:.4f}")
                    logging.info(f"  effect_state: {metrics['effect_state']:.4f}")
                    logging.info(f"  state_weight: {metrics['state_weight']:.4f}")
                    logging.info(f"  use_baseline: {metrics['use_baseline']}")
                else:
                    base_only_replan_count += 1
                    actions = cf_sampler.sample_baseline(
                        sub_rng,
                        observation,
                        num_steps=args.sampling_num_steps,
                    )
                    metrics = {
                        "effect_image": None,
                        "effect_state": None,
                        "state_weight": None,
                        "use_baseline": True,
                        "cf_guidance_scale": args.cf_guidance_scale,
                        "cf_skipped": True,
                    }

                replan_elapsed_s = time.perf_counter() - replan_start
                if args.log_replan_timing:
                    replan_mode = "cf" if use_cf_this_replan else "base_only"
                    logging.info(f"Replan timing at step {t}: {replan_elapsed_s:.3f}s (mode={replan_mode})")

                # Apply output transform (unnormalize)
                outputs = {"state": inputs["state"], "actions": actions}
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
                outputs = policy._output_transform(outputs)
                action_chunk = outputs["actions"]

                # Save CF effects
                if args.save_cf_effects:
                    effects_info = {
                        "step": t,
                        "replan_seconds": replan_elapsed_s,
                        "metrics": metrics,
                    }
                    cf_effects_data["steps"].append(effects_info)

                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())

            if done:
                cf_effects_data["cumulative"] = {
                    "replan_count": replan_count,
                    "cf_replan_count": cf_replan_count,
                    "base_only_replan_count": base_only_replan_count,
                }
                return EpisodeResult(
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    success=True,
                    total_steps=t - args.num_steps_wait,
                    cf_effects=cf_effects_data,
                    replay_images=replay_images,
                )

            t += 1

        except Exception as e:
            logging.error(f"Episode error: {e}")
            import traceback
            traceback.print_exc()
            break

    cf_effects_data["cumulative"] = {
        "replan_count": replan_count,
        "cf_replan_count": cf_replan_count,
        "base_only_replan_count": base_only_replan_count,
    }
    return EpisodeResult(
        task_id=task_id,
        task_description=task_description,
        episode_idx=episode_idx,
        success=False,
        total_steps=t - args.num_steps_wait if t > args.num_steps_wait else 0,
        cf_effects=cf_effects_data,
        replay_images=replay_images,
    )


def get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    elif task_suite_name == "libero_object":
        return 280
    elif task_suite_name == "libero_goal":
        return 300
    elif task_suite_name == "libero_10":
        return 520
    elif task_suite_name == "libero_90":
        return 400
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")


def save_episode_video(video_path: pathlib.Path, images: list, fps: int = 10):
    imageio.mimwrite(video_path, [np.asarray(x) for x in images], fps=fps)


def eval_single_task_suite(
    args: Args,
    task_suite_name: str,
    policy: _policy.Policy,
    cf_sampler: CfSampler,
    rng: jax.Array,
) -> dict:
    """Evaluate a single task suite with CF attention-level reweighting."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Task suite: {task_suite_name}")
    logging.info(f"CF guidance scale: {args.cf_guidance_scale}")
    logging.info(f"State weight base: {args.state_weight_base}")
    logging.info(f"Intervention type: Attention-level reweighting")
    logging.info(f"{'='*60}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.max_tasks > 0:
        num_tasks_to_run = min(num_tasks_in_suite, args.max_tasks)
    else:
        num_tasks_to_run = num_tasks_in_suite

    output_dir = pathlib.Path(args.video_out_path) / f"cf_attn_reweight_{task_suite_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    if args.save_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    total_episodes, total_successes = 0, 0
    all_episode_results = []

    logging.info(f"Tasks in suite: {num_tasks_in_suite}, tasks to run: {num_tasks_to_run}")

    for task_id in tqdm.tqdm(range(num_tasks_to_run)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        logging.info(f"Task: {task_description}")

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            rng, sub_rng = jax.random.split(rng)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            logging.info(f"Starting episode {episode_idx + 1}...")

            result = run_episode(
                env, task_description, policy, cf_sampler, args,
                episode_idx, task_id, task_suite_name, sub_rng
            )
            all_episode_results.append(result)

            if result.success:
                task_successes += 1
                total_successes += 1

            task_episodes += 1
            total_episodes += 1

            if args.save_videos and result.replay_images:
                suffix = "success" if result.success else "failure"
                task_segment = task_description.replace(" ", "_")
                video_path = videos_dir / f"task{task_id}_ep{episode_idx}_{suffix}.mp4"
                save_episode_video(video_path, result.replay_images)

            logging.info(f"Success: {result.success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"\nTask {task_id} final success rate: {task_successes / task_episodes * 100:.1f}%")

    final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

    # Save results
    results_file = output_dir / "results.txt"
    results_file.write_text(
        f"Intervention Type: Attention-level reweighting\n"
        f"Task Suite: {task_suite_name}\n"
        f"Total episodes: {total_episodes}\n"
        f"Total successes: {total_successes}\n"
        f"Success rate: {final_success_rate:.4f}\n"
        f"Checkpoint: {args.checkpoint_dir}\n"
        f"CF guidance scale: {args.cf_guidance_scale}\n"
        f"State weight base: {args.state_weight_base}\n"
        f"Sampling num steps: {args.sampling_num_steps}\n"
        f"CF inference interval: {args.cf_inference_interval}\n"
        f"Max tasks: {args.max_tasks}\n"
        f"Use state adaptive: {args.use_state_adaptive}\n"
        f"Effect threshold: {args.effect_threshold}\n"
    )

    # Save detailed effects
    if args.save_cf_effects:
        effects_file = output_dir / "cf_attn_reweight_effects.json"
        effects_data = {
            "config": {
                "intervention_type": "attention_level_reweight",
                "task_suite": task_suite_name,
                "cf_guidance_scale": args.cf_guidance_scale,
                "state_weight_base": args.state_weight_base,
                "sampling_num_steps": args.sampling_num_steps,
                "cf_inference_interval": args.cf_inference_interval,
                "max_tasks": args.max_tasks,
                "use_state_adaptive": args.use_state_adaptive,
                "effect_threshold": args.effect_threshold,
            },
            "summary": {
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "success_rate": final_success_rate,
            },
            "episodes": [r.to_dict() for r in all_episode_results],
        }
        effects_file.write_text(json.dumps(effects_data, indent=2))

    logging.info(f"\n{'='*60}")
    logging.info(f"Task suite evaluation complete!")
    logging.info(f"Task Suite: {task_suite_name}")
    logging.info(f"Success rate: {final_success_rate * 100:.1f}%")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"{'='*60}")

    return {
        "success_rate": final_success_rate,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "output_dir": str(output_dir),
    }


def eval_cf_attn_level_libero(args: Args) -> dict:
    """Evaluate CF attention-level reweighting on LIBERO benchmark."""
    np.random.seed(args.seed)

    task_suites = args.get_task_suites()
    logging.info(f"Will evaluate {len(task_suites)} task suites: {task_suites}")
    logging.info(f"Intervention type: Attention-level reweighting")
    logging.info(f"CF guidance scale: {args.cf_guidance_scale}")
    logging.info(f"State weight base: {args.state_weight_base}")
    logging.info(f"Sampling num steps: {args.sampling_num_steps}")
    logging.info(f"CF inference interval: {args.cf_inference_interval}")
    logging.info(f"Max tasks: {args.max_tasks}")
    logging.info(f"Requested gpu_id arg: {args.gpu_id}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    policy, cf_sampler = create_cf_sampler_policy(args)
    rng = jax.random.PRNGKey(args.seed)

    all_results = {}
    total_episodes_all = 0
    total_successes_all = 0

    start_time = time.time()

    for task_suite_name in task_suites:
        logging.info(f"\n{'#'*60}")
        logging.info(f"Evaluating task suite: {task_suite_name}")
        logging.info(f"{'#'*60}")

        rng, suite_rng = jax.random.split(rng)
        result = eval_single_task_suite(args, task_suite_name, policy, cf_sampler, suite_rng)
        all_results[task_suite_name] = result
        total_episodes_all += result["total_episodes"]
        total_successes_all += result["total_successes"]

    elapsed_time = time.time() - start_time
    overall_success_rate = total_successes_all / total_episodes_all if total_episodes_all > 0 else 0.0

    summary_dir = pathlib.Path(args.video_out_path) / "cf_attn_reweight_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_file = summary_dir / "overall_results.txt"
    summary_content = (
        f"Intervention Type: Attention-level reweighting\n"
        f"Task Suites: {', '.join(task_suites)}\n"
        f"Total episodes: {total_episodes_all}\n"
        f"Total successes: {total_successes_all}\n"
        f"Overall success rate: {overall_success_rate:.4f}\n"
        f"Total evaluation time: {elapsed_time / 60:.1f} minutes\n"
        f"Checkpoint: {args.checkpoint_dir}\n"
        f"CF guidance scale: {args.cf_guidance_scale}\n"
        f"State weight base: {args.state_weight_base}\n"
        f"Sampling num steps: {args.sampling_num_steps}\n"
        f"CF inference interval: {args.cf_inference_interval}\n"
        f"Max tasks: {args.max_tasks}\n"
        f"\n--- Per Task Suite Results ---\n"
    )
    for suite_name, result in all_results.items():
        summary_content += f"{suite_name}: {result['success_rate'] * 100:.1f}% ({result['total_successes']}/{result['total_episodes']})\n"
    summary_file.write_text(summary_content)

    logging.info(f"\n{'='*60}")
    logging.info(f"ALL EVALUATIONS COMPLETE!")
    logging.info(f"{'='*60}")
    logging.info(f"Intervention Type: Attention-level reweighting")
    logging.info(f"Overall success rate: {overall_success_rate * 100:.1f}%")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info(f"{'='*60}")

    return {
        "overall_success_rate": overall_success_rate,
        "total_episodes": total_episodes_all,
        "total_successes": total_successes_all,
        "task_suite_results": all_results,
        "summary_file": str(summary_file),
    }


def main():
    args = tyro.cli(Args)
    results = eval_cf_attn_level_libero(args)


if __name__ == "__main__":
    main()