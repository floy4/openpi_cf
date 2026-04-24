import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class CfMode(enum.Enum):
    """Counterfactual reweighting mode for model.sample_actions_with_cf."""

    BASE = "base"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8888
    # WebSocket ping interval in seconds. Set to 0 or negative to disable ping.
    ws_ping_interval_s: float = 20.0
    # WebSocket ping timeout in seconds. Increase for slow inference.
    ws_ping_timeout_s: float = 120.0
    # Record the policy's behavior for debugging.
    record: bool = False

    # ==========================================================================
    # CF Mode Selection (mutually exclusive, priority: pixel > token > input)
    # ==========================================================================
    # Whether to enable input-level counterfactual sampling (sample_actions_with_cf) when available.
    use_cf_sampling: bool = False

    # Whether to enable feature-level counterfactual sampling (sample_actions_with_feature_cf) when available.
    # Feature-level CF intervenes at VLM output features rather than input observations.
    use_cf_feature_sampling: bool = False

    # Token-level CF: Block high-attention image tokens in cf_attn_mask.
    use_cf_token_sampling: bool = False

    # Pixel-level CF: Zero high-attention pixel patches in raw images and re-encode.
    use_cf_pixel_sampling: bool = False

    # ==========================================================================
    # Attention-guided CF common parameters (for token/pixel-level CF)
    # ==========================================================================
    # Transformer layer index to extract attention from (default 16, middle layer).
    cf_attn_layer_index: int = 16
    # Diffusion time point for attention extraction (default 1.0 = pure noise).
    cf_attn_time: float = 1.0
    # Ratio of top-k high-attention tokens/pixels to intervene (default 0.2).
    cf_attn_topk_ratio: float = 0.2

    # ==========================================================================
    # Input-level CF parameters (existing)
    # ==========================================================================
    # CF reweighting mode (base / A-F), used for both input-level and feature-level CF.
    cf_mode: CfMode = CfMode.E
    # CF guidance strength.
    cf_guidance_scale: float = 0.1
    # If VLM effect is above this threshold, fallback to baseline action.
    vlm_effect_upper_threshold: float = 0.5
    # Whether to clear prompt in image-zero CF branch (input-level CF only).
    clear_prompt_for_image_cf: bool = True
    # Whether to apply CF reweighting.
    reweight_action_with_cf: bool = True

    # Whether to return CF metrics (weights/effects) alongside actions.
    return_cf_metrics: bool = True

    # ==========================================================================
    # Attention visualization parameters
    # ==========================================================================
    # Whether to visualize attention maps (for debugging/testing).
    visualize_attention: bool = False
    # Directory to save attention visualization outputs.
    visualization_dir: str = "results/attention_vis"
    # Frequency of visualization (save every N steps).
    visualization_frequency: int = 10

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="/data4/zhy/models/openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    use_cf_sampling: bool = False,
    use_cf_feature_sampling: bool = False,
    use_cf_token_sampling: bool = False,
    use_cf_pixel_sampling: bool = False,
    cf_attn_layer_index: int = 16,
    cf_attn_time: float = 1.0,
    cf_attn_topk_ratio: float = 0.2,
    sample_kwargs: dict | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            use_cf_sampling=use_cf_sampling,
            use_cf_feature_sampling=use_cf_feature_sampling,
            use_cf_token_sampling=use_cf_token_sampling,
            use_cf_pixel_sampling=use_cf_pixel_sampling,
            cf_attn_layer_index=cf_attn_layer_index,
            cf_attn_time=cf_attn_time,
            cf_attn_topk_ratio=cf_attn_topk_ratio,
            sample_kwargs=sample_kwargs,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    sample_kwargs = None
    use_cf = args.use_cf_sampling or args.use_cf_feature_sampling or args.use_cf_token_sampling or args.use_cf_pixel_sampling
    if use_cf:
        sample_kwargs = {}
        # Add input-level CF specific kwargs
        if args.use_cf_sampling or args.use_cf_feature_sampling:
            sample_kwargs.update({
                "return_cf_metrics": args.return_cf_metrics,
                "reweight_action_with_cf": args.reweight_action_with_cf,
                "cf_guidance_scale": args.cf_guidance_scale,
                "vlm_effect_upper_threshold": args.vlm_effect_upper_threshold,
                "cf_mode": args.cf_mode.value,
            })
            if args.use_cf_sampling:
                sample_kwargs["clear_prompt_for_image_cf"] = args.clear_prompt_for_image_cf
        # Add attention-guided CF specific kwargs
        if args.use_cf_token_sampling or args.use_cf_pixel_sampling:
            sample_kwargs.update({
                "layer_index": args.cf_attn_layer_index,
                "attention_time": args.cf_attn_time,
                # Keep JIT-friendly outputs in online evaluation. Raw attention payload
                # contains Python objects (e.g. dict bounds, strings) and should only be
                # enabled in non-JIT debugging.
                "return_attention_data": False,
                # Add reweighting parameters (same as input-level CF)
                "cf_mode": args.cf_mode.value,
                "cf_guidance_scale": args.cf_guidance_scale,
                "reweight_action_with_cf": args.reweight_action_with_cf,
                "effect_threshold": args.vlm_effect_upper_threshold,
                # Add visualization parameters
                "visualize_attention": args.visualize_attention,
                "visualization_dir": args.visualization_dir,
                "visualization_frequency": args.visualization_frequency,
            })
            if args.use_cf_token_sampling:
                sample_kwargs["token_topk_ratio"] = args.cf_attn_topk_ratio
            if args.use_cf_pixel_sampling:
                sample_kwargs["pixel_topk_ratio"] = args.cf_attn_topk_ratio

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                use_cf_sampling=args.use_cf_sampling,
                use_cf_feature_sampling=args.use_cf_feature_sampling,
                use_cf_token_sampling=args.use_cf_token_sampling,
                use_cf_pixel_sampling=args.use_cf_pixel_sampling,
                cf_attn_layer_index=args.cf_attn_layer_index,
                cf_attn_time=args.cf_attn_time,
                cf_attn_topk_ratio=args.cf_attn_topk_ratio,
                sample_kwargs=sample_kwargs,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                use_cf_sampling=args.use_cf_sampling,
                use_cf_feature_sampling=args.use_cf_feature_sampling,
                use_cf_token_sampling=args.use_cf_token_sampling,
                use_cf_pixel_sampling=args.use_cf_pixel_sampling,
                cf_attn_layer_index=args.cf_attn_layer_index,
                cf_attn_time=args.cf_attn_time,
                cf_attn_topk_ratio=args.cf_attn_topk_ratio,
                sample_kwargs=sample_kwargs,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        ## modified
        ping_interval=(None if args.ws_ping_interval_s <= 0 else args.ws_ping_interval_s),
        ping_timeout=(None if args.ws_ping_timeout_s <= 0 else args.ws_ping_timeout_s),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
