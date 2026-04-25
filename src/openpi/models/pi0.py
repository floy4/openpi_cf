import logging
import enum
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class CfMode(str, enum.Enum):
    """Counterfactual reweighting mode.

    Modes align with the experimental schemes in Robocket executor:
    A-F plus BASE passthrough.
    """

    BASE = "base"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class CfFeatureMode(str, enum.Enum):
    """Counterfactual reweighting mode for feature-level CF.

    Feature-level CF intervenes at the VLM output features (prefix_tokens)
    rather than input observations.

    Modes:
    - BASE: No CF intervention, returns baseline actions
    - VLM_ZERO: Zero VLM features (image + language tokens) in prefix
    - STATE_ZERO: Zero state features in prefix/suffix
    """

    BASE = "base"
    VLM_ZERO = "vlm_zero"
    STATE_ZERO = "state_zero"


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cf_attn_mask: at.Bool[at.Array, "b s p+s"] | None = None,
    ) -> _model.Actions:
        observation, prefix_mask, kv_cache = self.prepare_prefix_for_sampling(observation)
        return self.sample_actions_from_precomputed_prefix(
            rng,
            observation,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise,
            cf_attn_mask=cf_attn_mask,
        )

    def prepare_prefix_for_sampling(
        self,
        observation: _model.Observation,
    ) -> tuple[_model.Observation, at.Bool[at.Array, "b p"], Any]:
        """Preprocess observation and build prefix KV cache once for decoding."""
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        return observation, prefix_mask, kv_cache

    def _compute_image_token_bounds(
        self,
        observation: _model.Observation,
    ) -> dict[str, tuple[int, int]]:
        """Compute image token ranges in prefix order for each camera."""
        image_bounds = {}
        current = 0
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            token_count = int(image_tokens.shape[1])
            image_bounds[name] = (current, current + token_count)
            current += token_count
        return image_bounds

    def _extract_single_step_attention_map_from_precomputed_prefix(
        self,
        observation: _model.Observation,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: Any,
        *,
        layer_index: int,
        time: float,
        noise: at.Float[at.Array, "b ah ad"],
    ) -> dict[str, Any]:
        """Extract last-query attention on one decode step at one transformer layer."""
        batch_size = observation.state.shape[0]

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            noise,
            jnp.full((batch_size,), time, dtype=noise.dtype),
        )
        suffix_len = int(suffix_tokens.shape[1])

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, _), _, last_query_attn_by_head = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            method="decode_with_last_query_attention",
            layer_index=layer_index,
        )
        assert prefix_out is None

        # Average over heads to get a single attention profile per sample.
        last_query_attn_head_avg = jnp.mean(last_query_attn_by_head, axis=1)

        image_bounds = self._compute_image_token_bounds(observation)
        image_attn_maps: dict[str, Any] = {}
        for name, (start, end) in image_bounds.items():
            cam_attn = last_query_attn_head_avg[:, start:end]
            cam_attn = cam_attn * prefix_mask[:, start:end].astype(cam_attn.dtype)
            token_count = end - start
            side = int(round(token_count ** 0.5))
            if side * side == token_count:
                cam_attn = jnp.reshape(cam_attn, (batch_size, side, side))
            image_attn_maps[name] = cam_attn

        return {
            "layer_index": layer_index,
            "suffix_len": suffix_len,
            "image_token_bounds": image_bounds,
            "last_query_attn_by_head": last_query_attn_by_head,
            "last_query_attn_head_avg": last_query_attn_head_avg,
            "image_attn_maps": image_attn_maps,
        }

    def extract_single_step_attention_map(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        layer_index: int = 16,
        time: float = 1.0,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> dict[str, Any]:
        """Public API: extract one-step, one-layer last-query attention map."""
        observation, prefix_mask, kv_cache = self.prepare_prefix_for_sampling(observation)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        return self._extract_single_step_attention_map_from_precomputed_prefix(
            observation,
            prefix_mask,
            kv_cache,
            layer_index=layer_index,
            time=time,
            noise=noise,
        )

    def build_token_level_cf_attn_mask_from_attention(
        self,
        last_query_attn_head_avg: at.Float[at.Array, "b s"],
        prefix_mask: at.Bool[at.Array, "b p"],
        image_token_bounds: dict[str, tuple[int, int]],
        suffix_len: int,
        *,
        topk_ratio: float = 0.2,
    ) -> at.Bool[at.Array, "b s p+s"]:
        """Build a token-level CF mask by blocking top-k attended image tokens."""
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

        batch_size = int(prefix_mask.shape[0])
        prefix_len = int(prefix_mask.shape[1])
        total_len = int(last_query_attn_head_avg.shape[1])
        if total_len != prefix_len + suffix_len:
            raise ValueError(
                "Attention length mismatch: "
                f"got total_len={total_len}, expected prefix_len + suffix_len={prefix_len + suffix_len}"
            )

        image_token_mask = jnp.zeros((prefix_len,), dtype=jnp.bool_)
        for start, end in image_token_bounds.values():
            start = max(0, min(prefix_len, int(start)))
            end = max(0, min(prefix_len, int(end)))
            if start < end:
                image_token_mask = image_token_mask.at[start:end].set(True)

        scores = last_query_attn_head_avg[:, :prefix_len]
        candidate_mask = jnp.logical_and(prefix_mask, image_token_mask[None, :])

        candidate_counts = jnp.sum(candidate_mask, axis=1)
        k = jnp.ceil(candidate_counts.astype(jnp.float32) * topk_ratio).astype(jnp.int32)
        k = jnp.where(candidate_counts > 0, jnp.maximum(k, 1), 0)

        masked_scores = jnp.where(candidate_mask, scores, -jnp.inf)
        sorted_scores = jnp.sort(masked_scores, axis=1)

        threshold_indices = jnp.where(candidate_counts > 0, prefix_len - k, prefix_len - 1)
        threshold_indices = jnp.clip(threshold_indices, 0, max(prefix_len - 1, 0)).astype(jnp.int32)
        thresholds = jnp.take_along_axis(sorted_scores, threshold_indices[:, None], axis=1).squeeze(1)

        block_prefix = jnp.logical_and(candidate_mask, scores >= thresholds[:, None])
        block_prefix = jnp.where(candidate_counts[:, None] > 0, block_prefix, False)

        # True means visible. Blocked prefix tokens are set to False.
        cf_mask = jnp.ones((batch_size, suffix_len, prefix_len + suffix_len), dtype=jnp.bool_)
        cf_mask = cf_mask.at[:, :, :prefix_len].set(jnp.logical_not(block_prefix)[:, None, :])

        return cf_mask

    def sample_actions_with_single_step_token_cf(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        layer_index: int = 16,
        attention_time: float = 1.0,
        token_topk_ratio: float = 0.2,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_attention_data: bool = False,
        return_cf_metrics: bool = False,
        cf_mode: str = "E",
        cf_guidance_scale: float = 0.1,
        reweight_action_with_cf: bool = True,
        effect_threshold: float = 0.5,
        visualize_attention: bool = False,
        visualization_dir: str = "results/attention_vis",
        visualization_frequency: int = 10,
        step_idx: int = 0,
        episode_idx: int = 0,
    ):
        """Sample actions using token-level CF masking with optional reweighting.

        Flow:
        1. Compute baseline actions (no CF intervention)
        2. Extract attention map and build cf_attn_mask
        3. Compute CF actions (with token-level masking)
        4. Calculate effect size (L2 norm of delta)
        5. Apply reweighting based on cf_mode (similar to input-level CF)

        Args:
            rng: Random key for sampling
            observation: Input observation
            layer_index: Transformer layer to extract attention from
            attention_time: Diffusion time for attention extraction
            token_topk_ratio: Ratio of top-k tokens to mask
            num_steps: Number of diffusion steps
            noise: Optional initial noise
            return_attention_data: Whether to return attention data
            cf_mode: CF reweighting mode (BASE/A/B/C/D/E/F)
            cf_guidance_scale: Scale for CF delta in reweighting
            reweight_action_with_cf: Whether to apply CF reweighting
            effect_threshold: Threshold for fallback to baseline
            visualize_attention: Whether to save attention visualization
            visualization_dir: Directory for visualization outputs
            visualization_frequency: Save visualization every N steps
            step_idx: Current step index (for visualization filename)
            episode_idx: Current episode index (for visualization filename)

        Returns:
            actions: Final actions (reweighted or CF-only)
            attention_data: Optional dict with attention maps and CF metrics
        """
        observation, prefix_mask, kv_cache = self.prepare_prefix_for_sampling(observation)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 1. Compute baseline actions (no CF)
        actions_baseline = self.sample_actions_from_precomputed_prefix(
            rng,
            observation,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise,
            cf_attn_mask=None,
        )

        # 2. Extract attention map and build cf_attn_mask
        attn_data = self._extract_single_step_attention_map_from_precomputed_prefix(
            observation,
            prefix_mask,
            kv_cache,
            layer_index=layer_index,
            time=attention_time,
            noise=noise,
        )

        # Visualization is handled outside JIT-compiled sampling paths.

        cf_attn_mask = self.build_token_level_cf_attn_mask_from_attention(
            attn_data["last_query_attn_head_avg"],
            prefix_mask,
            attn_data["image_token_bounds"],
            attn_data["suffix_len"],
            topk_ratio=token_topk_ratio,
        )

        # 3. Compute CF actions (with token-level masking)
        actions_cf = self.sample_actions_from_precomputed_prefix(
            rng,
            observation,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise,
            cf_attn_mask=cf_attn_mask,
        )

        # 4. Calculate effect size
        effect_cf = self._compute_cf_action_diff(actions_baseline, actions_cf)

        # 5. Apply reweighting based on cf_mode
        mode_normalized = str(cf_mode).upper()
        if mode_normalized == "BASE":
            mode = CfMode.BASE
        else:
            try:
                mode = CfMode(mode_normalized)
            except ValueError:
                logger.warning("Unknown cf_mode=%s, fallback to E", cf_mode)
                mode = CfMode.E

        delta_cf = actions_baseline - actions_cf
        if reweight_action_with_cf:
            use_baseline = effect_cf > effect_threshold

            if mode == CfMode.BASE:
                actions_final = actions_baseline
            elif mode == CfMode.A:
                actions_final = actions_baseline + cf_guidance_scale * delta_cf
            elif mode == CfMode.B:
                actions_final = actions_baseline + cf_guidance_scale * delta_cf
            elif mode == CfMode.C:
                actions_final = actions_baseline + cf_guidance_scale * delta_cf
            elif mode == CfMode.D:
                delta_cf_clipped = jnp.clip(delta_cf, -0.1, 0.1)
                actions_final = actions_baseline + cf_guidance_scale * delta_cf_clipped
            elif mode == CfMode.F:
                delta_cf_soft = jnp.tanh(delta_cf / 0.1) * 0.1
                actions_final = actions_baseline + cf_guidance_scale * delta_cf_soft
            else:  # Mode E: adaptive
                actions_final = actions_baseline + cf_guidance_scale * delta_cf

            actions_final = jnp.where(use_baseline, actions_baseline, actions_final)
        else:
            actions_final = actions_cf

        if return_attention_data:
            return actions_final, {
                "actions_baseline": actions_baseline,
                "actions_cf": actions_cf,
                "effect_cf": effect_cf,
                "delta_cf_norm": jnp.mean(jnp.linalg.norm(jnp.reshape(delta_cf, (batch_size, -1)), axis=-1)),
                "use_baseline": effect_cf > effect_threshold,
                "cf_mode": cf_mode,
                "cf_guidance_scale": cf_guidance_scale,
                "effect_threshold": effect_threshold,
                "cf_attn_mask": cf_attn_mask,
                **attn_data,
            }

        if return_cf_metrics:
            return actions_final, {
                "effect_cf": effect_cf,
                "delta_cf_norm": jnp.mean(jnp.linalg.norm(jnp.reshape(delta_cf, (batch_size, -1)), axis=-1)),
                "use_baseline": effect_cf > effect_threshold,
                "cf_guidance_scale": jnp.asarray(cf_guidance_scale, dtype=jnp.float32),
                "effect_threshold": jnp.asarray(effect_threshold, dtype=jnp.float32),
                "token_topk_ratio": jnp.asarray(token_topk_ratio, dtype=jnp.float32),
                "layer_index": jnp.asarray(layer_index, dtype=jnp.int32),
                "attention_time": jnp.asarray(attention_time, dtype=jnp.float32),
            }

        return actions_final

    def _zero_high_attention_pixel_patches(
        self,
        images: dict[str, at.Float[at.Array, "b h w c"]],
        image_attn_maps: dict[str, at.Float[at.Array, "b side side"]],
        topk_ratio: float,
    ) -> dict[str, at.Float[at.Array, "b h w c"]]:
        """Zero pixel patches corresponding to top-k attention tokens.

        SigLIP So400m/14 produces 256 tokens (16x16 grid) for 224x224 images.
        Each token corresponds to a 14x14 pixel patch.

        Args:
            images: Original images dict, shape [b, h, w, c]
            image_attn_maps: Attention maps per camera, shape [b, side, side] where side=16
            topk_ratio: Ratio of top-k high attention tokens to zero (0.0-1.0)

        Returns:
            Modified images with high-attention pixel patches zeroed.
        """
        modified_images = {}
        for name, img in images.items():
            attn = image_attn_maps.get(name)
            if attn is None:
                modified_images[name] = img
                continue

            batch_size = img.shape[0]
            h, w = img.shape[1], img.shape[2]
            c = img.shape[3]

            # Calculate patch size: 224 / 16 = 14
            patch_size = h // attn.shape[1]
            num_patches_side = attn.shape[1]  # 16 for SigLIP

            # Build modified image using JAX operations
            modified = jnp.copy(img)
            zero_patch = jnp.zeros((1, patch_size, patch_size, c), dtype=img.dtype)

            for b in range(batch_size):
                # Flatten attention and select top-k indices
                flat_attn = attn[b].flatten()
                total_patches = flat_attn.shape[0]
                k = max(1, int(total_patches * topk_ratio))

                # Get top-k indices (highest attention)
                top_indices = jnp.argsort(flat_attn)[-k:]

                # Zero each top-k patch
                for idx in top_indices:
                    row = idx // num_patches_side
                    col = idx % num_patches_side
                    y_start = row * patch_size
                    x_start = col * patch_size
                    modified = jax.lax.dynamic_update_slice(
                        modified,
                        zero_patch,
                        (b, y_start, x_start, 0),
                    )

            modified_images[name] = modified

        return modified_images

    def sample_actions_with_pixel_level_cf(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        layer_index: int = 16,
        attention_time: float = 1.0,
        pixel_topk_ratio: float = 0.2,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_attention_data: bool = False,
        return_cf_metrics: bool = False,
        cf_mode: str = "E",
        cf_guidance_scale: float = 0.1,
        reweight_action_with_cf: bool = True,
        effect_threshold: float = 0.5,
        visualize_attention: bool = False,
        visualization_dir: str = "results/attention_vis",
        visualization_frequency: int = 10,
        step_idx: int = 0,
        episode_idx: int = 0,
    ):
        """Sample actions by zeroing high-attention pixel patches with optional reweighting.

        Flow:
        1. Compute baseline actions (no CF intervention)
        2. Extract attention map from original observation
        3. Zero high-attention pixel patches and create modified observation
        4. Compute CF actions (with pixel-level intervention)
        5. Calculate effect size (L2 norm of delta)
        6. Apply reweighting based on cf_mode (similar to input-level CF)

        Args:
            rng: Random key for sampling
            observation: Input observation
            layer_index: Transformer layer to extract attention from (default 16)
            attention_time: Diffusion time for attention extraction (default 1.0)
            pixel_topk_ratio: Ratio of pixel patches to zero (default 0.2)
            num_steps: Number of diffusion steps
            noise: Optional initial noise
            return_attention_data: Whether to return attention data alongside actions
            cf_mode: CF reweighting mode (BASE/A/B/C/D/E/F)
            cf_guidance_scale: Scale for CF delta in reweighting
            reweight_action_with_cf: Whether to apply CF reweighting
            effect_threshold: Threshold for fallback to baseline
            visualize_attention: Whether to save attention visualization
            visualization_dir: Directory for visualization outputs
            visualization_frequency: Save visualization every N steps
            step_idx: Current step index (for visualization filename)
            episode_idx: Current episode index (for visualization filename)

        Returns:
            actions: Final actions (reweighted or CF-only)
            attention_data: Optional dict with attention maps and CF metrics
        """
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 1. Compute baseline actions (no CF)
        actions_baseline = self.sample_actions(
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )

        # 2. Extract attention map from original observation
        attn_data = self.extract_single_step_attention_map(
            rng,
            observation,
            layer_index=layer_index,
            time=attention_time,
            noise=noise,
        )

        # Visualization is handled outside JIT-compiled sampling paths.

        # 3. Zero high-attention pixel patches
        modified_images = self._zero_high_attention_pixel_patches(
            observation.images,
            attn_data["image_attn_maps"],
            pixel_topk_ratio,
        )

        # 4. Create modified observation with zeroed pixel patches
        modified_obs = _model.Observation(
            images=modified_images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

        # 5. Compute CF actions (with pixel-level intervention)
        actions_cf = self.sample_actions(
            rng,
            modified_obs,
            num_steps=num_steps,
            noise=noise,
        )

        # 6. Calculate effect size
        effect_cf = self._compute_cf_action_diff(actions_baseline, actions_cf)

        # 7. Apply reweighting based on cf_mode
        mode_normalized = str(cf_mode).upper()
        if mode_normalized == "BASE":
            mode = CfMode.BASE
        else:
            try:
                mode = CfMode(mode_normalized)
            except ValueError:
                logger.warning("Unknown cf_mode=%s, fallback to E", cf_mode)
                mode = CfMode.E

        delta_cf = actions_baseline - actions_cf
        if reweight_action_with_cf:
            use_baseline = effect_cf > effect_threshold

            if mode == CfMode.BASE:
                actions_final = actions_baseline
            elif mode == CfMode.A:
                actions_final = actions_baseline + cf_guidance_scale * delta_cf
            elif mode == CfMode.B:
                actions_final = actions_baseline + cf_guidance_scale * delta_cf
            elif mode == CfMode.C:
                actions_final = actions_baseline + cf_guidance_scale * delta_cf
            elif mode == CfMode.D:
                delta_cf_clipped = jnp.clip(delta_cf, -0.1, 0.1)
                actions_final = actions_baseline + cf_guidance_scale * delta_cf_clipped
            elif mode == CfMode.F:
                delta_cf_soft = jnp.tanh(delta_cf / 0.1) * 0.1
                actions_final = actions_baseline + cf_guidance_scale * delta_cf_soft
            else:  # Mode E: adaptive
                actions_final = actions_baseline + cf_guidance_scale * delta_cf

            actions_final = jnp.where(use_baseline, actions_baseline, actions_final)
        else:
            actions_final = actions_cf

        if return_attention_data:
            return actions_final, {
                "actions_baseline": actions_baseline,
                "actions_cf": actions_cf,
                "effect_cf": effect_cf,
                "delta_cf_norm": jnp.mean(jnp.linalg.norm(jnp.reshape(delta_cf, (batch_size, -1)), axis=-1)),
                "use_baseline": effect_cf > effect_threshold,
                "cf_mode": cf_mode,
                "cf_guidance_scale": cf_guidance_scale,
                "effect_threshold": effect_threshold,
                "pixel_topk_ratio": pixel_topk_ratio,
                "modified_images": modified_images,
                **attn_data,
            }

        if return_cf_metrics:
            return actions_final, {
                "effect_cf": effect_cf,
                "delta_cf_norm": jnp.mean(jnp.linalg.norm(jnp.reshape(delta_cf, (batch_size, -1)), axis=-1)),
                "use_baseline": effect_cf > effect_threshold,
                "cf_guidance_scale": jnp.asarray(cf_guidance_scale, dtype=jnp.float32),
                "effect_threshold": jnp.asarray(effect_threshold, dtype=jnp.float32),
                "pixel_topk_ratio": jnp.asarray(pixel_topk_ratio, dtype=jnp.float32),
                "layer_index": jnp.asarray(layer_index, dtype=jnp.int32),
                "attention_time": jnp.asarray(attention_time, dtype=jnp.float32),
            }

        return actions_final

    def  sample_actions_from_precomputed_prefix(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache: Any,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cf_attn_mask: at.Bool[at.Array, "b s p+s"] | None = None,
    ) -> _model.Actions: 
        """Decode actions while reusing precomputed prefix context."""
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        prefix_len = prefix_mask.shape[1]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            # apply CF attention mask if provided (blocks attention to specific modalities)
            if cf_attn_mask is not None:
                full_attn_mask = jnp.logical_and(full_attn_mask, cf_attn_mask)

            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_len + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_triplet_shared_prefix(
        self,
        rng_base: at.KeyArrayLike,
        rng_no_image: at.KeyArrayLike,
        rng_no_state: at.KeyArrayLike,
        observation: _model.Observation,
        cf_attn_mask_no_image: at.Bool[at.Array, "b s p+s"],
        cf_attn_mask_no_state: at.Bool[at.Array, "b s p+s"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, _model.Actions, _model.Actions]:
        """Decode BASE/NO_IMAGE/NO_STATE with one shared prefix prefill."""
        observation, prefix_mask, kv_cache = self.prepare_prefix_for_sampling(observation)

        batch_size = observation.state.shape[0]
        noise_shape = (batch_size, self.action_horizon, self.action_dim)
        noise_base = jax.random.normal(rng_base, noise_shape)
        noise_no_image = jax.random.normal(rng_no_image, noise_shape)
        noise_no_state = jax.random.normal(rng_no_state, noise_shape)

        actions_base = self.sample_actions_from_precomputed_prefix(
            rng_base,
            observation,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise_base,
            cf_attn_mask=None,
        )
        actions_no_image = self.sample_actions_from_precomputed_prefix(
            rng_no_image,
            observation,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise_no_image,
            cf_attn_mask=cf_attn_mask_no_image,
        )
        actions_no_state = self.sample_actions_from_precomputed_prefix(
            rng_no_state,
            observation,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise_no_state,
            cf_attn_mask=cf_attn_mask_no_state,
        )

        return actions_base, actions_no_image, actions_no_state

    ## modified
    def _make_cf_observation_state_zero(self, observation: _model.Observation) -> _model.Observation:
        """Create a counterfactual observation by zeroing proprio state only."""
        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=jnp.zeros_like(observation.state),
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

    def _make_cf_observation_image_zero(
        self,
        observation: _model.Observation,
        *,
        clear_prompt: bool = True,
    ) -> _model.Observation:
        """Create a counterfactual observation by zeroing image inputs.

        Optionally clears tokenized prompt to avoid language leakage when estimating
        image-only effect.
        """
        zero_images = {name: jnp.zeros_like(image) for name, image in observation.images.items()}
        if clear_prompt:
            tokenized_prompt = None
            tokenized_prompt_mask = None
            token_ar_mask = None
            token_loss_mask = None
        else:
            tokenized_prompt = observation.tokenized_prompt
            tokenized_prompt_mask = observation.tokenized_prompt_mask
            token_ar_mask = observation.token_ar_mask
            token_loss_mask = observation.token_loss_mask

        return _model.Observation(
            images=zero_images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

    def _compute_cf_action_diff(self, actions_base: _model.Actions, actions_cf: _model.Actions) -> at.Float[at.Array, ""]:
        """Compute scalar action difference for modality effect estimation."""
        diff = actions_base - actions_cf
        batch = diff.shape[0]
        diff_flat = jnp.reshape(diff, (batch, -1))
        return jnp.mean(jnp.linalg.norm(diff_flat, axis=-1))

    def _visualize_attention(
        self,
        attn_data: dict,
        observation: _model.Observation,
        visualization_dir: str,
        step_idx: int,
        episode_idx: int,
        layer_index: int,
        cf_mode: str,
    ) -> None:
        """Save attention visualization to disk (non-JAX side effect)."""
        import os
        import numpy as np
        from pathlib import Path

        try:
            from openpi.utils.attention_visualization import save_attention_visualization
        except ImportError:
            logger.warning("attention_visualization module not available, skipping visualization")
            return

        output_path = Path(visualization_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert JAX arrays to numpy for visualization
        images_np = {}
        for cam_name, img in observation.images.items():
            if hasattr(img, "to_numpy"):
                images_np[cam_name] = img.to_numpy()
            else:
                images_np[cam_name] = np.asarray(img)

        # Save visualization
        saved_files = save_attention_visualization(
            attn_data,
            images_np,
            visualization_dir,
            step_idx,
            episode_idx,
            layer_index,
            cf_mode,
        )

        if saved_files:
            logger.info(f"Saved {len(saved_files)} attention visualization(s) to {visualization_dir}")
        else:
            logger.warning("Attention visualization produced no files in %s", visualization_dir)

    def sample_actions_with_cf(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        reweight_action_with_cf: bool = True,
        cf_guidance_scale: float = 0.1,
        vlm_effect_upper_threshold: float = 0.5,
        cf_mode: str = "E",
        clear_prompt_for_image_cf: bool = True,
        return_cf_metrics: bool = False,
    ):
        """Sample actions with counterfactual (CF) analysis and optional reweighting.

        CF branches:
        - baseline: original observation
        - proprio-zero: state is zeroed
        - image-zero: images are zeroed (and optional prompt clearing)
        """
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        actions_base = self.sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        obs_prop0 = self._make_cf_observation_state_zero(observation)
        actions_prop0 = self.sample_actions(rng, obs_prop0, num_steps=num_steps, noise=noise)

        obs_img0 = self._make_cf_observation_image_zero(
            observation,
            clear_prompt=clear_prompt_for_image_cf,
        )
        actions_img0 = self.sample_actions(rng, obs_img0, num_steps=num_steps, noise=noise)

        effect_prop = self._compute_cf_action_diff(actions_base, actions_prop0)
        effect_vlm = self._compute_cf_action_diff(actions_base, actions_img0)

        mode_normalized = str(cf_mode).upper()
        if mode_normalized == "BASE":
            mode = CfMode.BASE
        else:
            try:
                mode = CfMode(mode_normalized)
            except ValueError:
                logger.warning("Unknown cf_mode=%s, fallback to E", cf_mode)
                mode = CfMode.E

        if reweight_action_with_cf:
            delta_img = actions_base - actions_img0
            delta_prop = actions_base - actions_prop0
            use_base = effect_vlm > vlm_effect_upper_threshold

            eps = 1e-6
            total_effect = effect_vlm + effect_prop + eps
            w_vlm = effect_vlm / total_effect
            w_prop = effect_prop / total_effect

            if mode == CfMode.BASE:
                actions_cf = actions_base
            elif mode == CfMode.A:
                actions_cf = actions_base + cf_guidance_scale * delta_img
            elif mode == CfMode.B:
                prop_scale = 0.05
                actions_cf = actions_base + cf_guidance_scale * delta_img + prop_scale * delta_prop
            elif mode == CfMode.C:
                abs_max = jnp.max(jnp.abs(delta_prop), axis=(-2, -1), keepdims=True)
                delta_prop_norm = jnp.where(abs_max > 0, delta_prop / (abs_max + eps), delta_prop)
                actions_cf = actions_base + cf_guidance_scale * (
                    w_vlm * delta_img + w_prop * 0.1 * delta_prop_norm
                )
            elif mode == CfMode.D:
                delta_prop_clipped = jnp.clip(delta_prop, -0.1, 0.1)
                actions_cf = actions_base + cf_guidance_scale * delta_img + 0.05 * delta_prop_clipped
            elif mode == CfMode.F:
                delta_prop_soft = jnp.tanh(delta_prop / 0.1) * 0.1
                actions_cf = actions_base + cf_guidance_scale * delta_img + 0.05 * delta_prop_soft
            else:
                # Mode E: adaptive prop weight + clipped proprio delta.
                prop_ratio = effect_prop / (effect_vlm + eps)
                prop_weight = 0.05 * jnp.minimum(1.0, prop_ratio)
                delta_prop_clipped = jnp.clip(delta_prop, -0.1, 0.1)
                actions_cf = actions_base + cf_guidance_scale * delta_img + prop_weight * delta_prop_clipped

            actions_final = jnp.where(use_base, actions_base, actions_cf)
        else:
            actions_final = actions_base

        if return_cf_metrics:
            use_base = effect_vlm > vlm_effect_upper_threshold
            # 计算详细权重信息
            eps = 1e-6
            total_effect = effect_vlm + effect_prop + eps
            w_vlm = effect_vlm / total_effect
            w_prop = effect_prop / total_effect
            scalar_dtype = effect_vlm.dtype

            # 计算各模式下的实际权重
            if mode == CfMode.BASE:
                actual_vlm_weight = 0.0
                actual_prop_weight = 0.0
            elif mode == CfMode.A:
                actual_vlm_weight = cf_guidance_scale
                actual_prop_weight = 0.0
            elif mode == CfMode.B:
                actual_vlm_weight = cf_guidance_scale
                actual_prop_weight = 0.05
            elif mode == CfMode.C:
                actual_vlm_weight = cf_guidance_scale * w_vlm
                actual_prop_weight = cf_guidance_scale * w_prop * 0.1
            elif mode == CfMode.D:
                actual_vlm_weight = cf_guidance_scale
                actual_prop_weight = 0.05
            elif mode == CfMode.F:
                actual_vlm_weight = cf_guidance_scale
                actual_prop_weight = 0.05
            else:  # Mode E
                prop_ratio = effect_prop / (effect_vlm + eps)
                state_weight = 0.05 * jnp.minimum(1.0, prop_ratio)
                actual_vlm_weight = cf_guidance_scale
                actual_prop_weight = state_weight

            actual_vlm_weight = jnp.asarray(actual_vlm_weight, dtype=scalar_dtype)
            actual_prop_weight = jnp.asarray(actual_prop_weight, dtype=scalar_dtype)
            cf_guidance_scale_metric = jnp.asarray(cf_guidance_scale, dtype=scalar_dtype)
            vlm_effect_threshold_metric = jnp.asarray(vlm_effect_upper_threshold, dtype=scalar_dtype)

            # delta 的 L2 范数
            delta_vlm_norm = jnp.mean(jnp.linalg.norm(jnp.reshape(delta_img, (batch_size, -1)), axis=-1))
            delta_prop_norm = jnp.mean(jnp.linalg.norm(jnp.reshape(delta_prop, (batch_size, -1)), axis=-1))

            return actions_final, {
                "effect_vlm": effect_vlm,
                "effect_prop": effect_prop,
                "use_base": use_base,
                # 效应比例权重
                "effect_ratio_vlm": w_vlm,
                "effect_ratio_prop": w_prop,
                # 实际应用权重
                "actual_vlm_weight": actual_vlm_weight,
                "actual_prop_weight": actual_prop_weight,
                # delta 范数
                "delta_vlm_norm": delta_vlm_norm,
                "delta_prop_norm": delta_prop_norm,
                # 配置参数
                "cf_guidance_scale": cf_guidance_scale_metric,
                "vlm_effect_threshold": vlm_effect_threshold_metric,
            }
        return actions_final

    # =========================================================================
    # Feature-level Counterfactual Methods
    # =========================================================================
    # Feature-level CF intervenes at the VLM output features (prefix_tokens)
    # rather than input observations. This provides a more direct intervention
    # on the learned representations.

    def _compute_prefix_token_bounds(
        self,
        observation: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b s emb"],
    ) -> dict[str, tuple[int, int]]:
        """Compute the boundaries of each modality in prefix_tokens.

        Returns a dict with:
        - 'vlm_start': start index of VLM tokens (image + language)
        - 'vlm_end': end index of VLM tokens
        - 'total_len': total length of prefix tokens

        For pi05, state is part of prefix_tokens after language tokens.
        For pi0, state is NOT in prefix_tokens (it's in suffix_tokens).
        """
        total_len = prefix_tokens.shape[1]

        # VLM tokens are all prefix tokens for pi0
        # For pi05, we need to know how many image + language tokens there are
        # We can estimate this from the tokenized_prompt length
        vlm_start = 0
        vlm_end = total_len

        # For pi05, estimate state token count based on state dimension
        # State is discretized into ~state_dim tokens
        if self.pi05 and observation.tokenized_prompt is not None:
            # Language tokens length
            lang_len = observation.tokenized_prompt.shape[1]
            # Image tokens: each image produces 256 tokens (SigLIP)
            num_images = len(observation.images)
            image_tokens_per_cam = 256  # SigLIP So400m/14 on 224x224
            image_len = num_images * image_tokens_per_cam
            # VLM portion = images + language (before state tokens)
            vlm_end = image_len + lang_len
            # Note: This is approximate; actual boundaries depend on tokenizer

        return {
            "vlm_start": vlm_start,
            "vlm_end": vlm_end,
            "total_len": total_len,
        }

    def _make_cf_prefix_tokens_vlm_zero(
        self,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        bounds: dict[str, tuple[int, int]],
    ) -> at.Float[at.Array, "b s emb"]:
        """Create counterfactual prefix tokens by zeroing VLM features.

        This zeros the image + language portion of prefix_tokens,
        keeping state tokens intact (if present in pi05).
        """
        vlm_start = bounds["vlm_start"]
        vlm_end = bounds["vlm_end"]

        # Create a mask that zeros VLM portion
        mask = jnp.ones(prefix_tokens.shape[1])
        mask = jnp.where(
            (jnp.arange(prefix_tokens.shape[1]) >= vlm_start) &
            (jnp.arange(prefix_tokens.shape[1]) < vlm_end),
            0.0,
            mask,
        )
        mask = mask.astype(prefix_tokens.dtype)

        return prefix_tokens * mask[:, None]

    def _make_cf_suffix_tokens_state_zero(
        self,
        suffix_tokens: at.Float[at.Array, "b s emb"],
        is_pi0: bool,
    ) -> at.Float[at.Array, "b s emb"]:
        """Create counterfactual suffix tokens by zeroing state features.

        For pi0: state is the first token in suffix_tokens, zero it.
        For pi05: state is not in suffix_tokens (it's in prefix_tokens),
                  so this method returns suffix_tokens unchanged.
        """
        if not is_pi0:
            # For pi05, state is in prefix_tokens, not suffix
            return suffix_tokens

        # For pi0, the first token in suffix is state_token
        # Zero the first token
        mask = jnp.ones(suffix_tokens.shape[1])
        mask = mask.at[0].set(0.0)
        mask = mask.astype(suffix_tokens.dtype)

        return suffix_tokens * mask[:, None]

    def _sample_actions_with_modified_prefix(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        zero_suffix_state: bool = False,
    ) -> _model.Actions:
        """Sample actions with modified prefix tokens.

        This is used for feature-level CF where we modify the VLM output
        features directly rather than the input observations.

        Args:
            zero_suffix_state: For pi0, whether to also zero the state token in suffix.
        """
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build KV cache with modified prefix tokens
        prefix_ar_mask = jnp.zeros(prefix_tokens.shape[1], dtype=jnp.bool_)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Now decode actions using the modified prefix
        dt = -1.0 / num_steps
        prefix_len = prefix_mask.shape[1]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # Zero state in suffix if requested (for pi0)
            if zero_suffix_state and not self.pi05:
                suffix_tokens = self._make_cf_suffix_tokens_state_zero(suffix_tokens, is_pi0=True)

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_with_feature_cf(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        reweight_action_with_cf: bool = True,
        cf_guidance_scale: float = 0.1,
        vlm_effect_upper_threshold: float = 0.5,
        cf_mode: str = "E",
        return_cf_metrics: bool = False,
    ):
        """Sample actions with feature-level counterfactual analysis.

        Feature-level CF intervenes at the VLM output features (prefix_tokens)
        rather than input observations:
        - VLM_ZERO: Zero the VLM features (image + language tokens) in prefix_tokens
        - STATE_ZERO: Zero the state features (in prefix for pi05, in suffix for pi0)

        This provides a more direct intervention on learned representations
        compared to input-level CF.

        Args:
            rng: Random key for sampling
            observation: Input observation
            num_steps: Flow matching steps
            noise: Optional initial noise for sampling
            reweight_action_with_cf: Whether to apply CF reweighting
            cf_guidance_scale: Scale for VLM delta in reweighting
            vlm_effect_upper_threshold: Threshold for VLM effect fallback
            cf_mode: CF reweighting mode (BASE/A/B/C/D/E/F)
            return_cf_metrics: Whether to return CF metrics

        Returns:
            actions_final: Reweighted actions (or baseline if not reweighting)
            metrics: Optional dict with effect sizes and mode info
        """
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Preprocess observation
        observation = _model.preprocess_observation(None, observation, train=False)

        # Step 1: Get original prefix_tokens (VLM output features)
        prefix_tokens_base, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        bounds = self._compute_prefix_token_bounds(observation, prefix_tokens_base)

        # Step 2: Sample baseline actions using original prefix
        actions_base = self._sample_actions_with_modified_prefix(
            rng, observation, prefix_tokens_base, prefix_mask,
            num_steps=num_steps, noise=noise, zero_suffix_state=False,
        )

        # Step 3: Create counterfactual prefix_tokens with VLM features zeroed
        prefix_tokens_vlm_zero = self._make_cf_prefix_tokens_vlm_zero(prefix_tokens_base, bounds)
        actions_vlm_zero = self._sample_actions_with_modified_prefix(
            rng, observation, prefix_tokens_vlm_zero, prefix_mask,
            num_steps=num_steps, noise=noise, zero_suffix_state=False,
        )

        # Step 4: Create counterfactual with state features zeroed
        # For pi05: state is in prefix_tokens (after language tokens)
        # For pi0: state is the first token in suffix_tokens
        if self.pi05:
            # For pi05, we need to modify prefix_tokens to zero state portion
            # Create a mask that zeros everything EXCEPT VLM portion
            vlm_start = bounds["vlm_start"]
            vlm_end = bounds["vlm_end"]
            mask = jnp.ones(prefix_tokens_base.shape[1])
            mask = jnp.where(
                (jnp.arange(prefix_tokens_base.shape[1]) >= vlm_start) &
                (jnp.arange(prefix_tokens_base.shape[1]) < vlm_end),
                1.0,  # Keep VLM portion
                0.0,  # Zero state portion
            )
            mask = mask.astype(prefix_tokens_base.dtype)
            prefix_tokens_state_zero = prefix_tokens_base * mask[:, None]
            actions_state_zero = self._sample_actions_with_modified_prefix(
                rng, observation, prefix_tokens_state_zero, prefix_mask,
                num_steps=num_steps, noise=noise, zero_suffix_state=False,
            )
        else:
            # For pi0, state is in suffix_tokens, so we keep prefix unchanged
            # and zero state in suffix during decoding
            actions_state_zero = self._sample_actions_with_modified_prefix(
                rng, observation, prefix_tokens_base, prefix_mask,
                num_steps=num_steps, noise=noise, zero_suffix_state=True,
            )

        # Step 5: Compute effects
        effect_vlm = self._compute_cf_action_diff(actions_base, actions_vlm_zero)
        effect_state = self._compute_cf_action_diff(actions_base, actions_state_zero)

        # Step 6: Apply reweighting (same logic as input-level CF)
        mode_normalized = str(cf_mode).upper()
        if mode_normalized == "BASE":
            mode = CfMode.BASE
        else:
            try:
                mode = CfMode(mode_normalized)
            except ValueError:
                logger.warning("Unknown cf_mode=%s, fallback to E", cf_mode)
                mode = CfMode.E

        if reweight_action_with_cf:
            delta_vlm = actions_base - actions_vlm_zero
            delta_state = actions_base - actions_state_zero
            use_base = effect_vlm > vlm_effect_upper_threshold

            eps = 1e-6
            total_effect = effect_vlm + effect_state + eps
            w_vlm = effect_vlm / total_effect
            w_state = effect_state / total_effect

            if mode == CfMode.BASE:
                actions_cf = actions_base
            elif mode == CfMode.A:
                actions_cf = actions_base + cf_guidance_scale * delta_vlm
            elif mode == CfMode.B:
                state_scale = 0.05
                actions_cf = actions_base + cf_guidance_scale * delta_vlm + state_scale * delta_state
            elif mode == CfMode.C:
                abs_max = jnp.max(jnp.abs(delta_state), axis=(-2, -1), keepdims=True)
                delta_state_norm = jnp.where(abs_max > 0, delta_state / (abs_max + eps), delta_state)
                actions_cf = actions_base + cf_guidance_scale * (
                    w_vlm * delta_vlm + w_state * 0.1 * delta_state_norm
                )
            elif mode == CfMode.D:
                delta_state_clipped = jnp.clip(delta_state, -0.1, 0.1)
                actions_cf = actions_base + cf_guidance_scale * delta_vlm + 0.05 * delta_state_clipped
            elif mode == CfMode.F:
                delta_state_soft = jnp.tanh(delta_state / 0.1) * 0.1
                actions_cf = actions_base + cf_guidance_scale * delta_vlm + 0.05 * delta_state_soft
            else:
                # Mode E: adaptive state weight + clipped state delta
                state_ratio = effect_state / (effect_vlm + eps)
                state_weight = 0.05 * jnp.minimum(1.0, state_ratio)
                delta_state_clipped = jnp.clip(delta_state, -0.1, 0.1)
                actions_cf = actions_base + cf_guidance_scale * delta_vlm + state_weight * delta_state_clipped

            actions_final = jnp.where(use_base, actions_base, actions_cf)
        else:
            actions_final = actions_base

        if return_cf_metrics:
            use_base = effect_vlm > vlm_effect_upper_threshold

            # 计算详细权重信息
            eps = 1e-6
            total_effect = effect_vlm + effect_state + eps
            w_vlm = effect_vlm / total_effect
            w_state = effect_state / total_effect
            scalar_dtype = effect_vlm.dtype

            # 计算各模式下的实际权重
            if mode == CfMode.BASE:
                actual_vlm_weight = 0.0
                actual_state_weight = 0.0
            elif mode == CfMode.A:
                actual_vlm_weight = cf_guidance_scale
                actual_state_weight = 0.0
            elif mode == CfMode.B:
                actual_vlm_weight = cf_guidance_scale
                actual_state_weight = 0.05
            elif mode == CfMode.C:
                actual_vlm_weight = cf_guidance_scale * w_vlm
                actual_state_weight = cf_guidance_scale * w_state * 0.1
            elif mode == CfMode.D:
                actual_vlm_weight = cf_guidance_scale
                actual_state_weight = 0.05
            elif mode == CfMode.F:
                actual_vlm_weight = cf_guidance_scale
                actual_state_weight = 0.05
            else:  # Mode E
                state_ratio = effect_state / (effect_vlm + eps)
                state_weight = 0.05 * jnp.minimum(1.0, state_ratio)
                actual_vlm_weight = cf_guidance_scale
                actual_state_weight = state_weight

            actual_vlm_weight = jnp.asarray(actual_vlm_weight, dtype=scalar_dtype)
            actual_state_weight = jnp.asarray(actual_state_weight, dtype=scalar_dtype)
            cf_guidance_scale_metric = jnp.asarray(cf_guidance_scale, dtype=scalar_dtype)
            vlm_effect_threshold_metric = jnp.asarray(vlm_effect_upper_threshold, dtype=scalar_dtype)

            # delta 的 L2 范数
            delta_vlm_norm = jnp.mean(jnp.linalg.norm(jnp.reshape(delta_vlm, (batch_size, -1)), axis=-1))
            delta_state_norm = jnp.mean(jnp.linalg.norm(jnp.reshape(delta_state, (batch_size, -1)), axis=-1))

            return actions_final, {
                "effect_vlm": effect_vlm,
                "effect_state": effect_state,
                "use_base": use_base,
                # 效应比例权重
                "effect_ratio_vlm": w_vlm,
                "effect_ratio_state": w_state,
                # 实际应用权重
                "actual_vlm_weight": actual_vlm_weight,
                "actual_state_weight": actual_state_weight,
                # delta 范数
                "delta_vlm_norm": delta_vlm_norm,
                "delta_state_norm": delta_state_norm,
                # 配置参数
                "cf_guidance_scale": cf_guidance_scale_metric,
                "vlm_effect_threshold": vlm_effect_threshold_metric,
            }
        return actions_final
