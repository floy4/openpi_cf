"""Tests for the cf_attention module.

This module provides unit tests for the counterfactual attention functionality.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from openpi.models.cf_attention import (
    ModalityBounds,
    CfAttnMode,
    make_cf_attn_mask,
    create_prefix_suffix_attn_mask,
    get_modality_visibility,
    visualize_attn_mask,
    compute_modality_effect,
    create_modality_bounds,
)


class TestModalityBounds:
    """Tests for ModalityBounds dataclass."""
    
    def test_basic_creation(self):
        """Test basic ModalityBounds creation."""
        bounds = ModalityBounds(
            image_bounds={"base_0_rgb": (0, 100)},
            language_bounds=(100, 150),
            state_bounds=(150, 170),
            prefix_len=170,
            suffix_start=170,
        )
        
        assert bounds.total_image_tokens == 100
        assert bounds.total_language_tokens == 50
        assert bounds.total_state_tokens == 20
        assert bounds.has_state_tokens is True
    
    def test_empty_state_bounds(self):
        """Test with empty state bounds (Pi0 format)."""
        bounds = ModalityBounds(
            image_bounds={"base_0_rgb": (0, 100)},
            language_bounds=(100, 150),
            state_bounds=(150, 150),  # Empty
            prefix_len=150,
            suffix_start=150,
        )
        
        assert bounds.has_state_tokens is False
        assert bounds.total_state_tokens == 0
    
    def test_get_modality_at_position(self):
        """Test position-based modality identification."""
        bounds = ModalityBounds(
            image_bounds={"base_0_rgb": (0, 100), "wrist_0_rgb": (100, 200)},
            language_bounds=(200, 250),
            state_bounds=(250, 270),
            prefix_len=270,
            suffix_start=270,
        )
        
        # Image positions
        assert bounds.get_modality_at_position(50) == "image_base_0_rgb"
        assert bounds.get_modality_at_position(150) == "image_wrist_0_rgb"
        
        # Language positions
        assert bounds.get_modality_at_position(220) == "language"
        
        # State positions
        assert bounds.get_modality_at_position(260) == "state"
        
        # Action positions
        assert bounds.get_modality_at_position(280) == "action"
    
    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        bounds = ModalityBounds(
            image_bounds={"base_0_rgb": (0, 100)},
            language_bounds=(100, 150),
            state_bounds=(150, 170),
            prefix_len=170,
            suffix_start=170,
        )
        
        data = bounds.to_dict()
        restored = ModalityBounds.from_dict(data)
        
        assert restored.image_bounds == bounds.image_bounds
        assert restored.language_bounds == bounds.language_bounds
        assert restored.state_bounds == bounds.state_bounds
        assert restored.prefix_len == bounds.prefix_len
    
    def test_create_modality_bounds_helper(self):
        """Test helper function for creating bounds."""
        bounds = create_modality_bounds(
            image_token_counts={"base_0_rgb": 100, "wrist_0_rgb": 100},
            language_token_count=50,
            state_token_count=14,
            task_prefix_tokens=4,  # "Task: "
            state_prefix_tokens=8,  # ", State: "
            action_suffix_tokens=5,  # ";\nAction: "
        )
        
        assert bounds.total_image_tokens == 200
        assert bounds.image_bounds["base_0_rgb"] == (0, 100)
        assert bounds.image_bounds["wrist_0_rgb"] == (100, 200)


class TestCfAttnMode:
    """Tests for CfAttnMode and visibility functions."""
    
    def test_base_mode_visibility(self):
        """Test BASE mode has all modalities visible."""
        visibility = get_modality_visibility(CfAttnMode.BASE)
        
        assert visibility["image"] is True
        assert visibility["language"] is True
        assert visibility["state"] is True
    
    def test_no_image_mode(self):
        """Test NO_IMAGE mode blocks only images."""
        visibility = get_modality_visibility(CfAttnMode.NO_IMAGE)
        
        assert visibility["image"] is False
        assert visibility["language"] is True
        assert visibility["state"] is True
    
    def test_image_only_mode(self):
        """Test IMAGE_ONLY mode shows only images."""
        visibility = get_modality_visibility(CfAttnMode.IMAGE_ONLY)
        
        assert visibility["image"] is True
        assert visibility["language"] is False
        assert visibility["state"] is False
    
    def test_dual_modality_modes(self):
        """Test dual modality combination modes."""
        # IMAGE_LANG: block state
        visibility = get_modality_visibility(CfAttnMode.IMAGE_LANG)
        assert visibility["image"] is True
        assert visibility["language"] is True
        assert visibility["state"] is False
        
        # IMAGE_STATE: block language
        visibility = get_modality_visibility(CfAttnMode.IMAGE_STATE)
        assert visibility["image"] is True
        assert visibility["language"] is False
        assert visibility["state"] is True


class TestAttentionMask:
    """Tests for attention mask creation and modification."""
    
    def test_create_prefix_suffix_mask(self):
        """Test basic prefix-suffix mask creation."""
        mask = create_prefix_suffix_attn_mask(
            seq_len=20,
            prefix_len=10,
            batch_size=1,
        )
        
        # Check shape
        assert mask.shape == (1, 20, 20)
        
        # Check prefix bidirectional attention
        mask_np = np.array(mask[0])
        assert mask_np[:10, :10].all()  # All prefix tokens can attend to prefix
        
        # Check suffix can attend to prefix
        assert mask_np[10:, :10].all()  # All suffix can attend to prefix
        
        # Check causal attention in suffix
        # Token 10 can attend to tokens 10
        assert mask_np[10, 10] is True
        # Token 15 can attend to tokens 10-15
        assert mask_np[15, 10:16].all()
        # Token 15 cannot attend to token 16
        assert mask_np[15, 16] is np.False_
    
    def test_cf_mask_no_modification_for_base(self):
        """Test that BASE mode returns unchanged mask."""
        base_mask = create_prefix_suffix_attn_mask(20, 10)
        bounds = ModalityBounds(
            image_bounds={"cam": (0, 5)},
            language_bounds=(5, 8),
            state_bounds=(8, 10),
            prefix_len=10,
            suffix_start=10,
        )
        
        cf_mask = make_cf_attn_mask(base_mask, bounds, CfAttnMode.BASE)
        
        assert np.array_equal(np.array(base_mask), np.array(cf_mask))
    
    def test_cf_mask_blocks_image(self):
        """Test that NO_IMAGE mode blocks image positions."""
        base_mask = create_prefix_suffix_attn_mask(20, 10)
        bounds = ModalityBounds(
            image_bounds={"cam": (0, 5)},
            language_bounds=(5, 8),
            state_bounds=(8, 10),
            prefix_len=10,
            suffix_start=10,
        )
        
        cf_mask = np.array(make_cf_attn_mask(base_mask, bounds, CfAttnMode.NO_IMAGE))
        
        # Suffix should NOT be able to attend to image positions (0-5)
        assert not cf_mask[10:, 0:5].any()  # All False
        
        # Suffix should still attend to language and state
        assert cf_mask[10:, 5:10].all()  # All True
    
    def test_visualize_attn_mask(self):
        """Test mask visualization function."""
        mask = create_prefix_suffix_attn_mask(10, 5)
        visualization = visualize_attn_mask(mask, prefix_len=5)
        
        assert "Attention Mask" in visualization
        assert "Prefix length: 5" in visualization
        assert "(prefix end)" in visualization
        assert "(suffix start)" in visualization


class TestModalityEffect:
    """Tests for effect computation functions."""
    
    def test_l2_effect(self):
        """Test L2 distance effect computation."""
        baseline = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cf_actions = jnp.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])
        
        effect = compute_modality_effect(baseline, cf_actions, metric="l2")
        
        # First row should have L2 distance sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75)
        expected_first = np.sqrt(0.75)
        assert np.isclose(float(effect[0]), expected_first)
        
        # Second row should have 0 distance
        assert float(effect[1]) == 0.0
    
    def test_l1_effect(self):
        """Test L1 distance effect computation."""
        baseline = jnp.array([[1.0, 2.0, 3.0]])
        cf_actions = jnp.array([[1.5, 2.5, 3.5]])
        
        effect = compute_modality_effect(baseline, cf_actions, metric="l1")
        
        # L1 = 0.5 + 0.5 + 0.5 = 1.5
        assert float(effect[0]) == 1.5
    
    def test_cosine_effect(self):
        """Test cosine distance effect computation."""
        baseline = jnp.array([[1.0, 0.0, 0.0]])
        cf_actions = jnp.array([[0.0, 1.0, 0.0]])  # Orthogonal
        
        effect = compute_modality_effect(baseline, cf_actions, metric="cosine")
        
        # Cosine similarity = 0, distance = 1
        assert np.isclose(float(effect[0]), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])