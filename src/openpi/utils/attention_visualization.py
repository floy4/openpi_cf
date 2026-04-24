"""Attention visualization utilities for counterfactual analysis.

Provides functions to create and save attention heatmaps overlaid on original images.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_attention_heatmap(
    attention_map: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Create attention heatmap overlaid on original image.

    Args:
        attention_map: 2D attention weights [H, W] or flattened [N].
            Values should be normalized (0-1) for best visualization.
        original_image: RGB image [H, W, 3] in range 0-255.
        alpha: Transparency for heatmap overlay (0-1).
        colormap: Matplotlib colormap name (e.g., "jet", "hot", "viridis").

    Returns:
        Blended image [H, W, 3] with attention heatmap overlay.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for attention visualization")

    # Handle flattened attention
    if attention_map.ndim == 1:
        # Assume square grid if possible
        n_tokens = attention_map.shape[0]
        grid_size = int(np.sqrt(n_tokens))
        if grid_size * grid_size == n_tokens:
            attention_map = attention_map.reshape(grid_size, grid_size)
        else:
            # Not a perfect square, reshape to match image aspect ratio
            h, w = original_image.shape[:2]
            attention_map = attention_map.reshape(1, -1)  # Row vector

    # Normalize attention to 0-1
    attn_min = attention_map.min()
    attn_max = attention_map.max()
    if attn_max > attn_min:
        attention_map = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attention_map = np.zeros_like(attention_map)

    # Resize attention map to match image size if needed
    if attention_map.shape != original_image.shape[:2]:
        from scipy.ndimage import zoom
        scale_h = original_image.shape[0] / attention_map.shape[0]
        scale_w = original_image.shape[1] / attention_map.shape[1]
        attention_map = zoom(attention_map, (scale_h, scale_w), order=1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(attention_map)
    heatmap_rgb = heatmap_rgba[:, :, :3] * 255  # Convert to 0-255 range

    # Blend with original image
    original_float = original_image.astype(np.float32)
    heatmap_float = heatmap_rgb.astype(np.float32)
    blended = original_float * (1 - alpha) + heatmap_float * alpha

    return blended.astype(np.uint8)


def save_attention_visualization(
    attention_data: dict[str, Any],
    images: dict[str, np.ndarray],
    output_dir: str,
    step_idx: int,
    episode_idx: int | None = None,
    layer_index: int | None = None,
    cf_mode: str | None = None,
) -> list[str]:
    """Save attention visualization for all camera views.

    Args:
        attention_data: Dict containing attention maps and metadata.
            Expected keys: "last_query_attn_head_avg", "image_token_bounds".
        images: Dict of camera images {camera_name: image_array}.
            Each image should be [H, W, 3] in range 0-255.
        output_dir: Directory to save visualization outputs.
        step_idx: Current step index in the episode.
        episode_idx: Optional episode index for filename.
        layer_index: Transformer layer index (for filename).
        cf_mode: CF mode name (for filename).

    Returns:
        List of saved file paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Extract attention per camera
    attn_map = attention_data.get("last_query_attn_head_avg")
    image_bounds = attention_data.get("image_token_bounds", {})

    if attn_map is None:
        return saved_files

    # Convert to numpy if needed (from JAX)
    if hasattr(attn_map, "to_numpy"):
        attn_map = attn_map.to_numpy()
    attn_map = np.asarray(attn_map)

    # Handle batch dimension
    if attn_map.ndim == 3:
        attn_map = attn_map[0]  # Take first batch item

    # Process each camera
    for camera_name, image in images.items():
        if hasattr(image, "to_numpy"):
            image = image.to_numpy()
        image = np.asarray(image)

        # Get attention for this camera
        if camera_name in image_bounds:
            bounds = image_bounds[camera_name]
            if hasattr(bounds, "to_numpy"):
                bounds = bounds.to_numpy()
            bounds = np.asarray(bounds)
            start, end = int(bounds[0]), int(bounds[1])
            camera_attn = attn_map[start:end]
        else:
            # Use full attention if bounds not available
            camera_attn = attn_map

        # Create heatmap
        try:
            blended = create_attention_heatmap(camera_attn, image, alpha=0.4)
        except Exception:
            continue

        # Build filename
        parts = [camera_name]
        if episode_idx is not None:
            parts.append(f"ep{episode_idx}")
        parts.append(f"step{step_idx}")
        if layer_index is not None:
            parts.append(f"layer{layer_index}")
        if cf_mode is not None:
            parts.append(f"cf_{cf_mode}")
        filename = "_".join(parts) + ".png"

        filepath = output_path / filename
        plt.imsave(str(filepath), blended)
        saved_files.append(str(filepath))

    return saved_files


def visualize_topk_regions(
    attention_map: np.ndarray,
    original_image: np.ndarray,
    topk_indices: np.ndarray,
    output_path: str,
    title: str | None = None,
) -> str:
    """Visualize top-k attended regions highlighted on image.

    Args:
        attention_map: Full attention map [N] or [H, W].
        original_image: RGB image [H, W, 3].
        topk_indices: Indices of top-k attended positions.
        output_path: Path to save visualization.
        title: Optional title for the plot.

    Returns:
        Path to saved visualization.
    """
    if not HAS_MATPLOTLIB:
        return ""

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: attention heatmap
    if attention_map.ndim == 1:
        n = attention_map.shape[0]
        grid_size = int(np.sqrt(n))
        if grid_size * grid_size == n:
            attention_map = attention_map.reshape(grid_size, grid_size)
    axes[0].imshow(attention_map, cmap="jet")
    axes[0].set_title("Attention Map")
    axes[0].axis("off")

    # Right: original image with top-k highlighted
    axes[1].imshow(original_image)
    h, w = original_image.shape[:2]

    # Mark top-k positions
    if attention_map.ndim == 2:
        grid_h, grid_w = attention_map.shape
        patch_h = h / grid_h
        patch_w = w / grid_w
        for idx in topk_indices:
            row = idx // grid_w
            col = idx % grid_w
            y = row * patch_h
            x = col * patch_w
            rect = plt.Rectangle(
                (x, y), patch_w, patch_h,
                fill=False, edgecolor="red", linewidth=2
            )
            axes[1].add_patch(rect)

    axes[1].set_title(f"Top-{len(topk_indices)} Attended Regions")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path