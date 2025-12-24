"""
BrainLM model wrapper

Supports three model sizes from HuggingFace Hub (vandijklab/brainlm):
- brainlm_13m: Legacy 13M parameter model
- brainlm_vitmae_111m: Vision Transformer MAE 111M parameter model
- brainlm_vitmae_650m: Vision Transformer MAE 650M parameter model

All models use 200 timepoints with sliding window evaluation.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model

brainlm_repo_path = os.getenv("BRAINLM_REPO_PATH")
if brainlm_repo_path:
    sys.path.append(brainlm_repo_path)

try:
    from brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
except ImportError as exc:
    raise ImportError(
        "BrainLM code not found. Set BRAINLM_REPO_PATH to the BrainLM repo path "
        "(git clone https://github.com/vandijklab/BrainLM.git)."
    ) from exc


# HuggingFace repository
BRAINLM_REPO = "vandijklab/brainlm"

# Model subdirectories in the HuggingFace repo
BRAINLM_VARIANTS = {
    "13m": "old_13M",
    "111m": "vitmae_111M",
    "650m": "vitmae_650M",
}


class BrainLMTransform:
    """
    Transform for BrainLM model - matches exact training preprocessing.

    Based on train_vit_mae_on_fMRI_images.py preprocess_images() function.

    Preprocessing steps:
    0. Unnormalize per-voxel z-scored data using mean/std
    1. Apply voxelwise RobustScaler normalization (median/IQR across time)
    2. Extract sliding windows (200 timepoints each, stride=200)
    3. Transpose: (T, D) -> (D, T)
    4. Reorder voxels by Y coordinate
    5. Scale by max_val (dataset-specific, default for RobustScaler normalization)
    6. Repeat for 3 channels (R,G,B) for ViTMAE variants

    Output: (num_windows, 3, 424, 200) - model handles padding to (3, 432, 432)
    """

    def __init__(
        self,
        coords_dataset_path: str,
        num_timepoints: int = 200,
        window_stride: int = 200,
        max_val_to_scale: float = 5.6430855,  # Default for RobustScaler normalization
        repeat_channels: bool = True,
    ):
        """
        Args:
            coords_dataset_path: Path to brain region coordinates dataset for voxel reordering.
            num_timepoints: Number of timepoints per window (BrainLM uses 200).
            window_stride: Stride for sliding windows (use 200 for non-overlapping).
            max_val_to_scale: Max value for scaling - DATASET-SPECIFIC!
                              Default 5.6430855 is for RobustScaler normalized data.
                              Will be different for z-score normalized data.
        """
        self.num_timepoints = num_timepoints
        self.window_stride = window_stride
        self.max_val_to_scale = max_val_to_scale
        self.repeat_channels = repeat_channels

        # Load voxel reordering indices from coords dataset
        from datasets import load_from_disk

        coords_ds = load_from_disk(coords_dataset_path)
        voxel_y_coords = coords_ds["Y"]
        self.reorder_indices = sorted(
            range(len(voxel_y_coords)), key=lambda k: voxel_y_coords[k]
        )
        voxel_x_coords = coords_ds["X"]
        voxel_z_coords = coords_ds["Z"]
        coords = torch.tensor(
            [voxel_x_coords, voxel_y_coords, voxel_z_coords], dtype=torch.float32
        ).T
        self.xyz_vectors = coords[self.reorder_indices]

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor] | None:
        bold = sample["bold"]  # (T, D) - z-score normalized data
        mean = sample["mean"]  # (1, D)
        std = sample["std"]  # (1, D)

        # Convert z-scored data back to raw signal, then apply voxelwise RobustScaler.
        bold = bold.to(torch.float32) * std.to(torch.float32) + mean.to(torch.float32)
        median = torch.quantile(bold, 0.5, dim=0, keepdim=True)
        q1 = torch.quantile(bold, 0.25, dim=0, keepdim=True)
        q3 = torch.quantile(bold, 0.75, dim=0, keepdim=True)
        iqr = q3 - q1
        eps = 1e-6
        valid_mask = iqr > eps
        bold = (bold - median) / iqr.clamp(min=eps)
        bold = bold * valid_mask

        T, D = bold.shape

        # Create sliding windows with deterministic stride
        num_windows = T // self.num_timepoints
        windows = []

        for i in range(num_windows):
            start_idx = i * self.window_stride
            end_idx = start_idx + self.num_timepoints

            if end_idx > T:
                break  # Discard incomplete windows (no padding)

            window = bold[start_idx:end_idx, :]  # (200, 424)

            # Transpose: (T, D) -> (D, T)
            window = window.T  # (424, 200)

            # Reorder voxels by Y coordinate (critical for matching training!)
            window = window[self.reorder_indices, :]  # (424, 200)

            # Scale by max_val (dataset-specific!)
            window = window / self.max_val_to_scale

            # Repeat for 3 channels (R,G,B) for ViTMAE variants.
            if self.repeat_channels:
                window = window.unsqueeze(0).repeat(3, 1, 1)  # (3, 424, 200)

            # Model handles padding (424,200) -> (432,432) internally
            windows.append(window.contiguous().to(torch.float32))

        if len(windows) == 0:
            # Not enough timepoints for even one window
            return None

        # Stack multiple windows from same recording
        sample["bold"] = (
            torch.stack(windows) if len(windows) > 1 else windows[0].unsqueeze(0)
        )
        # Shape: (num_windows, 3, 424, 200)
        sample["xyz"] = self.xyz_vectors

        return sample


class BrainLMModelWrapper(nn.Module):
    """
    Wrapper for BrainLM encoder model.

    Takes input batch and returns embeddings in the format expected by fmri-fm-eval:
    - cls_embeds: (B, embed_dim) - CLS token embeddings
    - reg_embeds: None (no register tokens)
    - patch_embeds: (B, num_patches, embed_dim) - patch token embeddings

    Input shape: (B, 3, 424, 200) where B = num_windows * batch_size
    Model pads to: (B, 3, 432, 432)
    embed_dim: 768 (111M), 1280 (650M), 512 (13M)
    """

    __space__: str = "a424"

    def __init__(
        self,
        backbone: nn.Module,
        model_type: str = "vitmae",  # "vitmae" or "brainlm"
    ):
        super().__init__()
        self.backbone = backbone
        self.model_type = model_type

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        pixel_values = batch["bold"]

        if self.model_type == "vitmae":
            pixel_values = pixel_values.squeeze(0)
            return self._forward_vitmae(pixel_values)

        return self._forward_brainlm(batch)

    def _forward_vitmae(self, pixel_values: Tensor) -> Embeddings:
        """Forward pass for ViTMAE models (111M, 650M)."""
        # Set mask_ratio to 0 to disable masking during inference
        original_mask_ratio = self.backbone.vit.embeddings.config.mask_ratio
        self.backbone.vit.embeddings.config.mask_ratio = 0.0

        with torch.set_grad_enabled(False):
            outputs = self.backbone(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            sequence_output = outputs.hidden_states[-1]

        # Split CLS and patch tokens
        cls_embeds = sequence_output[:, 0, :]  # (B, embed_dim)
        patch_embeds = sequence_output[:, 1:, :]  # (B, num_patches, embed_dim)

        return Embeddings(
            cls_embeds=cls_embeds,
            reg_embeds=None,
            patch_embeds=patch_embeds,
        )

    def _forward_brainlm(self, batch: dict[str, Tensor]) -> Embeddings:
        """Forward pass for legacy BrainLM model (13M)."""
        # Set mask_ratio to 0 to disable masking during inference
        self.backbone.vit.embeddings.config.mask_ratio = 0.0

        bold = batch["bold"]
        signal_vectors = bold[:, 0, :, :]
        batch_size = signal_vectors.shape[0]

        xyz_vectors = batch.get("xyz")
        if xyz_vectors is None:
            raise ValueError("Missing xyz vectors for BrainLM 13M model input.")

        if xyz_vectors.ndim == 2:
            xyz_vectors = xyz_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        elif xyz_vectors.shape[0] != batch_size:
            raise ValueError("xyz batch size does not match signal batch size.")

        with torch.set_grad_enabled(False):
            encoder_outputs = self.backbone.vit(
                signal_vectors=signal_vectors,
                xyz_vectors=xyz_vectors,
                output_hidden_states=True,
                return_dict=True,
            )
            sequence_output = encoder_outputs.last_hidden_state
            # sequence_output: (B, 1 + num_patches, embed_dim)

        # Split CLS and patch tokens
        cls_embeds = sequence_output[:, 0, :]  # (B, embed_dim)
        patch_embeds = sequence_output[:, 1:, :]  # (B, num_patches, embed_dim)

        return Embeddings(
            cls_embeds=cls_embeds,
            reg_embeds=None,
            patch_embeds=patch_embeds,
        )


def load_brainlm_from_hf(
    variant: str = "111m",
    cache_dir: Optional[str | Path] = None,
) -> tuple[nn.Module, str]:
    """
    Load BrainLM model directly from HuggingFace Hub.

    Args:
        variant: Model variant - one of "13m", "111m", "650m"
        cache_dir: Optional cache directory for downloaded files

    Returns:
        Tuple of (loaded model, model_type) where model_type is "vitmae" or "brainlm"
    """
    if variant not in BRAINLM_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant}. Choose from: {list(BRAINLM_VARIANTS.keys())}"
        )

    subfolder = BRAINLM_VARIANTS[variant]

    print(f"Loading BrainLM {variant} from HuggingFace Hub...")

    if variant in ["111m", "650m"]:
        # ViTMAE models (111M, 650M)
        model = ViTMAEForPreTraining.from_pretrained(
            BRAINLM_REPO,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )
        model_type = "vitmae"
    else:
        # Legacy BrainLM model (13M)
        model = BrainLMForPretraining.from_pretrained(
            BRAINLM_REPO,
            subfolder=subfolder,
            cache_dir=cache_dir,
        )
        model_type = "brainlm"

    model.config.train_mode = "auto_encode"

    return model, model_type


def create_brainlm_model(
    coords_dataset_path: str,
    variant: str = "111m",
    max_val_to_scale: float = 5.6430855,
    cache_dir: Optional[str | Path] = None,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """
    Create BrainLM model and transform (loads from HuggingFace Hub: vandijklab/brainlm).

    Args:
        coords_dataset_path: Path to brain region coordinates dataset.
        variant: Model variant - one of "13m", "111m", "650m"
        max_val_to_scale: Max value for scaling - dataset-specific!
                          Default 5.6430855 is for RobustScaler normalized data.
        cache_dir: Optional cache directory for HuggingFace downloads.

    Returns:
        Tuple of (transform, model wrapper)
    """
    # Load from HuggingFace Hub
    backbone, model_type = load_brainlm_from_hf(variant, cache_dir)

    # Create wrapper
    model = BrainLMModelWrapper(backbone, model_type=model_type)

    # Create transform
    transform = BrainLMTransform(
        num_timepoints=200,
        window_stride=200,
        max_val_to_scale=max_val_to_scale,
        coords_dataset_path=coords_dataset_path,
        repeat_channels=variant in ["111m", "650m"],
    )

    return transform, model


@register_model
def brainlm_13m(
    coords_dataset_path: str,
    max_val_to_scale: float = 5.6430855,
    cache_dir: Optional[str | Path] = None,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """Legacy BrainLM 13M parameter model (from HuggingFace: vandijklab/brainlm)."""
    return create_brainlm_model(
        coords_dataset_path, "13m", max_val_to_scale, cache_dir
    )


@register_model
def brainlm_vitmae_111m(
    coords_dataset_path: str,
    max_val_to_scale: float = 5.6430855,
    cache_dir: Optional[str | Path] = None,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """BrainLM ViT-MAE 111M parameter model (from HuggingFace: vandijklab/brainlm)."""
    return create_brainlm_model(
        coords_dataset_path, "111m", max_val_to_scale, cache_dir
    )


@register_model
def brainlm_vitmae_650m(
    coords_dataset_path: str,
    max_val_to_scale: float = 5.6430855,
    cache_dir: Optional[str | Path] = None,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """BrainLM ViT-MAE 650M parameter model (from HuggingFace: vandijklab/brainlm)."""
    return create_brainlm_model(
        coords_dataset_path, "650m", max_val_to_scale, cache_dir
    )
