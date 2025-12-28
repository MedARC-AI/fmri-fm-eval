"""
BrainLM model wrapper

Supports three model sizes from HuggingFace Hub (vandijklab/brainlm):
- brainlm_13m: Legacy 13M parameter model
- brainlm_vitmae_111m: Vision Transformer MAE 111M parameter model
- brainlm_vitmae_650m: Vision Transformer MAE 650M parameter model

All models use 200 timepoints with sliding window evaluation.
"""

import importlib.resources

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model

try:
    from brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    import brainlm_toolkit
except ImportError as exc:
    raise ImportError("brainlm not installed. Please install the optional brainlm extra.") from exc


# HuggingFace repository
BRAINLM_REPO = "vandijklab/brainlm"

# Model subdirectories in the HuggingFace repo
BRAINLM_VARIANTS = {
    "13m": "old_13M",
    "111m": "vitmae_111M",
    "650m": "vitmae_650M",
}

DEFAULT_MAX_VAL_TO_SCALE = 5.6430855


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
        num_timepoints: int = 200,
        max_val_to_scale: float = DEFAULT_MAX_VAL_TO_SCALE,  # Default for RobustScaler normalization
        repeat_channels: bool = True,
    ):
        """
        Args:
            num_timepoints: Number of timepoints per window (BrainLM uses 200).
            max_val_to_scale: Max value for scaling - DATASET-SPECIFIC!
                              Default 5.6430855 is for RobustScaler normalized data.
                              Will be different for z-score normalized data.
        """
        self.num_timepoints = num_timepoints
        self.max_val_to_scale = max_val_to_scale
        self.repeat_channels = repeat_channels

        # Load voxel reordering indices from coords dataset
        coords_ds = load_a424_coords()  # (424, 4), cols [Index, X, Y, Z]
        # TODO: reference in original brainlm code for this reordering?
        # Is this the correct order for both model variants?
        self.reorder_indices = np.argsort(coords_ds["Y"])
        self.xyz_vectors = torch.from_numpy(coords_ds.loc[:, ["X", "Y", "Z"]].values)
        self.xyz_vectors = self.xyz_vectors[self.reorder_indices]

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor] | None:
        bold = sample["bold"]  # (T, D) - z-score normalized data
        mean = sample["mean"]  # (1, D)
        std = sample["std"]  # (1, D)

        # Convert z-scored data back to raw signal, then apply voxelwise RobustScaler.
        bold = bold * std + mean
        bold = robust_scale(bold)

        # TODO: temporal resample?

        T, D = bold.shape

        # Pad with zero if too short
        if T < self.num_timepoints:
            bold = F.pad(bold, (0, 0, 0, T - self.num_timepoints))
            T = len(bold)

        # Create sliding windows with non-overlapping stride
        num_windows = T // self.num_timepoints
        T = num_windows * self.num_timepoints
        bold = bold[:T, :].reshape(num_windows, self.num_timepoints, D)

        # Transpose [W, T, D] -> [W, D, T]
        bold = bold.transpose(1, 2)

        # Reorder voxels by Y coordinate (critical for matching training!)
        # TODO: reference in original brainlm code?
        bold = bold[:, self.reorder_indices]

        # Scale by max_val
        # TODO: where does this hard-coded value come from? dataset specific
        # normalization constants are not supported.
        bold = bold / self.max_val_to_scale

        # Expand channels dimension [W, C, D, T]
        bold = bold.unsqueeze(1)
        if self.repeat_channels:
            bold = bold.repeat(1, 3, 1, 1)

        sample["bold"] = bold  # [W, C, D, T]
        sample["xyz"] = self.xyz_vectors.repeat(num_windows, 1, 1)  # [W, D, 3]

        return sample


def load_a424_coords() -> pd.DataFrame:
    """
    Load BrainLM A424 brain coordinates, shape (424, 4). Columns are ["Index", "X", "Y",
    "Z"] with indices starting at 1.

    https://github.com/vandijklab/BrainLM/blob/eded39c86c27e03f5ead1d6a14311e92d1305e5e/toolkit/BrainLM_Toolkit.py#L334
    """
    files = importlib.resources.files(brainlm_toolkit)
    with files.joinpath("atlases/A424_Coordinates.dat").open() as f:
        coords = np.loadtxt(f, dtype=np.float32)
    coords = pd.DataFrame(coords, columns=["Index", "X", "Y", "Z"])
    return coords


def robust_scale(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    T, D = x.shape
    # do all quantiles at once to avoid sorting multiple times.
    q = torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype, device=x.device)
    q1, median, q3 = torch.quantile(x, q, dim=0, keepdim=True)
    iqr = q3 - q1
    valid_mask = iqr > eps
    x = (x - median) / iqr.clamp(min=eps)
    x = x * valid_mask
    return x


class BrainLMModelWrapper(nn.Module):
    """
    Wrapper for BrainLM encoder model.

    Takes input batch and returns embeddings in the format expected by fmri-fm-eval:
    - cls_embeds: (B, 1, embed_dim) - CLS token embeddings
    - reg_embeds: None (no register tokens)
    - patch_embeds: (B, num_patches, embed_dim) - patch token embeddings

    Input shape: (B, W, 3, 424, 200) where W = num_windows
    Model pads to: (*, 3, 432, 432)
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
        # Set mask_ratio to 0 to disable masking during inference
        # TODO: check that setting this like this works?
        self.backbone.vit.embeddings.config.mask_ratio = 0.0

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        pixel_values = batch["bold"]

        # Flatten windows together
        B, W, C, D, T = pixel_values.shape
        pixel_values = pixel_values.flatten(0, 1)

        if self.model_type == "vitmae":
            sequence_output = self._forward_vitmae(pixel_values)
        else:
            xyz_vectors = batch["xyz"].flatten(0, 1)
            sequence_output = self._forward_brainlm(pixel_values, xyz_vectors)

        # Unflatten windows
        _, N, C = sequence_output.shape
        sequence_output = sequence_output.reshape(B, W, N, C)

        # Split CLS and patch tokens
        cls_embeds = sequence_output[:, :, :1, :].mean(dim=1)  # (B, 1, C)
        patch_embeds = sequence_output[:, :, 1:, :].flatten(1, 2)  # (B, W * N, C)

        return Embeddings(
            cls_embeds=cls_embeds,
            reg_embeds=None,
            patch_embeds=patch_embeds,
        )

    def _forward_vitmae(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _forward_brainlm(
        self, pixel_values: torch.Tensor, xyz_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for legacy BrainLM model (13M)."""
        B, C, D, T = pixel_values.shape
        signal_vectors = pixel_values[:, 0, :, :]

        outputs = self.backbone.vit(
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.last_hidden_state


def load_brainlm_from_hf(
    variant: str = "111m",
) -> tuple[nn.Module, str]:
    """
    Load BrainLM model directly from HuggingFace Hub.

    Args:
        variant: Model variant - one of "13m", "111m", "650m"

    Returns:
        Tuple of (loaded model, model_type) where model_type is "vitmae" or "brainlm"
    """
    if variant not in BRAINLM_VARIANTS:
        raise ValueError(f"Unknown variant {variant}. Choose from: {list(BRAINLM_VARIANTS.keys())}")

    subfolder = BRAINLM_VARIANTS[variant]

    print(f"Loading BrainLM {variant} from HuggingFace Hub...")

    if variant in ["111m", "650m"]:
        # ViTMAE models (111M, 650M)
        model = ViTMAEForPreTraining.from_pretrained(BRAINLM_REPO, subfolder=subfolder)
        model_type = "vitmae"
    else:
        # Legacy BrainLM model (13M)
        model = BrainLMForPretraining.from_pretrained(BRAINLM_REPO, subfolder=subfolder)
        model_type = "brainlm"

    model.config.train_mode = "auto_encode"

    return model, model_type


def create_brainlm_model(
    variant: str = "111m",
    max_val_to_scale: float = DEFAULT_MAX_VAL_TO_SCALE,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """
    Create BrainLM model and transform (loads from HuggingFace Hub: vandijklab/brainlm).

    Args:
        variant: Model variant - one of "13m", "111m", "650m"
        max_val_to_scale: Max value for scaling - dataset-specific!
                          Default 5.6430855 is for RobustScaler normalized data.

    Returns:
        Tuple of (transform, model wrapper)
    """
    # Load from HuggingFace Hub
    backbone, model_type = load_brainlm_from_hf(variant)

    # Create wrapper
    model = BrainLMModelWrapper(backbone, model_type=model_type)

    # Create transform
    transform = BrainLMTransform(
        num_timepoints=200,
        max_val_to_scale=max_val_to_scale,
        repeat_channels=variant in ["111m", "650m"],
    )

    return transform, model


@register_model
def brainlm_13m(
    max_val_to_scale: float = DEFAULT_MAX_VAL_TO_SCALE,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """Legacy BrainLM 13M parameter model (from HuggingFace: vandijklab/brainlm)."""
    return create_brainlm_model("13m", max_val_to_scale)


@register_model
def brainlm_vitmae_111m(
    max_val_to_scale: float = DEFAULT_MAX_VAL_TO_SCALE,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """BrainLM ViT-MAE 111M parameter model (from HuggingFace: vandijklab/brainlm)."""
    return create_brainlm_model("111m", max_val_to_scale)


@register_model
def brainlm_vitmae_650m(
    max_val_to_scale: float = DEFAULT_MAX_VAL_TO_SCALE,
) -> tuple[BrainLMTransform, BrainLMModelWrapper]:
    """BrainLM ViT-MAE 650M parameter model (from HuggingFace: vandijklab/brainlm)."""
    return create_brainlm_model("650m", max_val_to_scale)
