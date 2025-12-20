"""
Brain-JEPA model wrapper for fmri-fm-eval.

Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking
Paper: https://arxiv.org/abs/2409.19407
Code: https://github.com/Eric-LRL/Brain-JEPA

This wrapper interfaces Brain-JEPA with the fmri-fm-eval evaluation suite.

Preprocessing Pipeline (matches original Brain-JEPA):
    1. Input: BOLD time series (T, 450) from schaefer400_tians3 parcellation
       - 400 cortical ROIs (Schaefer atlas)
       - 50 subcortical ROIs (Tian Scale III atlas)
    2. Center crop: Select middle clip of `sampling_rate * num_frames` frames
       (default: 3 * 160 = 480 frames, or T if shorter)
    3. Temporal sampling: Linearly sample to num_frames (default: 160)
    4. Reshape: (num_frames, 450) -> (1, 450, num_frames) for Conv2d patch embedding
    5. Optional: Global mean/std normalization (disabled by default)

Reference: Brain-JEPA/src/datasets/hca_sex_datasets.py
"""

import urllib.request
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model


# Cache directory for downloaded files
BRAIN_JEPA_CACHE_DIR = Path.home() / ".cache" / "fmri-fm-eval" / "brain-jepa"


def fetch_gradient_mapping() -> Path:
    """Download gradient_mapping_450.csv from GitHub with caching."""
    base_url = "https://github.com/Eric-LRL/Brain-JEPA/raw/main/data"
    filename = "gradient_mapping_450.csv"
    cache_dir = BRAIN_JEPA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / filename

    if not cached_file.exists():
        url = f"{base_url}/{filename}"
        try:
            urllib.request.urlretrieve(url, cached_file)
        except Exception as exc:
            raise ValueError(f"Download failed: {url}") from exc

    return cached_file


def fetch_brain_jepa_checkpoint() -> Path:
    """Download jepa-ep300.pth.tar from Google Drive with caching."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading Brain-JEPA checkpoint. "
            "Install with: pip install gdown"
        )

    # File ID from Brain-JEPA README.md (gdown 1LL3gM-i5SLDWCFyvj71M3peLeU6V2qMR)
    file_id = "1LL3gM-i5SLDWCFyvj71M3peLeU6V2qMR"
    cache_dir = BRAIN_JEPA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "jepa-ep300.pth.tar"

    if not cached_file.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(cached_file), quiet=False)

    return cached_file


class BrainJEPATransform:
    """
    Transform for Brain-JEPA model.

    Preprocessing steps (copied from original Brain-JEPA codebase):
    1. Temporal clip selection: center crop of `sampling_rate * num_frames` frames
    2. Temporal sampling: linearly sample to `num_frames` frames
    3. Reshape: (T, D) -> (1, D, T) for Brain-JEPA's Conv2d patch embedding
    4. Optional global normalization

    Input sample format (from fmri-fm-eval):
        - bold: (T, 450) - BOLD time series, normalized per-voxel to mean=0, std=1
        - mean: (1, 450) - original voxel means (unused)
        - std: (1, 450) - original voxel stds (unused)
        - tr: float - repetition time (unused)

    Output sample format:
        - bold: (1, 450, num_frames) - ready for Brain-JEPA encoder

    Reference: Brain-JEPA/src/datasets/hca_sex_datasets.py
    """

    def __init__(
        self,
        num_frames: int = 160,
        sampling_rate: int = 3,
        use_normalization: bool = False,
    ):
        """
        Args:
            num_frames: Number of output frames after temporal sampling. Default 160.
            sampling_rate: Ratio for clip size calculation (clip_size = sampling_rate * num_frames).
                Default 3, giving clip_size = 480 frames.
            use_normalization: Apply global mean/std normalization. Default False.
        """
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.use_normalization = use_normalization

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample["bold"]  # (T, D) - normalized per-ROI
        mean = sample["mean"]  # (1, D) - original means
        std = sample["std"]    # (1, D) - original stds
        
        # Unnormalize BOLD data
        bold = bold * std + mean
        
        T, D = bold.shape

        # Transpose to (D, T) to match Brain-JEPA's internal format
        bold = bold.T  # (D, T)

        # Apply temporal resampling if needed
        if T != self.num_frames:
            clip_size = self.sampling_rate * self.num_frames
            start_idx, end_idx = self._get_start_end_idx(T, clip_size)
            bold = self._temporal_sampling(bold, start_idx, end_idx, self.num_frames)

        # Add channel dimension: (D, T) -> (1, D, T)
        bold = bold.unsqueeze(0)  # (1, D, T)

        # Optional global normalization (Brain-JEPA style)
        if self.use_normalization:
            mean = bold.mean()
            std = bold.std()
            bold = (bold - mean) / (std + 1e-6)

        # Update sample in place
        sample["bold"] = bold.to(torch.float32)
        return sample

    def _get_start_end_idx(self, fmri_size: int, clip_size: int) -> tuple[float, float]:
        """
        Get start and end indices for center crop.

        For evaluation, we use deterministic center crop instead of random sampling.

        Reference: Brain-JEPA/src/datasets/hca_sex_datasets.py:_get_start_end_idx
        (Modified: uses center crop instead of random for evaluation)

        Args:
            fmri_size: Total number of frames in the fMRI sequence.
            clip_size: Desired clip size to extract.

        Returns:
            (start_idx, end_idx): Start and end frame indices.
        """
        clip_size = min(clip_size, fmri_size)  # Clamp to available frames
        delta = max(fmri_size - clip_size, 0)

        # Center crop for deterministic evaluation
        start_idx = delta / 2.0
        end_idx = start_idx + clip_size - 1

        return start_idx, end_idx

    def _temporal_sampling(
        self, frames: Tensor, start_idx: float, end_idx: float, num_samples: int
    ) -> Tensor:
        """
        Sample num_samples frames between start_idx and end_idx with equal interval.

        This is copied directly from the original Brain-JEPA codebase.

        Reference: Brain-JEPA/src/datasets/hca_sex_datasets.py:_temporal_sampling

        Args:
            frames: Tensor of shape (D, T) - ROIs x time points
            start_idx: Start frame index (can be float for interpolation)
            end_idx: End frame index (can be float for interpolation)
            num_samples: Number of frames to sample

        Returns:
            Tensor of shape (D, num_samples) - temporally sampled frames
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames


class BrainJEPAModelWrapper(nn.Module):
    """
    Wrapper for Brain-JEPA encoder model.

    Takes an input batch and returns embeddings in the format expected by fmri-fm-eval:
    - cls_embeds: None (Brain-JEPA doesn't use CLS token)
    - reg_embeds: None (no register tokens)
    - patch_embeds: (B, num_patches, embed_dim) - patch token embeddings

    The encoder processes input of shape (B, 1, 450, T) and outputs patch tokens.
    For vit_base with crop_size=(450, 160) and patch_size=16:
        num_patches = 450 * (160 / 16) = 450 * 10 = 4500
        embed_dim = 768
    """

    __space__: str = "schaefer400_tians3"

    def __init__(
        self,
        encoder: nn.Module,
        gradient_pos_embed: Tensor,
    ):
        super().__init__()
        self.encoder = encoder
        # Register gradient as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer("gradient_pos_embed", gradient_pos_embed)

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]  # (B, 1, D, T)

        # Forward through encoder to get patch tokens
        patch_tokens = self.encoder(x, masks=None, return_attention=False)
        # patch_tokens: (B, num_patches, embed_dim)

        # Brain-JEPA only produces patch tokens, no CLS or register tokens
        return Embeddings(
            cls_embeds=None,
            reg_embeds=None,
            patch_embeds=patch_tokens,
        )


def load_gradient_embeddings(
    gradient_csv_path: str | Path | None = None,
) -> Tensor:
    """Load gradient embeddings from CSV. Auto-downloads if path is None."""
    if gradient_csv_path is None:
        gradient_csv_path = fetch_gradient_mapping()
    gradient_csv_path = Path(gradient_csv_path)
    if not gradient_csv_path.exists():
        raise FileNotFoundError(
            f"Gradient CSV not found at {gradient_csv_path}. "
            "Please ensure the Brain-JEPA data directory is set up correctly."
        )
    df = pd.read_csv(gradient_csv_path, header=None)
    gradient = torch.tensor(df.values, dtype=torch.float32)
    return gradient.unsqueeze(0)  # (1, 450, 30)


def resolve_checkpoint_path(ckpt_path: str | Path | None) -> Path:
    """Resolve checkpoint path. Auto-downloads from Google Drive if None."""
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        if ckpt_path.exists():
            return ckpt_path
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Please provide a valid checkpoint path or ensure the default checkpoint is available."
        )

    return fetch_brain_jepa_checkpoint()


def build_brain_jepa_encoder(
    crop_size: tuple[int, int] = (450, 160),
    patch_size: int = 16,
    gradient_pos_embed: Tensor | None = None,
    attn_mode: str = "normal",
    add_w: str = "mapping",
    gradient_checkpointing: bool = False,
    device: torch.device | str = "cpu",
):
    """
    Build Brain-JEPA encoder model using ViT-Base architecture.

    This imports from the Brain-JEPA source code and initializes the encoder.

    Args:
        crop_size: (num_rois, num_frames) input size. Default (450, 160).
        patch_size: Temporal patch size. Default 16.
        gradient_pos_embed: Preloaded gradient embeddings tensor.
        attn_mode: Attention mode ('normal', 'flash_attn').
        add_w: Positional embedding mode ('origin', 'mapping').
            'mapping' is used by pretrained checkpoint.
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency.

    Returns:
        Initialized encoder model.
    """
    # Import Brain-JEPA's vision transformer
    import sys

    brain_jepa_src = BRAIN_JEPA_CACHE_DIR.parent / "Brain-JEPA"
    if not (brain_jepa_src.exists() and (brain_jepa_src / "src" / "models" / "vision_transformer.py").exists()):
        raise FileNotFoundError(
            f"Brain-JEPA repository not found at {brain_jepa_src}. "
            "Please clone the repository: "
            "git clone https://github.com/Eric-LRL/Brain-JEPA.git"
        )
    
    brain_jepa_src = str(brain_jepa_src)

    if brain_jepa_src not in sys.path:
        sys.path.insert(0, brain_jepa_src)

    import src.models.vision_transformer as vit

    # Build encoder (always use vit_base)
    encoder = vit.vit_base(
        patch_size=patch_size,
        img_size=(crop_size[0], crop_size[1]),
        in_chans=1,
        gradient_pos_embed=gradient_pos_embed,
        attn_mode=attn_mode,
        add_w=add_w,
        gradient_checkpointing=gradient_checkpointing,
    )
    return encoder


def load_brain_jepa_checkpoint(
    encoder: nn.Module,
    ckpt_path: str | Path,
) -> nn.Module:
    """
    Load Brain-JEPA pretrained checkpoint into encoder.

    Brain-JEPA checkpoints store the EMA encoder under the 'target_encoder' key.
    Keys have 'module.' prefix that needs to be removed (from DDP training).

    Args:
        encoder: Initialized encoder model to load weights into.
        ckpt_path: Path to checkpoint file (.pth.tar).

    Returns:
        Encoder with loaded weights.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Brain-JEPA stores EMA encoder as 'target_encoder'
    if "target_encoder" in checkpoint:
        state_dict = checkpoint["target_encoder"]
    elif "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
    else:
        # Assume the checkpoint is just the state dict
        state_dict = checkpoint

    # Remove 'module.' prefix from keys (from DDP training)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    # Load state dict
    msg = encoder.load_state_dict(new_state_dict, strict=False)
    if msg.missing_keys:
        print(f"Warning: Missing keys when loading Brain-JEPA checkpoint: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"Warning: Unexpected keys in Brain-JEPA checkpoint: {msg.unexpected_keys}")

    return encoder


@register_model
def brain_jepa(
    ckpt_path: str | Path | None = None,
    gradient_csv_path: str | Path | None = None,
) -> tuple[BrainJEPATransform, BrainJEPAModelWrapper]:
    """Create Brain-JEPA model and transform. Auto-downloads files if paths are None."""
    # Match the pretrained checkpoint
    crop_size = (450, 160)
    patch_size = 16
    attn_mode = "normal"
    add_w = "mapping"  # match pretrained checkpoint (ukb_vitb_ep300.yaml)
    gradient_checkpointing = False
    use_normalization = False

    # Load gradient positional embeddings
    gradient_pos_embed = load_gradient_embeddings(gradient_csv_path)

    # Build encoder (always uses vit_base)
    encoder = build_brain_jepa_encoder(
        crop_size=crop_size,
        patch_size=patch_size,
        gradient_pos_embed=gradient_pos_embed,
        attn_mode=attn_mode,
        add_w=add_w,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Resolve and load checkpoint (raises error if not found)
    resolved_ckpt = resolve_checkpoint_path(ckpt_path)
    print(f"Loading Brain-JEPA checkpoint from: {resolved_ckpt}")
    encoder = load_brain_jepa_checkpoint(encoder, resolved_ckpt)

    # Create wrapper
    model = BrainJEPAModelWrapper(encoder, gradient_pos_embed)

    # Create transform
    transform = BrainJEPATransform(
        num_frames=crop_size[1],
        use_normalization=use_normalization,
    )

    return transform, model
