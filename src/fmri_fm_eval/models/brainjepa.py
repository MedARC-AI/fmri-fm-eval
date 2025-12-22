"""
Brain-JEPA model wrapper
"""

import urllib.request
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model

try:
    from brain_jepa.models.vision_transformer import vit_base
except ImportError as exc:
    raise ImportError(
        "brain-jepa not installed. Please install the optional brain-jepa extra."
    ) from exc


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

    Preprocessing steps:
    1. Unnormalize BOLD data using mean and std
    2. Optional global normalization
    3. Resample to target TR if input TR differs from target
    4. Pad/crop input to target number of frames
    5. Reshape: (T, D) -> (1, D, T) for Brain-JEPA's Conv2d patch embedding
    """

    def __init__(
        self,
        num_frames: int = 160,
        target_tr: float = 2.0,
        use_normalization: bool = True,
    ):
        """
        Args:
            num_frames: Number of output frames after temporal sampling.
            target_tr: Target repetition time in seconds.
            use_normalization: Apply global mean/std normalization.
        """
        self.num_frames = num_frames
        self.target_tr = target_tr
        self.use_normalization = use_normalization

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample["bold"]  # (T, D) - normalized per-ROI
        mean = sample["mean"]  # (1, D) - original means
        std = sample["std"]  # (1, D) - original stds
        tr = sample["tr"]  # float - repetition time

        # Unnormalize BOLD data
        bold = bold * std + mean

        # Optional global normalization (Brain-JEPA style)
        # Normalization comes first in preprocessing pipeline
        # https://github.com/Eric-LRL/Brain-JEPA/blob/b1dfa93eb331d1e70476446b06fc8a1ac3c92345/src/datasets/hca_sex_datasets.py#L45
        # This should be enabled by default during evaluation
        # https://github.com/Eric-LRL/Brain-JEPA/blob/b1dfa93eb331d1e70476446b06fc8a1ac3c92345/scripts/classification/run_downstream_LP_hca_sex.sh#L17
        if self.use_normalization:
            mean = bold.mean()
            std = bold.std()
            bold = (bold - mean) / std

        # Resample to target TR if needed
        # Allow some wiggle room, since the TR doesn't have to be exact.
        if abs(tr - self.target_tr) >= 0.2:
            bold = self._resample_to_target_tr(bold, tr, self.target_tr)

        T, D = bold.shape

        # Pad with ROI mean if too short at the end of the time series
        if T < self.num_frames:
            roi_mean = bold.mean(dim=0)  # (D,)
            pad_size = self.num_frames - T
            padding = roi_mean.unsqueeze(0).repeat(pad_size, 1)  # (pad_size, D)
            bold = torch.cat([bold, padding], dim=0)  # (num_frames, D)
        # Clip if too long
        elif T > self.num_frames:
            bold = bold[: self.num_frames]

        # Transpose to (D, T) to match Brain-JEPA's internal format
        bold = bold.T  # (D, T)

        # Add channel dimension: (D, T) -> (1, D, T)
        bold = bold.unsqueeze(0)  # (1, D, T)

        # Update sample in place
        sample["bold"] = bold.contiguous().to(torch.float32)
        return sample

    def _resample_to_target_tr(self, bold: Tensor, tr: float, target_tr: float) -> Tensor:
        """
        Resample time series to target TR using linear interpolation.
        """
        T, D = bold.shape

        # Calculate new length
        duration = tr * T
        new_length = int(duration / target_tr)

        # Transpose to (D, T) for interpolation, then add batch and channel dims: (1, D, T)
        bold_t = bold.T.unsqueeze(0)  # (1, D, T)

        # Use 1D interpolation (works on last dimension)
        # Nearest interpolation for downsampling following official Brain-JEPA
        # implementation
        # https://github.com/Eric-LRL/Brain-JEPA/blob/b1dfa93eb331d1e70476446b06fc8a1ac3c92345/src/datasets/hca_sex_datasets.py#L88
        mode = "nearest" if target_tr > tr else "linear"
        bold_resampled = F.interpolate(
            bold_t,
            size=new_length,
            mode=mode,
        )  # (1, D, T_new)

        # Transpose back to (T_new, D)
        return bold_resampled.squeeze(0).T


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

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

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


def load_gradient_embeddings(gradient_csv_path: str | Path) -> Tensor:
    """Load gradient embeddings from CSV. Auto-downloads if path is None."""
    df = pd.read_csv(gradient_csv_path, header=None)
    gradient = torch.tensor(df.values, dtype=torch.float32)
    return gradient.unsqueeze(0)  # (1, 450, 30)


def load_brain_jepa_checkpoint(encoder: nn.Module, ckpt_path: str | Path) -> None:
    """
    Load Brain-JEPA pretrained checkpoint into encoder.
    """
    # Following official Brain-JEPA checkpoint loading
    # https://github.com/Eric-LRL/Brain-JEPA/blob/b1dfa93eb331d1e70476446b06fc8a1ac3c92345/downstream_tasks/main_linprobe.py#L97
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["target_encoder"]

    # Remove 'module.' prefix from keys (from DDP training)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    # we added the gradient_pos_embed as a buffer, so it's not in the original ckpt
    missing_keys, unexpected_keys = encoder.load_state_dict(new_state_dict, strict=False)
    assert not unexpected_keys and missing_keys == ["gradient_pos_embed"]


@register_model
def brain_jepa_vitb_ep300(
    attn_mode: str = "sdpa",
) -> tuple[BrainJEPATransform, BrainJEPAModelWrapper]:
    """Create Brain-JEPA model and transform. Auto-downloads files if paths are None."""
    # Match the pretrained checkpoint
    crop_size = (450, 160)
    patch_size = 16
    add_w = "mapping"  # match pretrained checkpoint (ukb_vitb_ep300.yaml)
    target_tr = 2.0

    # Load gradient positional embeddings
    gradient_csv_path = fetch_gradient_mapping()
    gradient_pos_embed = load_gradient_embeddings(gradient_csv_path)

    encoder = vit_base(
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=1,
        gradient_pos_embed=gradient_pos_embed,
        attn_mode=attn_mode,
        add_w=add_w,
    )

    ckpt_path = fetch_brain_jepa_checkpoint()
    print(f"Loading Brain-JEPA checkpoint from: {ckpt_path}")
    load_brain_jepa_checkpoint(encoder, ckpt_path)

    # Create wrapper
    model = BrainJEPAModelWrapper(encoder)

    transform = BrainJEPATransform(num_frames=crop_size[1], target_tr=target_tr)

    return transform, model
