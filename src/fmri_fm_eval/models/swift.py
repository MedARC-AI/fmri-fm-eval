"""

SwiFT: Swin 4D fMRI Transformer

"""

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model
from pathlib import Path
import numpy as np
import torch
from fmri_fm_eval import nisc
from einops import rearrange
import templateflow.api as tflow


try:
    from swiftfmri.pl_classifier import LitClassifier
except ImportError as exc:
    raise ImportError(
        "swiftfmri not installed. Please install the optional swiftfmri extra with `uv sync --extra swift`"
    ) from exc


# Cache directory for downloaded files
SWIFT_CACHE_DIR = Path.home() / ".cache" / "fmri-fm-eval" / "swift"


def fetch_swift_checkpoint() -> Path:
    """Download contrastive_pretrained.ckpt from Google Drive with caching."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading SwiFT checkpoint. Install with: pip install gdown"
        )

    # File ID from SwiFT README.md (gdown 11u4GGeTB361X01sge86U7JbGyEzZC7KJ)
    file_id = "11u4GGeTB361X01sge86U7JbGyEzZC7KJ"
    cache_dir = SWIFT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "contrastive_pretrained.ckpt"

    if not cached_file.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(cached_file), quiet=False)

    return cached_file


# Dummy datamodule to initialize LitClassifier
class _DummyTrainDataset:
    target_values = np.zeros((32, 1), dtype=np.float32)


class _DummyDataModule:
    def __init__(self):
        self.train_dataset = _DummyTrainDataset()


class SwiftWrapper(nn.Module):
    __space__: str = "mni"

    def __init__(self, ckpt_path: Path) -> None:
        super().__init__()

        self.ckpt_path = ckpt_path

        dm = _DummyDataModule()

        lit = LitClassifier.load_from_checkpoint(
            str(ckpt_path),
            data_module=dm,
            map_location="cpu",
            label_scaling_method="standardization",
            strict=False,  # emb_mlp and model.head are not mapped because these old checkpoint keys are not present in the new codebase.
        )

        self.backbone = lit.model
        self.expected_seq_len = 20

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        B, C, H, W, D, T = x.shape

        # handle sliding windows
        num_windows = T // self.expected_seq_len
        T = num_windows * self.expected_seq_len
        x = rearrange(x[..., :T], "b c x y z (w t) -> (b w) c x y z t", w=num_windows)

        feats = self.backbone(x)  # feats have shape (B, channels, H, W, D, T) (B, 288, 2, 2, 2, 20)
        feats = rearrange(
            feats, "(b w) c x y z t -> b (w x y z t) c", w=num_windows
        )  # convert to (B, patches, channels)

        return Embeddings(
            cls_embeds=None,
            reg_embeds=None,
            patch_embeds=feats,
        )


class SwiftTransform:
    """
    0. Unnormalize voxelwize z-scored data
    1. global z-score normalization
    2. temporal resampling
    3. pad/crop to expected sequence length t=20
    4. unmask input to full 4D volume
    5. spatial padding to (96, 96, 96)
    6. reshape to expected shape (C, H, W, D, T)
    """

    def __init__(self):
        # Mask calculation from fmri_fm_eval.readers
        roi_path = tflow.get(
            "MNI152NLin6Asym", desc="brain", resolution=2, suffix="mask", extension="nii.gz"
        )
        mask = nisc.read_nifti_data(roi_path) > 0  # (Z, Y, X)

        self.mask = torch.from_numpy(mask)
        self.mask_shape = mask.shape

        # Swift input size is (H, W, D, T) = (X, Y, Z, T) = (96, 96, 96, 20)
        # https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/models/patchembedding.py#L11
        self.expected_seq_len = 20
        self.spatial_target = 96

        # Swift trains on ABCD (TR=0.8), HCP (TR=0.72), and UKB (TR=0.735) without any
        # temporal resampling.
        self.target_tr = 0.8

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Transform bold volumes to model input format.

        sample dicts requires keys:
            - bold: (T,V) normalized bold signal,
            - mean: (1,V) mean of bold signal,
            - std: (1,V) standard deviation of bold signal

        sample dict is modified in place:
            - bold: (C, H, W, D, T)

        """
        # unnormalize
        bold = sample["bold"] * sample["std"] + sample["mean"]

        # global z-score
        # we don't need to worry about background since the data are already masked
        # https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L32-L35
        bold = (bold - bold.mean()) / bold.std()

        # temporal resampling in case the tr is different from pretraining data.
        tr = sample["tr"]
        if abs(tr - self.target_tr) >= 0.1:
            bold = resample_to_target_tr(bold, tr, self.target_tr)

        # Pad if too short - repeat mean (consistent with other models)
        T = len(bold)
        if T < self.expected_seq_len:
            mean = bold.mean(dim=0).repeat(self.expected_seq_len - T, 1)
            bold = torch.cat([bold, mean], dim=0)
            T = self.expected_seq_len

        # Crop to fixed number of non-overlapping windows
        num_windows = T // self.expected_seq_len
        T = num_windows * self.expected_seq_len
        bold = bold[:T, :]

        # unflatten
        # background is filled with min value
        # https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L40
        T, V = bold.shape
        Z, Y, X = self.mask_shape
        mask = self.mask.to(device=bold.device)
        fill_value = bold.min().item()
        volume = torch.full((T, Z, Y, X), fill_value, device=bold.device)
        volume[:, mask] = bold  # Assign flattened voxels to mask positions
        volume = rearrange(volume, "t z y x -> t x y z")

        # center crop or pad
        # fixed padding from original code for mni152 2mm volume
        # (91, 109, 91) -> (96, 96, 96)
        # https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/datasets.py#L267-L270
        assert (X, Y, Z) == (91, 109, 91), "unexpected volume shape"
        volume = F.pad(volume, (3, 2, -7, -6, 3, 2), value=fill_value)

        # rearrange to (C, H, W, D, T)
        # bit confusing but in their notation (H, W, D) is (X, Y, Z)
        # https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/pl_classifier.py#L143-L144
        # https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/datasets.py#L270
        volume = rearrange(volume, "t x y z -> 1 x y z t")

        sample["bold"] = volume
        return sample


def resample_to_target_tr(
    x: Tensor,
    tr: float,
    target_tr: float,
    mode: str = "linear",
) -> Tensor:
    # x: [T, D]
    x = F.interpolate(
        x.T.unsqueeze(0),
        size=round(float(tr) * len(x) / float(target_tr)),
        mode=mode,
    )  # [1, D, T]
    return x.squeeze(0).T


@register_model
def swift() -> tuple[SwiftTransform, SwiftWrapper]:
    return SwiftTransform(), SwiftWrapper(fetch_swift_checkpoint())
