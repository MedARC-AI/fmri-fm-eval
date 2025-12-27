"""

SwiFT: Swin 4D fMRI Transformer

"""


"""Template for a new model.

Instructions:

1. Create an `fmri_fm_eval` package inside *your* repo

    ```
    mkdir -p my_repo/src/fmri_fm_eval/models
    ```

    This will make your model discoverable to the eval suite as a [namespace package
plugin](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages).

2. Copy the `template.py` into the new package

    ```
    cp template.py my_repo/src/fmri_fm_eval/models/my_model.py
    ```

3. Implement the `ModelWrapper` and optionally `ModelTransform` for the new model.

    You can freely import from your official model code. You do not need to
    copy/re-implement the entire model.

4. Run the test to validate the model

    ```
    python -m fmri_fm_eval.models.test_models my_model
    ```

    If you want to debug your implementation, you can copy the provided `test_models.py`
    into your source tree and run locally.

5. (Optional) open a PR to add your model to the upstream repo

    Your PR should only include the single model wrapper file

    ```
    fmri-fm-eval/src/fmri_fm_eval/models/my_model.py
    ```

    Any extra dependencies needed should be added as optional dependencies for
    your specific model in the `pyproject.toml`
    (https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-and-requirements).
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


try:
    from swiftfmri.pl_classifier import LitClassifier
except ImportError as exc:
    raise ImportError(
        "swiftfmri not installed. Please install the optional swiftfmri extra."
    ) from exc


# Cache directory for downloaded files
SWIFT_CACHE_DIR = Path.home() / ".cache" / "fmri-fm-eval" / "swift"



def fetch_swift_checkpoint() -> Path:
    """Download contrastive_pretrained.ckpt from Google Drive with caching."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading SwiFT checkpoint. "
            "Install with: pip install gdown"
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
    """
    Wrap an fMRI encoder model. Takes an input batch and returns a tuple of embeddings.

    The wrapper should handle:

    - initializing the model as a child submodule
    - applying the forward pass correctly
    - reformatting the model's embeddings as a tuple of:
        - cls_embeds [B, 1, D]
        - reg_embeds [B, R, D]
        - patch_embeds [B, L, D]

    If the model doesn't use one or more of these embeddings, they can be set to None.

    The wrapper should assume that the data have been preprocessed into the model's
    required format. It's the job of the transform (below) to take care of this step.
    Otherwise, the data are in the default sample format.
    """

    __space__: str = "mni"
    """Expected input data space. E.g. 'schaefer400', 'flat', 'mni'."""

    def __init__(
        self,
        ckpt_path: Path,
        *,
        label_scaling_method: str = "standardization", 
    ) -> None:
        super().__init__()

        self.ckpt_path = ckpt_path
        self.label_scaling_method = label_scaling_method

        dm = _DummyDataModule()
        
        lit = LitClassifier.load_from_checkpoint(
            str(ckpt_path),
            data_module=dm,
            map_location='cpu',
            label_scaling_method=label_scaling_method,
            strict=False, # emb_mlp and model.head are not mapped because these old checkpoint keys are not present in the new codebase.
        )

        self.backbone = lit.model
        self._lit = lit

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch['bold']
        if x.ndim != 6:
            raise ValueError(
                f"Expected batch['x'] to be 6D (B, C, H, W, D, T), got shape {tuple(x.shape)}"
            )

        feats = self.backbone(x) # feats have shape (B, channels, H, W, D, T) (B, 288, 2, 2, 2, 20)
        feats = rearrange(feats, 'b c x y z t -> b (x y z t) c') # convert to (B, patches, channels)

        return Embeddings(
            cls_embeds=None,
            reg_embeds=None,
            patch_embeds=feats,
        )


class SwiftTransform:
    """
    Model specific data transform. Takes an input sample and returns a new sample with
    all model-specific transforms applied.

    Input samples have the following fields:

    - bold: bold time series, shape `(n_frames, dim)` where `dim` is the dimension of
        the input space (see `fmri_fm_eval.readers`). the time series has been
        normalized to mean zero unit stdev across time for each dimension.
    - mean: bold time series mean, shape `(1, dim)`.
    - std: bold time series stdev, shape `(1, dim)`.
    - tr: float repetition time.

    The transform should handle:

    - reconstructing un-normalized data if necessary (`bold = bold * std + mean`)
    - temporal resampling, if any
    - temporal trimming/padding to model expected sequence length
    - additional normalization if any
    - renaming keys to those expected by the model wrapper

    The transform can assume the input data are in the appropriate space for the model.
    See `fmri_fm_eval.readers` for a list of available spaces.
    """

    def __init__(
        self,
        expected_seq_len: int = 20,
        scaling_method: str = "z-norm", # from https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L56
        fill_zeroback: bool = False, # this is false by default in the original code so it's filling the background with the minimum value, see: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L7C92-L7C113
        spatial_target: int = 96,
        temporal_mode: str = "start", # "start" | "center" | "random" | "stride" 
        temporal_stride: int | None = None, # used if temporal_mode="stride"
    ):

        # Mask calculation from fmri_fm_eval.readers
        roi_path = nisc.fetch_schaefer(400, space="mni")
        mask = nisc.read_nifti_data(roi_path) > 0 # (Z, Y, X)

        self.mask = torch.from_numpy(mask)
        self.mask_shape = mask.shape

        self.expected_seq_len = expected_seq_len
        self.scaling_method = scaling_method
        self.fill_zeroback = fill_zeroback
        self.spatial_target = spatial_target
        self.temporal_mode = temporal_mode
        self.temporal_stride = temporal_stride
        self.fill_zeroback = fill_zeroback

    def _temporal_select(self, x: Tensor) -> Tensor:
        """
        Selects/pads the temporal dimension of the input volume to the expected sequence length.
        
        Args:
            x: (T, X, Y, Z)
        Returns:
            (expected_seq_len, X, Y, Z)
        """
        T = x.shape[0]
        target_len = self.expected_seq_len

        if T == target_len:
            return x
        
        # Pad if too short - repeat last frame
        if T < target_len:
            last_frame = repeat(x[-1], 'x y z -> t x y z', t=target_len - T)
            return torch.cat([x, last_frame], dim=0)
        
        # Crop if too long
        if self.temporal_mode == "start":
            return x[:target_len]
        
        elif self.temporal_mode == "center":
            start = (T - target_len) // 2
            return x[start:start + target_len]
        
        elif self.temporal_mode == "random":
            start = torch.randint(0, T - target_len + 1, (1,)).item()
            return x[start:start + target_len]
        
        elif self.temporal_mode == "stride":
            stride = self.temporal_stride or max(T // target_len, 1)
            indices = torch.arange(0, stride * target_len, stride)
            indices = torch.clamp(indices, max=T - 1)
            return x[indices]
        
        else:
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")

    def _scale_frame(self, frame: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Scales globally a single 3D frame, see https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L32C1-L37C107
        
        Args:
            frame: (X, Y, Z) single timepoint
            mask: (X, Y, Z) brain mask
            
        Returns:
            (X, Y, Z) scaled frame
        """
        frame_f = frame.float()
        mask_f = mask.float()
        
        # Count brain voxels
        brain_voxels = frame_f[mask]
        n_voxels = brain_voxels.numel()
        
        if n_voxels == 0:
            return frame  # if no brain voxels, return as-is, can this happen??
        
        if self.scaling_method == "z-norm":
            mean = brain_voxels.mean()
            std = brain_voxels.std(unbiased=False).clamp_min(eps)
            return (frame_f - mean) / std
        
        elif self.scaling_method == "minmax":
            vmin = brain_voxels.min()
            vmax = brain_voxels.max()
            denom = (vmax - vmin).clamp_min(eps)
            return (frame_f - vmin) / denom
        
        else:
            raise ValueError("scaling_method must be 'z-norm' or 'minmax'")


    def _scale_and_fill_volume(self, x: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        """
        Scales each frame independently and fill background,
        and fills the background with the min value of the current frame after scaling or zeros if fill_zeroback is True,
        by default we're using the min value as original code, see: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L39C5-L40C88
        
        Args:
            x: (T, X, Y, Z) volume
            
        Returns:
            scaled_filled: (T, X, Y, Z) processed volume
            fill_values: (T,) fill value used per frame, required for padding
        """
        T, X, Y, Z = x.shape
        
        brain_mask = x != 0  # (T, X, Y, Z)
        
        scaled = torch.empty_like(x)
        fill_values = torch.empty(T, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            frame = x[t]  # (X, Y, Z)
            mask_t = brain_mask[t]  # (X, Y, Z)
            
            # Scale frame
            scaled[t] = self._scale_frame(frame, mask_t, eps)
            
            if self.fill_zeroback:
                # just use zeros
                fill_values[t] = 0.0
            else:
                # get minimum
                brain_voxels = scaled[t][mask_t]
                fill_values[t] = brain_voxels.min() if brain_voxels.numel() > 0 else 0.0
            
            # fill background with corresponding value
            scaled[t] = torch.where(mask_t, scaled[t], fill_values[t])
        
        return scaled, fill_values

    def _scale_brain_voxels(self, x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Apply global scaling to brain voxels, for each volume independently.
        
        Args:
            x: (T, X, Y, Z) input volume
            mask: (T, X, Y, Z) brain mask, true for voxel.
        
        Returns:
            (T, X, Y, Z) scaled volume
        """
        B, T = x.shape[:2]
        xf = x.float()
        mf = mask.float()
        
        count = rearrange(mf, 'b t x y z -> b t (x y z)').sum(dim=2)
        count = count.clamp_min(1.0)
        
        # each frame is globally scaled: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L32C1-L37C107
        if self.scaling_method == "z-norm":
            brain_sum = rearrange(xf * mf, 'b t x y z -> b t (x y z)').sum(dim=2)
            brain_sq_sum = rearrange(xf.square() * mf, 'b t x y z -> b t (x y z)').sum(dim=2)
            
            mean = brain_sum / count  # (B, T)
            var = (brain_sq_sum / count) - mean.square()
            std = torch.sqrt(torch.clamp(var, min=eps))  # (B, T)
            
            mean = rearrange(mean, 'b t -> b t 1 1 1')
            std = rearrange(std, 'b t -> b t 1 1 1')
            return (xf - mean) / std
        
        elif self.scaling_method == "minmax":
            inf = torch.tensor(float("inf"), device=x.device)
            ninf = torch.tensor(float("-inf"), device=x.device)
            
            x_for_min = torch.where(mask, xf, inf)
            x_for_max = torch.where(mask, xf, ninf)
            
            vmin = rearrange(x_for_min, 'b t x y z -> b t (x y z)').amin(dim=2)  # (B, T)
            vmax = rearrange(x_for_max, 'b t x y z -> b t (x y z)').amax(dim=2)  # (B, T)
            
            vmin = rearrange(vmin, 'b t -> b t 1 1 1')
            vmax = rearrange(vmax, 'b t -> b t 1 1 1')
            denom = (vmax - vmin).clamp_min(eps)
            
            return (xf - vmin) / denom
        
        else:
            raise ValueError("scaling_method must be 'z-norm' or 'minmax'")

    def _center_crop_or_pad(self, x: Tensor, fill_val: Tensor) -> Tensor:
        """
        Center crop or pad volume to target spatial size,
        odd padding goes to left, same as original implementation, see: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/datasets.py#L120C1-L121C1
        
        Args:
            x: (T, X, Y, Z)
            fill_val: (T,) per-frame background fill value
            
        Returns:
            (T, target, target, target)
        """
        T, X, Y, Z = x.shape
        tgt = self.spatial_target
        
        def get_crop_slice(n: int) -> slice:
            if n <= tgt:
                return slice(None)
            start = (n - tgt) // 2
            return slice(start, start + tgt)
        
        x_cropped = x[:, get_crop_slice(X), get_crop_slice(Y), get_crop_slice(Z)]
        _, Xc, Yc, Zc = x_cropped.shape
        
        pad_x = max(0, tgt - Xc)
        pad_y = max(0, tgt - Yc)
        pad_z = max(0, tgt - Zc)
        
        pad_x_left, pad_x_right = (pad_x + 1) // 2, pad_x // 2
        pad_y_left, pad_y_right = (pad_y + 1) // 2, pad_y // 2
        pad_z_left, pad_z_right = (pad_z + 1) // 2, pad_z // 2
        
        out = torch.empty((T, tgt, tgt, tgt), device=x.device, dtype=x.dtype)
        padding = (pad_z_left, pad_z_right, pad_y_left, pad_y_right, pad_x_left, pad_x_right)
        
        for t in range(T):
            out[t] = F.pad(x_cropped[t], padding, value=fill_val[t].item())
        
        return out

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
        bold = sample['bold'] * sample['std'] + sample['mean']

        # unflatten
        T, V = bold.shape
        Z, Y, X = self.mask_shape
        mask = self.mask.to(device=bold.device)
        volume = torch.zeros((T, Z, Y, X), dtype=bold.dtype, device=bold.device)
        volume[:, mask] = bold  # Assign flattened voxels to mask positions
        volume = rearrange(volume, 't z y x -> t x y z')

        # select frames
        volume = self._temporal_select(volume) # (T', X, Y, Z)

        # scale and fill
        volume, fill_values = self._scale_and_fill_volume(volume)  # (T', X, Y, Z)

        # center crop or pad
        volume = self._center_crop_or_pad(volume, fill_values)

        # rearrange to (C, H, W, D, T)
        volume = rearrange(volume, 't x y z -> 1 x y z t')

        sample['bold'] = volume
        return sample


@register_model
def swift(**kwargs) -> SwiftWrapper | tuple[SwiftTransform, SwiftWrapper]:
    """Model constructor.

    This function should return a fully initialized model and optional transform. It
    should handle:

    - downloading and caching necessary supplementary data files, e.g. static position
      embeddings, normalization stats. Cf `nisc.download_file`.
    - downloading and caching pretrained checkpoint weights. alternatively, if
      checkpoint weights are not available for programmatic download, they can be passed
      as a keyword argument `ckpt_path`.
    - defining fixed config settings
    - initializing transform. alternatively, if no special transform is needed the
      transform can be None or dropped altogether.
    - initializing model
    - loading model checkpoint weights
    - freezing weights
    """
    return SwiftTransform(), SwiftWrapper(fetch_swift_checkpoint())
