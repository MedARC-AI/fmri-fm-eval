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
        print("batch.keys(): ", batch.keys())
        print("batch bold shape: ", batch['bold'].shape)
        if x.ndim != 6:
            raise ValueError(
                f"Expected batch['x'] to be 6D (B, C, H, W, D, T), got shape {tuple(x.shape)}"
            )

        feats = self.backbone(x) # feats have shape (B, channels, H, W, D, T) (B, 288, 2, 2, 2, 20)
        feats = rearrange(feats, 'b c x y z t -> b (x y z t) c')

        print("feats shape: ", feats.shape)
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
        roi_path = nisc.fetch_schaefer(400, space="mni")
        # Mask calculation matches fmri_fm_eval.readers.mni_reader
        mask = nisc.read_nifti_data(roi_path) > 0 # (Z, Y, X)
        self.mask_shape = mask.shape
        self.mask = torch.from_numpy(mask)

        self.expected_seq_len = expected_seq_len
        self.scaling_method = scaling_method
        self.fill_zeroback = fill_zeroback
        self.spatial_target = spatial_target
        self.temporal_mode = temporal_mode
        self.temporal_stride = temporal_stride
        self.fill_zeroback = fill_zeroback

    def _temporal_select(self, x: Tensor) -> Tensor:
        """
        x: (B, T, X, Y, Z)
        returns: (B, T', X, Y, Z) where T' == expected_seq_len
        """
        B, T, X, Y, Z = x.shape
        L = self.expected_seq_len

        if T == L:
            return x

        if T > L:
            if self.temporal_mode == "start":
                return x[:, :L]
            elif self.temporal_mode == "center":
                s = (T - L) // 2
                return x[:, s:s+L]
            elif self.temporal_mode == "random":
                s = torch.randint(0, T - L + 1, (1,), device=x.device).item()
                return x[:, s:s+L]
            elif self.temporal_mode == "stride":
                stride = self.temporal_stride
                if stride is None:
                    stride = max(T // L, 1)
                idx = torch.arange(0, stride * L, stride, device=x.device)
                idx = torch.clamp(idx, max=T-1)
                return x.index_select(1, idx)
            else:
                raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")

        # T < L: pad by repeating last frame
        pad_n = L - T
        last = x[:, -1:].expand(B, pad_n, X, Y, Z)
        return torch.cat([x, last], dim=1)

    def _global_scale_and_fill_bg(self, x: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        """
        x: (B,T,X,Y,Z), background defined as x==0
        returns: (x_scaled_filled, fill_val_per_sample)
        """
        # background mask from *original* background (zeros introduced by unflatten)
        mask = x != 0
        xf = x.float()
        mf = mask.float()

        B = x.shape[0]
        count = mf.sum(dim=(1,2,3,4)).clamp_min(1.0)

        # each frame is globally scaled across the 4D volume: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L32C1-L37C107
        if self.scaling_method == "z-norm":
            s1 = (xf * mf).sum(dim=(1,2,3,4))
            s2 = (xf.square() * mf).sum(dim=(1,2,3,4))
            mean = s1 / count
            var = (s2 / count) - mean.square()
            std = torch.sqrt(torch.clamp(var, min=eps))

            mean_ = mean.view(B,1,1,1,1)
            std_  = std.view(B,1,1,1,1)
            x_temp = (xf - mean_) / std_

        elif self.scaling_method == "minmax":
            inf = torch.tensor(float("inf"), device=x.device)
            ninf = torch.tensor(float("-inf"), device=x.device)

            x_for_min = torch.where(mask, xf, inf)
            x_for_max = torch.where(mask, xf, ninf)

            vmin = x_for_min.amin(dim=(1,2,3,4))
            vmax = x_for_max.amax(dim=(1,2,3,4))

            vmin_ = vmin.view(B,1,1,1,1)
            vmax_ = vmax.view(B,1,1,1,1)
            denom = (vmax_ - vmin_).clamp_min(eps)

            x_temp = (xf - vmin_) / denom
        else:
            raise ValueError("scaling_method must be 'z-norm' or 'minmax'")

        # find the min value of the brain after scaling for each sample
        inf = torch.tensor(float("inf"), device=x.device)
        x_for_min = torch.where(mask, x_temp, inf)
        brain_min = x_for_min.amin(dim=(1,2,3,4)) 

        # fill the background with the min value of the brain after scaling or zeros if fill_zeroback is True, see: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/preprocessing.py#L39C5-L40C88
        fill_val = torch.zeros_like(brain_min) if self.fill_zeroback else brain_min
        fill_ = fill_val.view(B,1,1,1,1)

        x_filled = torch.where(mask, x_temp, fill_)
        return x_filled.to(dtype=x.dtype), fill_val.to(dtype=x.dtype)

    def _center_crop_pad_to(self, x: Tensor, fill_val: Tensor) -> Tensor:
        """
        x: (B,T,X,Y,Z) -> (B,T,96,96,96) with center crop/pad.
        fill_val: (B,) used for padding regions (per-sample background fill)
        """
        B, T, X, Y, Z = x.shape
        tgt = self.spatial_target

        def crop_slice(n: int):
            if n <= tgt:
                return slice(0, n)
            s = (n - tgt) // 2
            return slice(s, s + tgt)

        xs = crop_slice(X)
        ys = crop_slice(Y)
        zs = crop_slice(Z)

        xc = x[:, :, xs, ys, zs]
        _, _, Xc, Yc, Zc = xc.shape

        px = tgt - Xc
        py = tgt - Yc
        pz = tgt - Zc

        px_l, px_r = px // 2, px - (px // 2)
        py_l, py_r = py // 2, py - (py // 2)
        pz_l, pz_r = pz // 2, pz - (pz // 2)

        # create output filled with per-sample fill
        out = torch.empty((B, T, tgt, tgt, tgt), device=x.device, dtype=x.dtype)
        out[:] = fill_val.view(B, 1, 1, 1, 1)

        out[:, :, px_l:px_l+Xc, py_l:py_l+Yc, pz_l:pz_l+Zc] = xc
        return out

    def center_crop_pad_to(
        x: Tensor,
        fill_val: Tensor,
        tgt: int = 96,
        prefer_left_extra: bool = True,
    ) -> Tensor:
        """
        Dynamic crop/pad to (tgt,tgt,tgt) for x: (B,T,X,Y,Z).

        If prefer_left_extra=True:
            - odd padding puts the extra voxel on the LEFT  (e.g., 5 -> 3L/2R)
            - odd cropping removes the extra voxel on the LEFT (e.g., 13 -> 7L/6R)
            This is the behavior of the original code, see: https://github.com/Transconnectome/SwiFT/blob/0b07cb156d77a7de33078c8f6fe3ddffa7b5ae9c/project/module/utils/data_preprocess_and_load/datasets.py#L120C1-L121C1
        
        fill_val: (B,) per-sample background fill value used for padded regions.
        """
        assert x.ndim == 5, f"expected (B,T,X,Y,Z), got {x.shape}"
        B, T, X, Y, Z = x.shape
        
        def split_odd(total: int):
            if prefer_left_extra:
                return (total + 1) // 2, total // 2
            return total // 2, (total + 1) // 2
        
        # Crop each dimension
        def get_slice(n: int):
            if n <= tgt:
                return slice(None)
            crop_l, crop_r = split_odd(n - tgt)
            return slice(crop_l, n - crop_r)
        
        xc = x[:, :, get_slice(X), get_slice(Y), get_slice(Z)]
        _, _, Xc, Yc, Zc = xc.shape
        
        pad_z = split_odd(max(0, tgt - Zc))
        pad_y = split_odd(max(0, tgt - Yc))
        pad_x = split_odd(max(0, tgt - Xc))
        
        padding = (*pad_z, *pad_y, *pad_x)
        
        out = xc.clone()
        for b in range(B):
            out[b] = F.pad(xc[b], padding, value=fill_val[b].item())
        
        return out

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample['bold']

        # unnormalize
        bold = bold * sample['std'] + sample['mean']

        if bold.ndim == 2:
            bold = bold.unsqueeze(0)  # (1,T,V) create batch dimension

        B, T, V = bold.shape
        mask = self.mask.to(device=bold.device)
        Z, Y, X = self.mask_shape
        vol = torch.zeros((B, T, Z, Y, X), dtype=bold.dtype, device=bold.device)
        vol[:, :, mask] = bold  # mask flattens (Z,Y,X) -> V positions

        x = vol.permute(0, 1, 4, 3, 2)  # (B,T,X,Y,Z)

        x, fill_val = self._global_scale_and_fill_bg(x)
        x = self._center_crop_pad_to(x, fill_val)

        x = self._temporal_select(x)  # (B,T',96,96,96)

        x = x.permute(0, 2, 3, 4, 1).unsqueeze(1) # model expects a channel dimension (B, C, H, W, D, T)
        sample["bold"] = x[0] # remove batch dimension
        return sample
        
        # # bold is (T, V) or (B, T, V) but here likely (T, V) as it is a transform on sample
        
        # # Reconstruct volume (T, Z, Y, X)
        # # Note: nisc.read_nifti_data outputs (T, Z, Y, X) but we only have T and V.
        # # We place V back into (Z, Y, X).
        
        # T = bold.shape[0]
        
        # # Initialize volume
        # vol = torch.zeros((T, *self.mask_shape), dtype=bold.dtype, device=bold.device)
        
        # # Fill volume
        # # mask is (Z, Y, X). Broadcasting over T works for assignment.
        # vol[:, self.mask] = bold
        
        # # Swift expects (B, C, H, W, D, T)
        # # vol is (T, Z, Y, X)
        # # Permute to (X, Y, Z, T) -> (C, X, Y, Z, T)
        # # Assuming H=X, W=Y, D=Z or similar standard
        
        # vol = vol.permute(3, 2, 1, 0) # (X, Y, Z, T)
        # vol = vol.unsqueeze(0) # (1, X, Y, Z, T) => C=1
        
        # sample['bold'] = vol
        # return sample


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
