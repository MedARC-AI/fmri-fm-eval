from brainharmonix.libs.attn_utils import fa2_utils

# NOTE (will): Monkey patch to not try and import flash attn ever
fa2_utils.is_flash_attn_2_available = lambda: False


# flake8: noqa: E402
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model
import brainharmonix.libs.model as model
import brainharmonix.libs.position_embedding as pos_embeds
import torch
import numpy as np
import math
import urllib

DEVICE = "cuda"
# NOTE (will): Needed for their flash attention stuff
DTYPE = torch.bfloat16

BRAIN_HARMONY_CACHE_DIR = Path.home() / ".cache" / "fmri-fm-eval" / "brain-harmony"


class BrainHarmonixFTransform:
    """
    Hacked together from referencing the fmri dataloaders in brain_datasets/datasets.py
    """

    def __init__(
        self,
        *,
        target_tr: float = 0.735,
        preprocess: bool = None,
        standard_time=48 * 0.735,
        sampling_rate: int = 1,
        # NOTE (will): manually hardcoded to 18 to match mode conf, patch_size * target_num_patches needs to == pos_emb dim in the model. supposedly there's a None codepath that's more flexible, but have only been able to get it to work this way. Downstream effect needs to be checked
        target_num_patches: int | None = 18,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.standard_time = standard_time
        self.preprocess = preprocess
        self.target_tr = target_tr
        self.patch_size = round(self.standard_time / self.target_tr)
        self.target_num_patches = target_num_patches

    def preprocess_fmri(self, ts_array):
        # NOTE (will): Preprocess from fmri_BaseDataset in brain_datasets/dataset.py. Removed dataset-scale norm

        if self.preprocess == "centering":
            ts_array = ts_array - np.mean(ts_array, axis=1)[:, None]
        elif self.preprocess == "zscore":
            ts_array = (ts_array - np.mean(ts_array, axis=1)[:, None]) / (
                np.std(ts_array, axis=1)[:, None] + 1e-9
            )

        # NOTE (will) - this removed because it references the enter dataset
        # # Apply robust scaling
        # if self.norm == "all_robust_scaling":
        #     median, iqr = (
        #         self.normalization_params["medians"],
        #         self.normalization_params["iqrs"],
        #     )
        #     ts_array = (ts_array - median[:, None]) / iqr[:, None]
        # elif self.norm == "all_mean_std":
        #     mean, std = (
        #         self.normalization_params["mean"],
        #         self.normalization_params["std"],
        #     )
        #     ts_array = (ts_array - mean[:, None]) / std[:, None]
        # else:
        #     pass

        return ts_array

    def pad(self, ts_array, original_time_length, target_pad_length):
        padded = torch.zeros((400, target_pad_length), dtype=ts_array.dtype)
        assert original_time_length <= target_pad_length
        padded[:, :original_time_length] = ts_array[:, :original_time_length]

        return padded

    def signal_attn_mask(self, num_patches):
        if self.target_num_patches is None:
            arange = torch.arange(num_patches)
            attn_mask_ = (arange[None, :] < num_patches).int()
            mask2d = attn_mask_.repeat(400, 1)
            return mask2d.view(-1)

        # NOTE (will): this may be fishy
        arange = torch.arange(self.target_num_patches)
        attn_mask_ = (arange[None, :] < num_patches).int()
        mask2d = attn_mask_.repeat(400, 1)
        return mask2d.view(-1)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        # NOTE (will): The library logic uses np here and I don't want to mess with it
        bold = np.array(sample["bold"])
        std = np.array(sample["std"])
        mean = np.array(sample["mean"])

        # Expect bold to be [seq_length, roi]
        seq_length = bold.shape[0]
        num_patches = math.ceil(seq_length // self.sampling_rate / self.patch_size)

        if self.target_num_patches is not None:
            assert self.target_num_patches >= num_patches
            target_pad_length = self.target_num_patches * self.patch_size
        else:
            target_pad_length = num_patches * self.patch_size

        # NOTE (will): Should we do this trimming?
        # bold = bold[: self.seq_length, :]
        # std = std[: self.seq_length, :]
        # mean = mean[: self.seq_length, :]

        series_raw = (std * bold) + mean

        # Flip from [seq, roi] to [roi, seq]
        ts_array = series_raw.T
        ts_array = self.preprocess_fmri(ts_array)

        # NOTE (will): This skips downsampling. Do we need?
        ts_array = torch.from_numpy(ts_array)

        original_time_length = ts_array.shape[1]
        padded = self.pad(ts_array, original_time_length, target_pad_length)
        ts = torch.unsqueeze(padded, 0)

        attn_mask = self.signal_attn_mask(num_patches)

        return {
            "ts": ts,
            "attention_mask": attn_mask,
            "patch_size": self.patch_size,
            **sample,
        }


def get_pos_embed(DEVICE, name, **kwargs):
    return getattr(pos_embeds, name)(DEVICE, kwargs["model_args"])


def get_encoder(pos_embed, cls_token, name, **kwargs):
    return getattr(model, name)(
        pos_embed=pos_embed, cls_token=cls_token, attn_mode="sdpa", **kwargs
    )


class BrainHarmonixFWrapper(nn.Module):
    __space__ = "schaefer400"

    def __init__(self, *args, encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        ts = batch["ts"].to(DEVICE, dtype=DTYPE)
        batch_attention_patch_size = batch["patch_size"][0].item()
        attention_mask = batch["attention_mask"].to(DEVICE, dtype=DTYPE)

        # NOTE (will): it can be set by the framework but we need autocast here or else their
        # patched FA2 flex transformer implementation is going to freak out
        with torch.autocast(dtype=DTYPE, device_type=DEVICE):
            patch_embeds = self.encoder(
                ts, batch_attention_patch_size, attention_mask=attention_mask
            )
            # We don't have cls embeds or reg embeds
            return None, None, patch_embeds


def fetch_brain_harmonix_checkpoint() -> Path:
    """Download harmonix-f/model.pth from Google Drive with caching."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading Brain-JEPA checkpoint. "
            "Install with: pip install gdown"
        )

    # File ID from Brain-Harmony Readme -https://drive.google.com/drive/folders/12MkUAOcegU60YVlK8u8_Owmgk4eQVheB, github.com/hzlab/Brain-Harmony
    file_id = "1M4SHZx4L09d8jvP_-kgEHPqeDoNqWtqB"
    cache_dir = BRAIN_HARMONY_CACHE_DIR
    cached_file = cache_dir / "harmonix-f" / "model.pth"
    cached_file.parent.mkdir(exist_ok=True, parents=True)

    if not cached_file.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(cached_file), quiet=False)

    return cached_file


def download_file(url: str, path: Path) -> None:
    if path.exists():
        return
    urllib.request.urlretrieve(url, path)


@register_model
def brain_harmonix_f(
    **kwargs,
):
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

    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        from brainharmonix.configs.harmonizer.stage0_embed import conf_embed_pretrain

        # Download the CSVs from the OG repo that we need for the pos embed
        pos_emb_dir = BRAIN_HARMONY_CACHE_DIR / "pos_emb"
        pos_emb_dir.mkdir(exist_ok=True, parents=True)

        geo_harm_file = pos_emb_dir / "schaefer400_roi_eigenmodes.csv"
        download_file(
            "https://raw.githubusercontent.com/hzlab/Brain-Harmony/refs/heads/main/brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv",
            geo_harm_file,
        )

        gradient_file = pos_emb_dir / "gradient_mapping_400.csv"
        download_file(
            "https://raw.githubusercontent.com/hzlab/Brain-Harmony/refs/heads/main/brainharmony_pos_embed/gradient_mapping_400.csv",
            gradient_file,
        )

        conf_path = Path(conf_embed_pretrain.__file__)
        # parents: stage0_embed -> harmonizer -> configs -> brainharmonix
        repo_path = conf_path.parent.parent.parent.parent
        pos_emb_dir = repo_path / "brainharmony_pos_embed"

        args = conf_embed_pretrain.Args(
            cls_token=False,
            embed_dim=768,
            geo_harm=str(geo_harm_file),
            gradient=str(gradient_file),
            geoh_dim=200,
            grad_dim=30,
            grid_size=(400, 18),
            predictor_embed_dim=384,
            use_pos_embed_decoder=True,
        )
        fmri_encoder_pos_embed = (
            pos_embeds.BrainGradient_GeometricHarmonics_Anatomical_400_PosEmbed(DEVICE, args)
        )
        print(fmri_encoder_pos_embed)

        encoder_config = {
            "gradient_checkpointing": False,
            "img_size": (400, 864),
            "name": "vit_base_flex",
            "patch_size": 48,
        }
        print("loading fmri encoder")
        fmri_encoder = get_encoder(fmri_encoder_pos_embed, None, **encoder_config).to(DEVICE)
        print(fmri_encoder)

        print("Loading fmri checkpoint")
        ckpt = fetch_brain_harmonix_checkpoint()
        ckpt_pth = torch.load(
            ckpt,
            map_location=DEVICE,
            weights_only=True,
        )

        prefix = "encoder_ema."
        ema_state_dict = {k[len(prefix) :]: v for k, v in ckpt_pth.items() if k.startswith(prefix)}
        encoder_state_dict = ema_state_dict.copy()
        encoder_model_state_dict = fmri_encoder.state_dict()

        # NOTE (will): This deeply scares me but it's copied from the original!
        for key in list(encoder_state_dict.keys()):
            if (
                key in encoder_model_state_dict
                and encoder_state_dict[key].size() != encoder_model_state_dict[key].size()
            ):
                print(
                    f"[encoder] skip param {key} due to size mismatch{encoder_state_dict[key].size()} vs {encoder_model_state_dict[key].size()}"
                )
                del encoder_state_dict[key]

        msg = fmri_encoder.load_state_dict(encoder_state_dict, strict=False)
        print(msg)

    # Freeze model
    fmri_encoder.eval()
    for param in fmri_encoder.parameters():
        param.requires_grad = False

    return BrainHarmonixFTransform(**kwargs), BrainHarmonixFWrapper(encoder=fmri_encoder)
