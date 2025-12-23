import torch.nn as nn
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model


class IdentityBackbone(nn.Module):
    __space__: str | None = None

    def extra_repr(self):
        return f"'{self.__space__}'"

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        # Processed datasets stored via make_hcpya_rest1lr_dataset.py expose the
        # region-aggregated time series under the "bold" key.
        roi_time_series = batch["bold"]
        # Expose the raw temporal sequence as register tokens so downstream probes
        # can operate over (B, T, D) features without any learned backbone.
        return None, roi_time_series, None


@register_model
def identity_schaefer400(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "schaefer400"
    return model


@register_model
def identity_schaefer400_tians3(**kwargs) -> IdentityBackbone:
    model = IdentityBackbone()
    model.__space__ = "schaefer400_tians3"
    return model
