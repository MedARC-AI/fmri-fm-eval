import torch.nn as nn
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model


class IdentityBackbone(nn.Module):
    __space__: str | None = None

    def extra_repr(self):
        return f"'{self.__space__}'"

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        # get ROI time series, shape [B, T, D]
        roi_time_series = batch["bold"]
        # return as patch embeddings
        return None, None, roi_time_series


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
