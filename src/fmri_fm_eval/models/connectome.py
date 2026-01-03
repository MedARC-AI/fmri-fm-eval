import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model


class Connectome(nn.Module):
    __space__: str | None = None

    def extra_repr(self):
        return f"'{self.__space__}'"

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        B, T, R = x.shape

        # normalize to mean zero, unit norm
        x = x - x.mean(dim=1, keepdim=True)
        x = F.normalize(x, dim=1, eps=1e-6)

        # R x R pearson connectome
        conn = x.transpose(1, 2) @ x  # [B, R, R]

        # flatten upper triangle
        row, col = torch.unbind(torch.triu_indices(R, R, offset=1, device=x.device))
        conn = conn[:, row, col]  # [B, R*(R-1)/2]

        cls_embeds = conn[:, None, :]  # [B, 1, D]
        return cls_embeds, None, None


@register_model
def connectome_schaefer400(**kwargs) -> Connectome:
    model = Connectome()
    model.__space__ = "schaefer400"
    return model


@register_model
def connectome_schaefer400_tians3(**kwargs) -> Connectome:
    model = Connectome()
    model.__space__ = "schaefer400_tians3"
    return model
