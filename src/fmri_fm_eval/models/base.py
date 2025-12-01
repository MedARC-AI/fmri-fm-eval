from typing import NamedTuple

import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class Embeddings(NamedTuple):
    cls_embeds: Float[Tensor, "B 1 D"] | None
    reg_embeds: Float[Tensor, "B R D"] | None
    tok_embeds: Float[Tensor, "B L D"] | None


class ModelWrapper(nn.Module):
    """
    Wrap an fMRI encoder model. Takes an input batch and returns a tuple of embeddings.
    """

    def forward(self, batch: dict[str, Tensor]) -> Embeddings: ...


class ModelTransform(nn.Module):
    """
    Model specific data transform. Takes an input sample and returns a new sample
    with all model-specific transforms applied.
    """

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]: ...
