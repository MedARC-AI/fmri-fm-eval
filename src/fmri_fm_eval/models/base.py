from typing import Callable, NamedTuple

import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class Embeddings(NamedTuple):
    cls_embeds: Float[Tensor, "B 1 D"] | None
    reg_embeds: Float[Tensor, "B R D"] | None
    patch_embeds: Float[Tensor, "B L D"] | None


class ModelWrapper(nn.Module):
    """
    Wrap an fMRI encoder model. Takes an input batch and returns a tuple of embeddings.
    """

    __space__: str
    """Expected input data space."""

    def forward(self, batch: dict[str, Tensor]) -> Embeddings: ...


class ModelTransform:
    """
    Model specific data transform. Takes an input sample and returns a new sample
    with all model-specific transforms applied.
    """

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]: ...


def default_transform(sample: dict[str, Tensor]) -> dict[str, Tensor]:
    """Default No-op transform."""
    return sample


ModelTransformPair = tuple[ModelTransform, ModelWrapper]

ModelFn = Callable[..., ModelWrapper | ModelTransformPair]
