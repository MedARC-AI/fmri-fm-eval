from typing import Callable, Iterable, NamedTuple

import torch.nn as nn
from torch import Tensor


class Embeddings(NamedTuple):
    cls_embeds: Tensor | None
    """cls embeddings [B 1 D]"""

    reg_embeds: Tensor | None
    """register embeddings [B R D]"""

    patch_embeds: Tensor | None
    """patch embeddings [B L D]"""


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

    def fit(self, train_dataset: Iterable[dict[str, Tensor]]) -> None:
        """
        Precompute global transform parameters (e.g. mean, stdev) on training dataset.

        Optional, doesn't have to be defined.
        """


ModelTransformPair = tuple[ModelTransform | None, ModelWrapper]

ModelFn = Callable[..., ModelWrapper | ModelTransformPair]
