"""
Brain JEPA model interface plugin for fmri-fm-eval.

This file should be placed inside the official Brain JEPA repo at

```
src/fmri_fm_eval/models/brain_jepa.py
```

It should freely import from all Brain JEPA code, just as the official Brain JEPA code
does.
"""

from typing import NamedTuple

import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class Embeddings(NamedTuple):
    cls_embeds: Float[Tensor, "B 1 D"] | None
    reg_embeds: Float[Tensor, "B R D"] | None
    tok_embeds: Float[Tensor, "B L D"] | None


class BrainJEPAWrapper(nn.Module):
    """
    Wrap Brain JEPA encoder model. Takes an input batch and returns a tuple of embeddings.

    It should handle:

    - initializing the brain jepa model as a child submodule
    - applying the forward pass correctly
    - reformatting the output embeddings into the expected Embeddings shape. If one or
      more of the expected embeddings are missing they can be left as None.

    It can assume that the data have been preprocessed into the expected format by the
    data transform below.
    """

    def forward(self, batch: dict[str, Tensor]) -> Embeddings: ...


class BrainJEPATransform(nn.Module):
    """
    Brain JEPA specific data transform. Takes an input sample and returns a new
    sample with all model-specific transforms applied.

    It should handle:

    - unpacking data from the common huggingface format
        (https://huggingface.co/datasets/clane9/fmri-fm-eval)
    - reconstructing data (`bold = bold * std + mean`)
    - temporal resampling, if any
    - temporal trimming/padding to model expected sequence length
    - normalization if any
    - renaming keys to those expected by the model wrapper

    It can assume the data are in the appropriate input space, eg one of
    'schaefer400', 'schaefer400_tians3', 'fslr64k', ...
    """

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]: ...


def brain_jepa_vitb_ep300(**kwargs) -> BrainJEPAWrapper:
    """
    This function should return a fully initialized Brain JEPA wrapper model for the
    pretrained vitb_ep300 model. It should handle:

    - downloading and caching necessary supplementary data files, e.g. gradient position
      embeddings. Cf `nisc.download_file`.
    - downloading and caching pretrained checkpoint weights
    - defining fixed config variables
    - reading runtime defined config variables (if any) from kwargs
    - initializing model
    - loading checkpoint weights
    """
    ...


def brain_jepa_transform(**kwargs) -> BrainJEPATransform:
    """
    This function should return a fully initialized Brain JEPA data transform. It should
    handle:

    - downloading any supplementary data files, e.g. normalization stats, if any
    - defining fixed transform parameters, eg sequence length, normalization
    """
    ...
