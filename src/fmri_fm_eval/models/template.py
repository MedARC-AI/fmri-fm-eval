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


class NewModelWrapper(nn.Module):
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

    __space__: str = "NOT_IMPLEMENTED"
    """Expected input data space. E.g. 'schaefer400', 'flat', 'mni'."""

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        raise NotImplementedError


class NewModelTransform:
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

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError


@register_model
def new_model(**kwargs) -> NewModelWrapper | tuple[NewModelTransform, NewModelWrapper]:
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
    raise NotImplementedError
