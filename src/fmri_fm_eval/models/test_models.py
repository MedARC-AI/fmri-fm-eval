import argparse

import pytest
import torch
from torch import Tensor
from torch.utils.data import default_collate

from fmri_fm_eval.models.registry import list_models, create_model
import fmri_fm_eval.readers as readers


def get_dummy_sample(space: str, n_samples: int) -> dict[str, Tensor]:
    dim = readers.DATA_DIMS[space]
    sample = {
        "tr": 1.0,
        "bold": torch.randn(n_samples, dim),
        "mean": torch.randn(1, dim),
        "std": torch.randn(1, dim),
    }
    return sample


@pytest.mark.parametrize("n_samples", [16, 160])
@pytest.mark.parametrize("name", list_models())
def test_model(name: str, n_samples: int):
    transform, model = create_model(name)

    batch_size = 2
    batch = []
    for _ in range(batch_size):
        sample = get_dummy_sample(model.__space__, n_samples)
        if transform is not None:
            sample = transform(sample)
        batch.append(sample)
    batch = default_collate(batch)

    cls_embeds, reg_embeds, patch_embeds = model(batch)
    if cls_embeds is not None:
        assert cls_embeds.ndim == 3
        assert cls_embeds.shape[:2] == (batch_size, 1)

    if reg_embeds is not None:
        assert reg_embeds.ndim == 3
        assert reg_embeds.shape[0] == batch_size

    if patch_embeds is not None:
        assert patch_embeds.ndim == 3
        assert patch_embeds.shape[0] == batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        nargs="*",
        help=f"model name or pattern (models: {', '.join(list_models())})",
    )
    args = parser.parse_args()

    if args.model:
        filter_args = ["-k", " or ".join(args.model)]
    else:
        filter_args = []
    # ignore annoying error due to repeated import of pytest plugins
    opts = ["-W", "ignore::pytest.PytestAssertRewriteWarning"]
    pytest.main(filter_args + opts + [__file__])
