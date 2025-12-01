import pytest

import torch
from torch import Tensor

from fmri_fm_eval.models.brain_jepa import brain_jepa_vitb_ep300, brain_jepa_transform


@pytest.fixture
def dummy_sample() -> dict[str, Tensor]:
    sample = {
        "bold": torch.randn(500, 450),  # [num_trs, num_rois]
        "mean": torch.randn(1, 450),
        "std": torch.randn(1, 450),
    }
    return sample


@pytest.fixture
def dummy_batch() -> dict[str, Tensor]:
    batch = {
        "bold": torch.randn(2, 450, 160)
    }  # TODO: is this the correct expected shape for brain jepa?
    return batch


def test_brain_jepa_vitb_ep300(dummy_batch: dict[str, Tensor]):
    model = brain_jepa_vitb_ep300()
    cls_embeds, reg_embeds, tok_embeds = model(dummy_batch)
    assert cls_embeds is None  # brain jepa has no cls (right ?)
    assert reg_embeds is None  # brain jepa has no reg tokens
    assert tok_embeds is not None
    assert tok_embeds.ndim == 3 and tok_embeds.shape[0] == 2


def test_brain_jepa_transform(dummy_sample: dict[str, Tensor]):
    transform = brain_jepa_transform()
    sample = transform(dummy_sample)
    x = sample["bold"]
    assert x.shape == (450, 160)  # TODO: is this the correct expected shape for brain jepa?
