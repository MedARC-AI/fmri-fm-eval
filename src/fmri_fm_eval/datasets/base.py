import json
from pathlib import Path
from typing import Any, Callable

import datasets as hfds
import numpy as np
import fsspec
import torch


class Dataset:
    """
    Abstract dataset.
    """

    __num_classes__: int
    """Number of target classes, or target dimension for regression."""


class ArrowDataset(torch.utils.data.Dataset):
    """
    HF Arrow dataset.

    Args:
        url: local or remote path for dataset.

    Notes:
        Supports remote urls such as

        ```
        s3://medarc/fmri-fm-eval/processed/hcpya-rest1lr.schaefer400_tians3.arrow/train
        hf://datasets/clane9/fmri-fm-eval/hcpya-rest1lr.schaefer400_tians3.arrow/train
        ```
    """

    __num_classes__: int

    def __init__(
        self,
        url: str,
        target_map_path: str | Path | None = None,
        target_key: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        keep_in_memory: bool = False,
        storage_options: dict[str, Any] | None = None,
    ):
        self.url = url
        self.target_key = target_key
        self.target_map_path = target_map_path
        self.keep_in_memory = keep_in_memory

        dataset = hfds.load_from_disk(
            url, keep_in_memory=keep_in_memory, storage_options=storage_options
        )
        dataset.set_format("torch")

        if target_map_path is not None:
            with fsspec.open(target_map_path, "r") as f:
                target_map = json.load(f)

            indices = np.array(
                [ii for ii, key in enumerate(dataset[target_key]) if key in target_map]
            )
        else:
            target_map = None
            indices = np.arange(len(dataset))

        self.dataset = dataset
        self.target_map = target_map
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index: int):
        sample = self.dataset[self.indices[index]]

        if self.target_map:
            sample["target"] = self.target_map[sample[self.target_key]]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.indices)

    def set_transform(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.transform = transform
