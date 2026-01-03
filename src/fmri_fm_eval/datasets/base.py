import json
from pathlib import Path
from typing import Any, Callable, Literal

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

    __task__: Literal["classification", "regression"]
    """Type of prediction task."""


class HFDataset(torch.utils.data.Dataset):
    __num_classes__: int
    __task__: str

    def __init__(
        self,
        dataset: hfds.Dataset,
        target_map_path: str | Path | None = None,
        target_key: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.dataset = dataset
        self.target_map_path = target_map_path
        self.target_key = target_key
        self.transform = transform

        self.dataset.set_format("torch")

        if target_map_path is not None:
            with fsspec.open(target_map_path, "r") as f:
                target_map = json.load(f)

            indices = np.array(
                [ii for ii, key in enumerate(dataset[target_key]) if key in target_map]
            )

            # getting labels first before pruning the target map in case there are 0
            # examples for some classes in some splits (though that shouldn't happen).
            labels = np.unique(list(target_map.values())).tolist()
            label_to_id = {label: ii for ii, label in enumerate(labels)}

            keys = set(dataset[target_key])
            target_map = {k: v for k, v in target_map.items() if k in keys}

            label_counts = {label: 0 for label in labels}
            for v in target_map.values():
                label_counts[v] += 1
        else:
            target_map = labels = label_to_id = label_counts = None
            indices = np.arange(len(dataset))

        self.target_map = target_map
        self.indices = indices
        self.labels = labels
        self.label_to_id = label_to_id
        self.label_counts = label_counts

    def __getitem__(self, index: int):
        sample = self.dataset[self.indices[index]]

        if self.target_map:
            sample["target"] = self.label_to_id[self.target_map[sample[self.target_key]]]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def set_transform(self, transform: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        s = (
            f"    dataset={self.dataset},\n"
            f"    target_map_path='{self.target_map_path}',\n"
            f"    target_key='{self.target_key}',\n"
            f"    counts={self.label_counts}"
        )
        s = f"HFDataset(\n{s}\n)"
        return s
