import json
from pathlib import Path
from typing import Any, Callable

import datasets as hfds
import fsspec
import numpy as np
import pandas as pd
import torch


class HFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: hfds.Dataset,
        target_key: str | None = "target",
        target_map_path: str | Path | None = None,
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

            keys = self.dataset[target_key]
            indices = np.array(
                [
                    ii
                    for ii, key in enumerate(keys)
                    if key in target_map and not pd.isna(target_map[key])
                ]
            )
            targets = np.array([target_map[keys[idx]] for idx in indices])
        else:
            targets = self.dataset[target_key]
            indices = np.array([ii for ii, target in enumerate(targets) if not pd.isna(target)])
            targets = np.asarray(targets[indices])

        labels, target_ids, label_counts = np.unique(
            targets, return_inverse=True, return_counts=True
        )

        self.indices = indices
        self.labels = labels
        self.label_counts = label_counts
        self.targets = targets
        self.target_ids = target_ids
        self.num_classes = len(labels)

    def __getitem__(self, index: int):
        sample = self.dataset[self.indices[index]]
        sample["target"] = self.target_ids[index]
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
            f"    labels={self.labels},\n"
            f"    counts={self.label_counts}"
        )
        s = f"HFDataset(\n{s}\n)"
        return s
