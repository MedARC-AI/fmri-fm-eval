import os

import datasets as hfds
import numpy as np

from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.datasets.registry import register_dataset
import fmri_fm_eval.nisc as nisc

# TODO: package specific cache dir?

HCPYA_ROOT = os.getenv("HCPYA_ROOT", "s3://medarc/fmri-fm-eval/processed")

HCPYA_TARGET_MAP_DICT = {
    "age": "hcpya_target_map_Age.json",
    "gender": "hcpya_target_map_Gender.json",
    "flanker": "hcpya_target_map_Flanker_Unadj.json",
    "neofacn": "hcpya_target_map_NEOFAC_N.json",
    "pmat24": "hcpya_target_map_PMAT24_A_CR.json",
}

HCPYA_TARGET_NUM_CLASSES = {
    "age": 3,
    "gender": 2,
    "flanker": 3,
    "neofacn": 3,
    "pmat24": 3,
}


def _resample_to_1s_tr(sample):
    """Resample timeseries from original TR to TR=1.0s"""
    bold = np.array(sample['bold'])
    if abs(sample['tr'] - 1.0) > 0.01:
        bold = nisc.resample_timeseries(
            bold,
            tr=sample['tr'],
            new_tr=1.0,
            kind='linear'
        )
        sample['bold'] = bold.astype(np.float16)
        sample['tr'] = 1.0
        sample['end'] = len(bold)
    return sample


def _create_hcpya_rest1lr(space: str, target: str, **kwargs):
    target_key = "sub"
    target_map_path = HCPYA_TARGET_MAP_DICT[target]
    target_map_path = f"{HCPYA_ROOT}/targets/{target_map_path}"

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{HCPYA_ROOT}/hcpya-rest1lr.{space}.arrow/{split}"
        dataset = hfds.load_dataset("arrow", data_files=f"{url}/*.arrow", split="train", **kwargs)
        dataset = dataset.map(_resample_to_1s_tr, num_proc=8, desc=f"Resampling {split}")
        dataset = HFDataset(
            dataset,
            target_map_path=target_map_path,
            target_key=target_key,
        )
        dataset.__num_classes__ = HCPYA_TARGET_NUM_CLASSES[target]
        dataset.__task__ = "classification"

        dataset_dict[split] = dataset

    return dataset_dict


@register_dataset
def hcpya_rest1lr_age(space: str, **kwargs):
    return _create_hcpya_rest1lr(space, target="age", **kwargs)


@register_dataset
def hcpya_rest1lr_gender(space: str, **kwargs):
    return _create_hcpya_rest1lr(space, target="gender", **kwargs)


@register_dataset
def hcpya_rest1lr_flanker(space: str, **kwargs):
    return _create_hcpya_rest1lr(space, target="flanker", **kwargs)


@register_dataset
def hcpya_rest1lr_neofacn(space: str, **kwargs):
    return _create_hcpya_rest1lr(space, target="neofacn", **kwargs)


@register_dataset
def hcpya_rest1lr_pmat24(space: str, **kwargs):
    return _create_hcpya_rest1lr(space, target="pmat24", **kwargs)
