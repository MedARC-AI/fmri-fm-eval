import os

import datasets as hfds

from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.datasets.registry import register_dataset

# TODO: package specific cache dir?

HCPYA_ROOT = os.getenv("HCPYA_ROOT", "s3://medarc/fmri-fm-eval/processed")

HCPYA_TARGET_MAP_DICT = {
    "age": "hcpya_target_map_Age.json",
    "gender": "hcpya_target_map_Gender.json",
    "flanker": "hcpya_target_map_Flanker_Unadj.json",
    "neofacn": "hcpya_target_map_NEOFAC_N.json",
    "pmat24": "hcpya_target_map_PMAT24_A_CR.json",
}


def _create_hcpya_rest1lr(space: str, target: str, **kwargs):
    target_key = "sub"
    target_map_path = HCPYA_TARGET_MAP_DICT[target]
    target_map_path = f"{HCPYA_ROOT}/targets/{target_map_path}"

    dataset_dict = {}
    splits = ["train", "validation", "test"]
    for split in splits:
        url = f"{HCPYA_ROOT}/hcpya-rest1lr.{space}.arrow/{split}"
        dataset = hfds.load_dataset("arrow", data_files=f"{url}/*.arrow", split="train", **kwargs)
        dataset = HFDataset(dataset, target_key=target_key, target_map_path=target_map_path)
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
