import json
import logging
from pathlib import Path

import datasets as hfds
import nibabel as nib
import numpy as np

import fmri_fm_eval.utils as ut

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)
_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
HCP_ROOT = ROOT / "data/sourcedata/HCP_1200"

# ~500 subs total, 6:2:2 ratio
SUB_BATCH_SPLITS = {
    "train": [0, 1, 2, 3, 4, 5],
    "validation": [16, 17],
    "test": [18, 19],
}

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

# Number of total fslr vertices across cortex
NUM_VERTICES = 64984

# 500 TRs = 6 mins
MAX_NUM_TRS = 500

NUM_PROC = 16


def main():
    outdir = ROOT / "data/processed/hcpya-rest1lr.fslr64k.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return

    # construct train/val/test splits by combining subject batches.
    # nb, across batches subjects are unrelated. we use the batches to dial how much
    # data to include.
    with (ROOT / "splits/hcpya_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    path_splits = {}
    for split, batch_ids in SUB_BATCH_SPLITS.items():
        paths = [
            f"{sub}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii"
            for batch_id in batch_ids
            for sub in sub_batch_splits[f"batch-{batch_id:02d}"]
        ]
        path_splits[split] = paths = [p for p in paths if (HCP_ROOT / p).exists()]
        _logger.info("Num subjects (%s): %d", split, len(paths))

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "mag": hfds.Value("string"),
            "dir": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, NUM_VERTICES), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, NUM_VERTICES), dtype="float16"),
            "std": hfds.Array2D(shape=(1, NUM_VERTICES), dtype="float16"),
        }
    )

    dataset_dict = {}
    for split, paths in path_splits.items():
        dataset_dict[split] = hfds.Dataset.from_generator(
            generate_samples,
            features=features,
            gen_kwargs={"paths": paths},
            num_proc=NUM_PROC,
            split=hfds.NamedSplit(split),
        )
    dataset = hfds.DatasetDict(dataset_dict)

    outdir.parent.mkdir(exist_ok=True, parents=True)
    dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(paths: list[str]):
    for path in paths:
        fullpath = HCP_ROOT / path

        meta = parse_hcp_metadata(fullpath)
        tr = HCP_TR[meta["mag"]]

        img = nib.load(fullpath)
        series = ut.get_cifti_surf_data(img)
        series = np.ascontiguousarray(series.T)

        T, D = series.shape
        assert D == NUM_VERTICES
        if T < MAX_NUM_TRS:
            _logger.warning(f"Path {path} does not have enough data ({T}<{MAX_NUM_TRS}); skipping.")
            continue

        start, end = 0, MAX_NUM_TRS
        series = series[start:end]
        series, mean, std = ut.scale(series)
        series = series.astype(np.float16)

        sample = {
            **meta,
            "path": str(path),
            "start": start,
            "end": end,
            "tr": tr,
            "bold": series,
            "mean": mean[None, :],
            "std": std[None, :],
        }
        yield sample


def parse_hcp_metadata(path: Path) -> dict[str, str]:
    sub = path.parents[3].name
    acq = path.parent.name
    if "7T" in acq:
        mod, task, mag, dir = acq.split("_")
    else:
        mod, task, dir = acq.split("_")
        mag = "3T"
    metadata = {"sub": sub, "mod": mod, "task": task, "mag": mag, "dir": dir}
    return metadata


if __name__ == "__main__":
    main()
