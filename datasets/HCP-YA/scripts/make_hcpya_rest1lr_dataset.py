import argparse
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets as hfds
import numpy as np
from cloudpathlib import AnyPath, CloudPath

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

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

# 500 TRs = 6 mins
MAX_NUM_TRS = 500


def main(args):
    outdir = ROOT / f"data/processed/hcpya-rest1lr.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # construct train/val/test splits by combining subject batches.
    # nb, across batches subjects are unrelated. we use the batches to dial how much
    # data to include.
    with (ROOT / "splits/hcpya_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    if args.space.startswith("mni"):
        suffix = "rfMRI_REST1_LR.nii.gz"
    else:
        suffix = "rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii"

    root = AnyPath(args.root or HCP_ROOT)
    path_splits = {}
    for split, batch_ids in SUB_BATCH_SPLITS.items():
        paths = [
            f"{sub}/MNINonLinear/Results/rfMRI_REST1_LR/{suffix}"
            for batch_id in batch_ids
            for sub in sub_batch_splits[f"batch-{batch_id:02d}"]
        ]
        path_splits[split] = paths = [p for p in paths if (root / p).exists()]
        _logger.info("Num subjects (%s): %d", split, len(paths))

    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

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
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    dataset_dict = {}
    for split, paths in path_splits.items():
        dataset_dict[split] = hfds.Dataset.from_generator(
            generate_samples,
            features=features,
            gen_kwargs={"paths": paths, "root": root, "reader": reader, "dim": dim},
            num_proc=args.num_proc,
            split=hfds.NamedSplit(split),
        )
    dataset = hfds.DatasetDict(dataset_dict)

    outdir.parent.mkdir(exist_ok=True, parents=True)
    dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(paths: list[str], *, root: AnyPath, reader: readers.Reader, dim: int):
    for path, fullpath in prefetch(root, paths):
        meta = parse_hcp_metadata(fullpath)
        tr = HCP_TR[meta["mag"]]

        series = reader(fullpath)

        T, D = series.shape
        assert D == dim
        if T < MAX_NUM_TRS:
            _logger.warning(f"Path {path} does not have enough data ({T}<{MAX_NUM_TRS}); skipping.")
            continue

        start, end = 0, MAX_NUM_TRS
        series = series[start:end]
        series, mean, std = nisc.scale(series)

        sample = {
            **meta,
            "path": str(path),
            "start": start,
            "end": end,
            "tr": tr,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
        yield sample


def prefetch(root: AnyPath, paths: list[str], *, max_workers: int = 1):
    with tempfile.TemporaryDirectory(prefix="prefetch-") as tmpdir:

        def fn(path: str):
            fullpath = root / path
            if isinstance(fullpath, CloudPath):
                tmppath = Path(tmpdir) / path
                tmppath.parent.mkdir(parents=True, exist_ok=True)
                fullpath = fullpath.download_to(tmppath)
            return path, fullpath

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(fn, p) for p in paths]

            for future in futures:
                path, fullpath = future.result()
                yield path, fullpath

                if str(fullpath).startswith(tmpdir):
                    fullpath.unlink()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--space", type=str, default="fslr64k", choices=list(readers.READER_DICT))
    parser.add_argument("--num_proc", "-j", type=int, default=16)
    args = parser.parse_args()
    sys.exit(main(args))
