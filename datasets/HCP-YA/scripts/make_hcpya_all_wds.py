import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import webdataset as wds
from cloudpathlib import AnyPath, CloudPath, S3Client

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
META_PATH = ROOT / "metadata/hcpya_metadata.parquet"

# Subject batches to generate
# Generate all 20 batches, but we should be careful about which ranges of shards to
# pretrain on.
SUB_BATCHES = list(range(20))

# We use a fixed number of shards per subject batch for convenient dataset subsampling.
# We use a consistent number of shards per batch across data formats so that sampling
# behavior is consistent.
SHARDS_PER_BATCH = 100

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

# Resample all time series to 1s tr. This is mainly for consistency with previous
# versions of the dataset.
TARGET_TR = 1.0
INTERPOLATION = "pchip"

# Fixed seed for shuffling runs within each batch of shards.
SEED = 2146


def main(args):
    # check shard id
    batch_ids = SUB_BATCHES
    shards_per_batch = SHARDS_PER_BATCH
    num_shards = shards_per_batch * len(batch_ids)
    if not (0 <= args.shard_id < num_shards):
        _logger.error(f"Invalid shard_id {args.shard_id}; expected in [0, {num_shards}).")
        return 1

    outdir = AnyPath(args.outdir or ROOT / "data/processed")
    outdir = outdir / f"hcpya-all.{args.space}.wds"
    outpath = outdir / f"hcpya-all-{args.space}-{args.shard_id:05d}.tar"
    _logger.info(
        "Generating dataset: %s (%04d/%d)",
        outdir.relative_to(outdir.parents[1]),
        args.shard_id,
        num_shards,
    )

    # use a different s3 credential for upload
    upload_client = get_r2_client()
    if isinstance(outpath, CloudPath) and upload_client is not None:
        outpath = CloudPath(outpath, client=upload_client)

    if outpath.exists():
        _logger.info("Output %s exists; exiting.", outpath)
        return 0

    # get the batch of subjects for the current shard
    with (ROOT / "metadata/hcpya_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)
    batch_id = batch_ids[args.shard_id // shards_per_batch]
    batch_shard_id = args.shard_id % shards_per_batch
    batch_subjects = sub_batch_splits[f"batch-{batch_id:02d}"]

    # get the preprocessed time series paths for the current batch of subjects
    meta_df = pd.read_parquet(META_PATH)
    sub_mask = meta_df["sub"].isin(batch_subjects)
    batch_series_paths = sorted(meta_df.loc[sub_mask, "path"].values)

    # volume space for a424 and mni, otherwise cifti space
    if args.space in {"a424", "mni", "mni_cortex"}:
        batch_series_paths = [
            p.replace("_Atlas_MSMAll.dtseries.nii", ".nii.gz") for p in batch_series_paths
        ]

    # split series paths into shards
    # shuffle so that data order is randomized when reading a shard sequentially.
    # nb, this will be a different sample order than previous versions of hcp-flat wds.
    rng = np.random.default_rng(SEED)
    rng.shuffle(batch_series_paths)

    path_offsets = np.linspace(0, len(batch_series_paths), shards_per_batch + 1)
    path_offsets = np.round(path_offsets).astype(int)
    path_start, path_stop = path_offsets[batch_shard_id : batch_shard_id + 2]
    shard_series_paths = batch_series_paths[path_start:path_stop]
    _logger.info(
        "Batch %d shard %02d/%d (n=%d)",
        batch_id,
        batch_shard_id,
        shards_per_batch,
        len(shard_series_paths),
    )

    # load the data reader for the target space and look up the data dimension.
    # all readers return a bold data array of shape (n_samples, dim).
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

    root = AnyPath(args.root or HCP_ROOT)

    # Temp output path, in case of incomplete processing.
    with tempfile.TemporaryDirectory(prefix="wds-") as tmp_outdir:
        tmp_outpath = AnyPath(tmp_outdir) / outpath.name

        # Generate wds samples.
        with wds.TarWriter(str(tmp_outpath)) as sink:
            for ii, sample in enumerate(
                generate_samples(shard_series_paths, root=root, reader=reader, dim=dim)
            ):
                sink.write(sample)
                _logger.info("[%02d/%d] %s", ii, len(shard_series_paths), sample["__key__"])

        outpath.parent.mkdir(exist_ok=True, parents=True)
        with tmp_outpath.open("rb") as fsrc:
            with outpath.open("wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)

    _logger.info(f"Done: {outpath}")


def generate_samples(paths: list[str], *, root: AnyPath, reader: readers.Reader, dim: int):
    for path, fullpath in prefetch(root, paths):
        meta = parse_hcp_metadata(fullpath)
        key = "sub-{sub}_mod-{mod}_task-{task}_mag-{mag}_dir-{dir}".format(**meta)
        tr = HCP_TR[meta["mag"]]

        series = reader(fullpath)
        series, mean, std = nisc.scale(series)

        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind=INTERPOLATION)
        tr = TARGET_TR

        T, D = series.shape
        assert D == dim
        n_frames = T

        sample = {
            "__key__": key,
            "meta.json": {**meta, "path": str(path), "n_frames": n_frames, "tr": tr},
            "bold.npy": series.astype(np.float16),
            "mean.npy": mean.astype(np.float32),
            "std.npy": std.astype(np.float32),
        }
        yield sample


def prefetch(root: AnyPath, paths: list[str], *, max_workers: int = 1):
    """Prefetch files from remote storage."""

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


def get_r2_client():
    if "R2_ACCESS_KEY_ID" in os.environ:
        return S3Client(
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ["R2_ENDPOINT_URL_S3"],
        )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--space", type=str, default="flat", choices=list(readers.READER_DICT))
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()
    sys.exit(main(args))
