"""Create AABC evaluation dataset with REST only (Arrow format).

This script creates HuggingFace Arrow datasets containing REST fMRI only:
- REST: 1912 TRs -> windowed to 3 x 500 TR segments

Using REST-only matches HCPYA eval (rest1lr) and ensures consistent sample sizes.
Supports all parcellations: schaefer400, schaefer400_tians3, flat, a424, mni
Follows the same pattern as HCP-YA evaluation datasets.
"""
import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np

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
AABC_ROOT = Path(os.getenv("AABC_ROOT", "/teamspace/studios/this_studio/AABC_data"))

# Evaluation pool uses batches 18-19, then split 70/15/15
# Batches 0-17 are reserved for pretraining
EVAL_BATCHES = [18, 19]
SPLIT_SEED = 2912

# Split ratios (train/val/test)
SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

# AABC TR (constant across all tasks)
AABC_TR = 0.72

# Task configurations: (directory_name, file_suffix, window_size, max_windows)
# REST only: window into 500-TR segments (up to 3 windows from 1912 TRs)
TASK_CONFIG = {
    "REST": ("rfMRI_REST", "rfMRI_REST_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii", 500, 3),
}


def main(args):
    outdir = ROOT / f"data/processed/aabc.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    if args.space == "mni":
        has_nifti = next(AABC_ROOT.rglob("*.nii.gz"), None) is not None
        if not has_nifti:
            _logger.warning(
                "Space '%s' requires volumetric NIfTI inputs, but none were found under %s. Skipping.",
                args.space,
                AABC_ROOT,
            )
            return 0
    if args.space == "a424":
        try:
            nisc.fetch_a424(cifti=True)
        except Exception as exc:
            _logger.error(
                "A424 from MSMAll requires a CIFTI parcellation. "
                "Set A424_CIFTI_PATH or provide volumetric NIfTI inputs. (%s)",
                exc,
            )
            return 1

    # Load subject batch splits
    with (ROOT / "metadata/aabc_subject_batch_splits.json").open() as f:
        sub_batch_splits = json.load(f)

    # Find all subject-visit directories
    subject_visits = {}
    for subdir in AABC_ROOT.iterdir():
        if subdir.is_dir() and subdir.name.startswith("HCA"):
            parts = subdir.name.split("_")
            if len(parts) >= 2:
                sub = parts[0]
                visit = parts[1]
                if sub not in subject_visits:
                    subject_visits[sub] = []
                subject_visits[sub].append(visit)

    _logger.info("Found %d subjects with visits", len(subject_visits))

    # Construct pooled sample list (all tasks, with windowing)
    pooled_samples = []
    for batch_id in EVAL_BATCHES:
        for sub in sub_batch_splits[f"batch-{batch_id:02d}"]:
            for visit in subject_visits.get(sub, []):
                for task, (task_dir, suffix, window_size, max_windows) in TASK_CONFIG.items():
                    path = f"{sub}_{visit}_MR/MNINonLinear/Results/{task_dir}/{suffix}"
                    fullpath = AABC_ROOT / path
                    if fullpath.exists():
                        # Create samples for each window
                        for segment in range(max_windows):
                            pooled_samples.append({
                                "path": path,
                                "task": task,
                                "window_size": window_size,
                                "segment": segment,
                            })
    _logger.info("Num pooled samples: %d", len(pooled_samples))

    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(pooled_samples)
    n_total = len(pooled_samples)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["validation"])
    sample_splits = {
        "train": pooled_samples[:n_train],
        "validation": pooled_samples[n_train:n_train + n_val],
        "test": pooled_samples[n_train + n_val:],
    }
    for split, samples in sample_splits.items():
        _logger.info("Num samples (%s): %d", split, len(samples))

    # Count tasks per split
    for split, samples in sample_splits.items():
        task_counts = {}
        for s in samples:
            task = s["task"]
            task_counts[task] = task_counts.get(task, 0) + 1
        _logger.info("  %s task breakdown: %s", split, task_counts)

    # Load reader for target space
    if args.space == "a424":
        reader = readers.a424_reader(cifti=True)
    else:
        reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    _logger.info("Using reader for space '%s' with dimension: %d", args.space, dim)

    # Features (matches HCP-YA pattern with AABC-specific fields)
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "visit": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "segment": hfds.Value("int32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    writer_batch_size = args.writer_batch_size
    if writer_batch_size is None and args.space == "flat":
        writer_batch_size = 16

    # Generate datasets
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, samples in sample_splits.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={"samples": samples, "reader": reader, "dim": dim},
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
            )
        dataset = hfds.DatasetDict(dataset_dict)

        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")

    _logger.info("Dataset saved to: %s", outdir)


def generate_samples(samples: list[dict], *, reader, dim: int):
    """Generate samples for evaluation dataset with windowing."""
    for sample_info in samples:
        path = sample_info["path"]
        task = sample_info["task"]
        window_size = sample_info["window_size"]
        segment = sample_info["segment"]

        fullpath = AABC_ROOT / path
        meta = parse_aabc_metadata(fullpath)

        try:
            series = reader(fullpath)
            T, D = series.shape
            assert D == dim, f"Expected dim {dim}, got {D} for {path}"

            start = segment * window_size
            end = start + window_size

            # Check if we have enough data for this segment
            if end > T:
                if segment == 0:
                    # For first segment, use what we have if it's close enough
                    if T >= window_size * 0.9:
                        end = T
                    else:
                        _logger.warning(
                            "Path %s has fewer TRs than expected (%d < %d); skipping.",
                            path, T, window_size
                        )
                        continue
                else:
                    # Skip additional segments that don't have enough data
                    continue

            series_window = series[start:end]
            series_window, mean, std = nisc.scale(series_window)

            yield {
                **meta,
                "task": task,  # Use task from config, not parsed
                "path": str(path),
                "start": start,
                "end": end,
                "tr": AABC_TR,
                "segment": segment,
                "bold": series_window.astype(np.float16),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }

        except Exception as e:
            _logger.error("Error processing %s: %s", path, e)
            continue


def parse_aabc_metadata(path: Path) -> dict[str, str]:
    """Parse AABC metadata from file path.

    Path structure: HCA{sub}_{visit}_MR/MNINonLinear/Results/{task_dir}/{scan}.dtseries.nii
    """
    subject_visit_dir = path.parents[3].name
    parts = subject_visit_dir.split("_")

    sub = parts[0]  # e.g., "HCA6000030"
    visit = parts[1]  # e.g., "V1"
    mod = "MR"

    return {"sub": sub, "visit": visit, "mod": mod}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AABC evaluation dataset")
    parser.add_argument(
        "--space",
        type=str,
        default="flat",
        choices=list(readers.READER_DICT),
        help="Target anatomical space for processing (default: flat)"
    )
    parser.add_argument(
        "--num_proc",
        "-j",
        type=int,
        default=32,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=None,
        help="Arrow writer batch size (default: 16 for flat, otherwise datasets default)"
    )
    args = parser.parse_args()
    sys.exit(main(args))
