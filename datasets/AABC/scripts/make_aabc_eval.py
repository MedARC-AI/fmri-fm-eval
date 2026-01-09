"""Create AABC evaluation dataset with REST only (Arrow format).

This script creates HuggingFace Arrow datasets containing REST fMRI only:
- REST: 1912 TRs -> single 500 TR window per subject

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
AABC_ROOT = os.getenv("AABC_ROOT")
assert AABC_ROOT is not None, (
    "AABC_ROOT environment variable is not set. "
    "Please set it to the directory containing AABC raw data. "
)
AABC_ROOT = Path(AABC_ROOT)
# Evaluation pool uses batches 10-19 (10 batches), then split 80/10/10
# Uses only 1 visit per subject (randomly selected) to maintain data quantity
# Batches 0-9 are reserved for pretraining
EVAL_BATCHES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
SPLIT_SEED = 2912

# Split ratios (train/val/test)
SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}

# AABC TR (constant across all tasks)
AABC_TR = 0.72

# Task configurations: (directory_name, file_suffix, window_size, max_windows)
# REST only: single 500-TR window per subject (no multiple windows)
TASK_CONFIG = {
    "REST": ("rfMRI_REST", "rfMRI_REST_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii", 500, 1),
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
    # Important: Use only ONE visit per subject for eval to avoid data leakage
    rng = np.random.default_rng(SPLIT_SEED)
    samples_by_batch = {batch_id: [] for batch_id in EVAL_BATCHES}

    for batch_id in EVAL_BATCHES:
        for sub in sub_batch_splits[f"batch-{batch_id:02d}"]:
            visits = subject_visits.get(sub, [])
            if not visits:
                continue
            # Randomly select ONE visit per subject (deterministic due to fixed seed)
            selected_visit = rng.choice(visits)
            for task, (task_dir, suffix, window_size, max_windows) in TASK_CONFIG.items():
                path = f"{sub}_{selected_visit}_MR/MNINonLinear/Results/{task_dir}/{suffix}"
                fullpath = AABC_ROOT / path
                if fullpath.exists():
                    # Create samples for each window
                    for segment in range(max_windows):
                        samples_by_batch[batch_id].append({
                            "path": path,
                            "task": task,
                            "window_size": window_size,
                            "segment": segment,
                            "sub": sub,  # Track subject for stratification
                        })

    n_total = sum(len(samples) for samples in samples_by_batch.values())
    _logger.info("Num pooled samples (1 visit per subject): %d", n_total)

    # Load metadata for stratified splitting
    metadata_dir = ROOT / "metadata/targets"
    target_maps = {}
    target_files = ["age_open", "sex", "FluidIQ_Tr35_60y"]

    for target in target_files:
        target_path = metadata_dir / f"aabc_target_map_{target}.json"
        if target_path.exists():
            with target_path.open() as f:
                target_maps[target] = json.load(f)

    # Combine all samples
    all_samples = []
    for batch_samples in samples_by_batch.values():
        all_samples.extend(batch_samples)

    # Stratified split by age + sex + FluidIQ to ensure balanced metadata distribution
    samples_by_strata = {}
    for sample in all_samples:
        sub = sample["sub"]

        # Get metadata bins
        age_bin = target_maps.get("age_open", {}).get(sub, -1)
        sex = target_maps.get("sex", {}).get(sub, -1)
        fluid_bin = target_maps.get("FluidIQ_Tr35_60y", {}).get(sub, -1)

        # Create composite strata key
        strata_key = f"age{age_bin}_sex{sex}_fluid{fluid_bin}"

        if strata_key not in samples_by_strata:
            samples_by_strata[strata_key] = []
        samples_by_strata[strata_key].append(sample)

    _logger.info("Stratification: Created %d strata", len(samples_by_strata))

    # Stratified split: split each stratum proportionally
    sample_splits = {"train": [], "validation": [], "test": []}

    for strata_key, strata_samples in samples_by_strata.items():
        # Shuffle samples within stratum
        rng.shuffle(strata_samples)
        n_strata = len(strata_samples)

        # For very small strata (< 10 samples), use simpler split
        if n_strata < 10:
            # Put most in train, 1 in val if possible, 1 in test if possible
            if n_strata >= 3:
                sample_splits["validation"].append(strata_samples[0])
                sample_splits["test"].append(strata_samples[1])
                sample_splits["train"].extend(strata_samples[2:])
            elif n_strata == 2:
                sample_splits["validation"].append(strata_samples[0])
                sample_splits["train"].append(strata_samples[1])
            else:
                sample_splits["train"].extend(strata_samples)
        else:
            # For larger strata, do proper 80/10/10 split
            n_val = int(n_strata * SPLIT_RATIOS["validation"])
            n_test = int(n_strata * SPLIT_RATIOS["test"])

            sample_splits["validation"].extend(strata_samples[:n_val])
            sample_splits["test"].extend(strata_samples[n_val:n_val + n_test])
            sample_splits["train"].extend(strata_samples[n_val + n_test:])

    # Shuffle within each split
    for split in sample_splits:
        rng.shuffle(sample_splits[split])
    for split, samples in sample_splits.items():
        _logger.info("Num samples (%s): %d", split, len(samples))

    # Count tasks per split
    for split, samples in sample_splits.items():
        task_counts = {}
        for s in samples:
            task = s["task"]
            task_counts[task] = task_counts.get(task, 0) + 1
        _logger.info("  %s task breakdown: %s", split, task_counts)

    # Load and verify metadata distribution across splits
    _logger.info("\n" + "="*80)
    _logger.info("METADATA DISTRIBUTION ACROSS SPLITS")
    _logger.info("="*80)

    # Load all target maps
    target_maps = {}
    metadata_dir = ROOT / "metadata/targets"
    target_files = [
        "age_open", "sex", "Memory_Tr35_60y", "FluidIQ_Tr35_60y",
        "CrystIQ_Tr35_60y", "neo_n", "neo_e", "neo_o", "neo_a", "neo_c"
    ]

    for target in target_files:
        target_path = metadata_dir / f"aabc_target_map_{target}.json"
        if target_path.exists():
            with target_path.open() as f:
                target_maps[target] = json.load(f)

    if target_maps:
        # Extract subjects from each split
        split_subjects = {}
        for split, samples in sample_splits.items():
            subjects = set()
            for s in samples:
                # Extract subject from path (e.g., "HCA6000030_V1_MR/..." -> "HCA6000030")
                path_parts = s["path"].split("_")
                if path_parts:
                    subjects.add(path_parts[0])
            split_subjects[split] = subjects

        # Compute and display distributions
        for target, target_data in target_maps.items():
            _logger.info(f"\n{target.upper()}:")

            # Load target info for context
            info_path = metadata_dir / f"aabc_target_info_{target}.json"
            target_info = {}
            if info_path.exists():
                with info_path.open() as f:
                    target_info = json.load(f)

            for split, subjects in split_subjects.items():
                # Get target values for subjects in this split
                values = []
                for sub in subjects:
                    if sub in target_data:
                        values.append(target_data[sub])

                if not values:
                    _logger.info(f"  {split}: No data")
                    continue

                # Compute statistics based on target type
                if target == "sex":
                    # Binary classification
                    counts = {0: 0, 1: 0}  # F=0, M=1
                    for v in values:
                        counts[v] = counts.get(v, 0) + 1
                    total = len(values)
                    _logger.info(
                        f"  {split}: n={total}, F={counts[0]} ({100*counts[0]/total:.1f}%), "
                        f"M={counts[1]} ({100*counts[1]/total:.1f}%)"
                    )
                else:
                    # Continuous/binned targets
                    values_array = np.array(values)
                    if "bins" in target_info:
                        # Show bin distribution
                        bins = target_info["bins"]
                        bin_counts = np.bincount(values_array, minlength=len(bins)+1)
                        bin_pcts = [f"{100*c/len(values):.1f}%" for c in bin_counts]
                        _logger.info(
                            f"  {split}: n={len(values)}, bins={bin_counts.tolist()}, "
                            f"pcts={bin_pcts}"
                        )
                    else:
                        _logger.info(
                            f"  {split}: n={len(values)}, dist={np.bincount(values_array).tolist()}"
                        )

        _logger.info("\n" + "="*80)
    else:
        _logger.warning("No target metadata found at %s", metadata_dir)

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
