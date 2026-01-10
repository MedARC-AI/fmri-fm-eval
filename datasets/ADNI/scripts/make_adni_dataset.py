# Generate ADNI eval dataset in multiple output spaces.

import argparse
import json
import logging
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

ADNI_FMRIPREP_ROOT = Path("/teamspace/studios/this_studio/ADNI_fmriprep/output")

CURATION_CSV_PATH = Path("/teamspace/studios/this_studio/ADNI_fmriprep/ADNI_curation.csv")

# Default TR if not found in metadata
DEFAULT_TR = 3.0

# File suffixes for different spaces
SPACE_SUFFIXES = {
    "schaefer400": "_task-rest_space-fsLR_den-91k_bold.dtseries.nii",
    "schaefer400_tians3": "_task-rest_space-fsLR_den-91k_bold.dtseries.nii",
    "flat": "_task-rest_space-fsLR_den-91k_bold.dtseries.nii",
    "a424": "_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
    "mni": "_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
}


def main(args):
    outdir = ROOT / f"data/processed/adni-rest.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists() and not args.overwrite:
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    splits_path = ROOT / "metadata/adni_subject_splits.json"
    if not splits_path.exists():
        _logger.error("Subject splits not found. Run make_adni_metadata.py first.")
        return 1
    
    with splits_path.open() as f:
        subject_splits = json.load(f)
    
    suffix = SPACE_SUFFIXES[args.space]
    
    path_splits = {}
    for split, sessions in subject_splits.items():
        paths = []
        subjects_found = set()
        for sess in sessions:
            sub, ses = sess["sub"], sess["ses"]
            rel_path = f"sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}{suffix}"
            full_path = ADNI_FMRIPREP_ROOT / rel_path
            if full_path.exists():
                paths.append(rel_path)
                subjects_found.add(sub)
            else:
                _logger.debug("File not found: %s", full_path)
        
        path_splits[split] = paths
        total_subjects = len(set(s["sub"] for s in sessions))
        _logger.info(
            "Split '%s': %d subjects (%d sessions) found out of %d subjects (%d sessions)",
            split, len(subjects_found), len(paths), total_subjects, len(sessions)
        )
    
    total_found = sum(len(p) for p in path_splits.values())
    if total_found == 0:
        _logger.error("No files found for any split. Check paths.")
        return 1
    
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]
    
    # Dynamically set writer_batch_size (setting 700 MB per shard) 
    estimated_sample_bytes = 200 * dim * 2
    max_batch_bytes = 700 * 1024 * 1024  
    writer_batch_size = max(1, int(max_batch_bytes / estimated_sample_bytes))
    _logger.info("Using writer_batch_size=%d for dim=%d", writer_batch_size, dim)
    
    # Define features
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "ses": hfds.Value("string"),
            "path": hfds.Value("string"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )
    
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, paths in path_splits.items():
            if len(paths) == 0:
                _logger.warning("Split '%s' has no data; skipping.", split)
                continue
            
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={
                    "paths": paths,
                    "root": ADNI_FMRIPREP_ROOT,
                    "reader": reader,
                    "dim": dim,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                writer_batch_size=writer_batch_size,
            )
        
        dataset = hfds.DatasetDict(dataset_dict)
        
        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")
    
    _logger.info("Dataset saved to %s", outdir)
    return 0


def generate_samples(paths: list[str], *, root: Path, reader, dim: int):
    """Generate samples from paths."""
    for path in paths:
        full_path = root / path
        
        meta = parse_adni_metadata(path)
        
        tr = get_tr_from_sidecar(full_path)
        
        try:
            series = reader(str(full_path))
            
            T, D = series.shape
            if D != dim:
                _logger.warning("Path %s has wrong dimension (%d != %d); skipping.", path, D, dim)
                continue
            
            if T < 10:
                _logger.warning("Path %s has too few frames (%d); skipping.", path, T)
                continue
            
            series, mean, std = nisc.scale(series)
            
            sample = {
                "sub": meta["sub"],
                "ses": meta["ses"],
                "path": path,
                "tr": tr,
                "bold": series.astype(np.float16),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
            yield sample
            
        except Exception as e:
            _logger.warning("Error processing %s: %s", path, str(e))
            continue


def parse_adni_metadata(path: str) -> dict[str, str]:
    """Parse subject and session from ADNI BIDS path."""
    parts = Path(path).parts
    
    sub = None
    ses = None
    
    for part in parts:
        if part.startswith("sub-"):
            sub = part[4:]  # Remove "sub-" prefix
        elif part.startswith("ses-"):
            ses = part[4:]  # Remove "ses-" prefix
    
    return {"sub": sub, "ses": ses}


def get_tr_from_sidecar(nifti_path: Path) -> float:
    """Get TR from sidecar JSON file."""
    json_candidates = [
        nifti_path.with_suffix(".json"),
        nifti_path.with_suffix("").with_suffix(".json"),  # For .nii.gz
    ]
    
    if str(nifti_path).endswith(".dtseries.nii"):
        json_path = Path(str(nifti_path).replace(".dtseries.nii", ".json"))
        json_candidates.insert(0, json_path)
    
    for json_path in json_candidates:
        if json_path.exists():
            try:
                with json_path.open() as f:
                    meta = json.load(f)
                if "RepetitionTime" in meta:
                    return float(meta["RepetitionTime"])
            except (json.JSONDecodeError, KeyError):
                pass
    
    _logger.debug("TR not found in sidecar for %s, using default %.2f", nifti_path, DEFAULT_TR)
    return DEFAULT_TR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space",
        type=str,
        default="schaefer400_tians3",
        choices=list(readers.READER_DICT.keys()),
        help="Output space for the dataset",
    )
    parser.add_argument(
        "--num_proc",
        "-j",
        type=int,
        default=16,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    args = parser.parse_args()
    sys.exit(main(args))
