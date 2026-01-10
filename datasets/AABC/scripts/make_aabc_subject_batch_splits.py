import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import check_random_state

ROOT = Path(__file__).parents[1]
AABC_CSV_PATH = ROOT / "AABC_subjects_2026_01_03_14_21_56.csv"
METADATA_PATH = ROOT / "metadata/aabc_metadata.parquet"

# Number of batches of non-overlapping unrelated subjects
NUM_BATCHES = 20
SEED = 2912


def main():
    outpath = ROOT / "metadata/aabc_subject_batch_splits.json"
    if outpath.exists():
        print(f"Removing existing splits: {outpath}")
        outpath.unlink()

    rng = check_random_state(SEED)

    # Load metadata to get all unique subjects
    metadata_df = pd.read_parquet(METADATA_PATH)
    all_subjects = sorted(metadata_df["sub"].unique())

    print(f"Found {len(all_subjects)} unique subjects")
    print(f"First 10 subjects: {all_subjects[:10]}")

    # Load family groups and sex from AABC phenotypic CSV
    groups, sex_labels = load_aabc_family_groups_and_sex(AABC_CSV_PATH, all_subjects)

    # Use StratifiedGroupKFold to stratify by sex while respecting family groups
    splitter = StratifiedGroupKFold(n_splits=NUM_BATCHES, shuffle=True, random_state=rng)

    splits = {}
    for ii, (_, ind) in enumerate(splitter.split(all_subjects, y=sex_labels, groups=groups)):
        subject_list = [all_subjects[i] for i in ind]
        splits[f"batch-{ii:02d}"] = subject_list
        # Calculate sex distribution for this batch
        batch_sex = [sex_labels[i] for i in ind]
        sex_counts = Counter(batch_sex)
        n_f, n_m = sex_counts.get(0, 0), sex_counts.get(1, 0)
        total = n_f + n_m
        pct_f = n_f / total * 100 if total > 0 else 0
        print(f"Batch {ii:02d}: {len(subject_list)} subjects (F={n_f} [{pct_f:.1f}%], M={n_m})")

    outpath.parent.mkdir(exist_ok=True)
    with outpath.open("w") as f:
        print(json.dumps(splits, indent=4), file=f)

    print(f"\nSaved batch splits to {outpath}")

    # Print allocation summary
    print("\nData allocation:")
    print("  Pretraining: batches 0-13 (14 batches)")
    print("  Eval train:  batches 14-17 (4 batches)")
    print("  Eval val:    batch 18 (1 batch)")
    print("  Eval test:   batch 19 (1 batch)")


def load_aabc_family_groups_and_sex(aabc_csv: str | Path, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load family group IDs and sex labels for each subject.

    Args:
        aabc_csv: Path to AABC phenotypic CSV
        subjects: List of subject IDs (e.g., ["HCA6000030", ...])

    Returns:
        Tuple of:
            - Array of family group indices (relabeled to [0, N))
            - Array of sex labels (0=F, 1=M)
    """
    # Load AABC CSV - the id_event column contains subject-visit IDs like "HCA6000030_V1"
    df = pd.read_csv(aabc_csv, usecols=["id_event", "pedid", "sex"])

    # Extract subject ID from id_event (remove visit suffix)
    # HCA6000030_V1 -> HCA6000030
    df["sub"] = df["id_event"].str.rsplit("_", n=1).str[0]

    # For subjects with multiple visits, use the pedid/sex from the first visit
    # (all visits from same subject should have same pedid and sex)
    df = df.drop_duplicates(subset="sub", keep="first")

    # Set subject ID as index
    df.set_index("sub", inplace=True)

    # Get pedid for each subject in our list
    family_id = df.loc[subjects, "pedid"]

    # Relabel to [0, N) for GroupKFold
    _, family_groups = np.unique(family_id.values, return_inverse=True)

    # Get sex labels (F=0, M=1)
    sex_raw = df.loc[subjects, "sex"].values
    sex_labels = np.array([0 if s == "F" else 1 for s in sex_raw])

    return family_groups, sex_labels


if __name__ == "__main__":
    main()
