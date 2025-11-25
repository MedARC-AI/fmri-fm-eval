import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_random_state

ROOT = Path(__file__).parents[1]
HCP_RESTRICTED_CSV_PATH = ROOT / "metadata/hcpya_restricted.csv"

HCP_ROOT = ROOT / "data/sourcedata/HCP_1200"

# Total number of HCP subjects released in HCP-1200
# Nb: this is an outdated number of subjects. As of 2025-11-18, there are now 1113
# subjects on s3://hcp-openaccess/HCP_1200. But we're not updating to keep consistency
# with earlier work.
HCP_NUM_SUBJECTS = 1098

# Number of batches of non-overlapping unrelated subjects
NUM_BATCHES = 20
SEED = 2912


def main():
    outpath = ROOT / "metadata/hcpya_subject_batch_splits.json"
    assert not outpath.exists(), f"output splits {outpath} already exist"

    rng = check_random_state(SEED)

    all_subjects = np.array(sorted(p.name for p in HCP_ROOT.glob("[0-9]*")))
    print(f"Found {len(all_subjects)} subjects:\n{all_subjects[:10]}")
    assert len(all_subjects) == HCP_NUM_SUBJECTS, "unexpected number of subjects"

    groups = load_hcp_family_groups(HCP_RESTRICTED_CSV_PATH)
    groups = groups.loc[all_subjects].values

    splitter = GroupKFold(n_splits=NUM_BATCHES, shuffle=True, random_state=rng)

    splits = {}
    for ii, (_, ind) in enumerate(splitter.split(all_subjects, groups=groups)):
        splits[f"batch-{ii:02d}"] = all_subjects[ind].tolist()

    outpath.parent.mkdir(exist_ok=True)
    with outpath.open("w") as f:
        print(json.dumps(splits, indent=4), file=f)


def load_hcp_family_groups(hcp_restricted_csv: str | Path) -> pd.Series:
    df = pd.read_csv(hcp_restricted_csv, dtype={"Subject": str})
    df.set_index("Subject", inplace=True)
    hcp_family_id = df.loc[:, "Pedigree_ID"]

    # Relabel to [0, N)
    _, hcp_family_groups = np.unique(hcp_family_id.values, return_inverse=True)
    hcp_family_groups = pd.Series(
        hcp_family_groups,
        index=hcp_family_id.index,
        name="Family_Group",
    )
    return hcp_family_groups


if __name__ == "__main__":
    main()
