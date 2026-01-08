import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Upsampled 1.0s TR for high-res func1mm data.
# https://cvnlab.slite.page/p/vjWTghPTb3/Time-series-data
NSD_TR = 1.0

# Images were presented for 3 seconds with a 1 second gap.
# https://cvnlab.slite.page/p/9gFSd5MubN#3834b73a
NSD_IMAGE_PRESENTATION_TIME = 3.0

# https://cvnlab.slite.page/p/h_T_2Djeid/Technical-notes
NSD_SES_PER_SUBJECT = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}
NSD_SUBJECTS = list(NSD_SES_PER_SUBJECT)

ROOT = Path(__file__).parents[1]
NSD_ROOT = ROOT / "data/NSD"

NSD_TOTAL_RUNS = 3600


def main():
    design_paths = sorted(NSD_ROOT.rglob("design_session*_run*.tsv"))
    print(f"NSD total runs: {len(design_paths)}")
    assert len(design_paths) == NSD_TOTAL_RUNS

    run_table = []
    trial_table = []
    trial_ids = {f"subj{ii:02d}": 0 for ii in range(1, 9)}
    for path in tqdm(design_paths):
        meta = parse_nsd_metadata(path)
        sub = meta["sub"]
        events = load_nsd_events(path)
        run_table.append({**meta, "n_trials": len(events)})
        for event in events:
            trial_id = trial_ids[sub]
            trial_table.append({**meta, "trial_id": trial_id, **event})
            trial_ids[sub] += 1

    run_table = pd.DataFrame.from_records(run_table)
    run_table.to_parquet(ROOT / "metadata/nsd_run_metadata.parquet")

    trial_table = pd.DataFrame.from_records(trial_table)
    trial_table.to_parquet(ROOT / "metadata/nsd_trial_metadata.parquet")


def parse_nsd_metadata(path: Path) -> dict[str, Any]:
    match = re.search(r"(subj[0-9]+)/.*/.*_session([0-9]+)_run([0-9]+)\.", str(path))
    metadata = {
        "sub": match.group(1),
        "ses": int(match.group(2)),
        "run": int(match.group(3)),
    }
    return metadata


def load_nsd_events(path: str | Path) -> list[dict[str, Any]]:
    """Load NSD image presentation events.

    Returns a list of records following the BIDS events specification. The nsd_id field
    is the 0-based NSD image index.

    Reference:
        https://cvnlab.slite.page/p/vjWTghPTb3#bb24a15b
    """

    design = np.loadtxt(path, dtype=np.int64)
    trial_indices = design.nonzero()[0]

    # Make 0-based to match the nsd_stim_info_merged.csv table.
    # https://cvnlab.slite.page/p/NKalgWd__F#bf18f984
    nsd_ids = design[trial_indices] - 1

    events = [
        {
            "onset": int(idx) * NSD_TR,
            "duration": NSD_IMAGE_PRESENTATION_TIME,
            "nsd_id": int(nsd_id),
        }
        for idx, nsd_id in zip(trial_indices, nsd_ids)
    ]
    return events


def get_nsd_design(sub: str, ses: int, run: int):
    path = (
        f"nsddata_timeseries/ppdata/{sub}/func1mm/design/design_session{ses:02d}_run{run:02d}.tsv"
    )
    return path


if __name__ == "__main__":
    main()
