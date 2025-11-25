import logging
from pathlib import Path
from typing import Any

import datasets as hfds
import nibabel as nib
import pandas as pd

logging.getLogger("nibabel").setLevel(logging.ERROR)

EXCLUDE_ACQS = {
    # Exclude merged retinotopic localizer scan.
    "tfMRI_7T_RETCCW_AP_RETCW_PA_RETEXP_AP_RETCON_PA_RETBAR1_AP_RETBAR2_PA",
}
EXCLUDE_CONDS = {
    # The sync time is not needed, it's already subtracted
    # https://www.mail-archive.com/hcp-users@humanconnectome.org/msg00616.html
    "Sync",
}

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

# Total number of expected runs
HCP_NUM_RUNS = 21633

NUM_PROC = 16

ROOT = Path(__file__).parents[1]

HCP_ROOT = ROOT / "data/sourcedata/HCP_1200"


def main():
    series_paths = sorted(
        path
        for path in HCP_ROOT.rglob("*_Atlas_MSMAll.dtseries.nii")
        if path.parent.name not in EXCLUDE_ACQS
    )
    assert len(series_paths) == HCP_NUM_RUNS, "unexpected number of runs"

    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "mod": hfds.Value("string"),
            "task": hfds.Value("string"),
            "mag": hfds.Value("string"),
            "dir": hfds.Value("string"),
            "events": hfds.List(
                {
                    "onset": hfds.Value("float32"),
                    "duration": hfds.Value("float32"),
                    "trial_type": hfds.Value("string"),
                }
            ),
            "tr": hfds.Value("float32"),
            "n_frames": hfds.Value("int32"),
            "path": hfds.Value("string"),
        }
    )

    dataset = hfds.Dataset.from_generator(
        generate_metadata,
        features=features,
        gen_kwargs={"paths": series_paths},
        num_proc=NUM_PROC,
    )

    dataset.to_parquet(ROOT / "metadata/hcpya_metadata.parquet")


def generate_metadata(paths: list[str]):
    for path in paths:
        path = Path(path)
        meta = parse_hcp_metadata(path)
        tr = HCP_TR[meta["mag"]]

        # load task events if available
        events = load_hcp_events(path.parent)

        img = nib.load(path)
        n_frames = img.shape[0]

        sample = {
            **meta,
            "events": events,
            "tr": tr,
            "n_frames": n_frames,
            "path": str(path.relative_to(HCP_ROOT)),
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


def load_hcp_events(run_dir: Path) -> list[dict[str, Any]]:
    """Read all events from a run directory.

    Returns a list of records following the BIDS events specification.

    EV files have the format `'{cond}.txt'` and look like:

    ```
    30.471  10.443  1.0
    41.16   10.238  1.0
    81.57   10.664  1.0
    92.513  10.114  1.0
    131.537 10.485  1.0
    142.293 10.852  1.0
    153.395 11.93   1.0
    165.577 12.213  1.0
    210.319 10.513  1.0
    ```
    """
    ev_dir = Path(run_dir) / "EVs"
    if not ev_dir.exists():
        return []

    events = []
    ev_paths = ev_dir.glob("*.txt")
    for path in ev_paths:
        name = path.stem
        if name in EXCLUDE_CONDS:
            continue

        cond_events = pd.read_csv(path, sep="\t", names=["onset", "duration", "value"])
        if len(cond_events) == 0:
            continue

        cond_events.drop("value", inplace=True, axis=1)
        cond_events = cond_events.astype({"duration": float})
        cond_events["trial_type"] = name
        events.append(cond_events)

    events = pd.concat(events, axis=0, ignore_index=True)
    events = events.sort_values("onset")
    events = events.to_dict(orient="records")
    return events


if __name__ == "__main__":
    main()
