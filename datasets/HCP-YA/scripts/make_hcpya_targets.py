import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
HCP_UNRESTRICTED_CSV_PATH = ROOT / "metadata/hcpya_unrestricted.csv"

GENDER_MAP = {
    "F": 0,
    "M": 1,
}
GENDER_BINS = ["F", "M"]

# mapping of age ranges to discrete targets
AGE_MAP = {
    "22-25": 0,
    "26-30": 1,
    "31-35": 2,
    "36+": 2,  # merge with previous due to only 14 instances
}
AGE_BINS = [25, 30]

# number of discrete quantile bins
# 3 bins is consistent with age and with some other recent works
NUM_BINS = 3

# Phenotypic/Cognitive targets
#
# - NEOFAC_N: Neuroticism Scale Score NEO-FFI.
# - Flanker_Unadj: Flanker task score.
# - PMAT24_A_CR: Penn Matrix Test (PMAT): Number of Correct Responses.
TARGETS = ["Gender", "Age", "NEOFAC_N", "Flanker_Unadj", "PMAT24_A_CR"]


def main():
    df = pd.read_csv(HCP_UNRESTRICTED_CSV_PATH, dtype={"Subject": str})
    df = df.set_index("Subject")

    outdir = ROOT / "metadata/targets"
    outdir.mkdir(exist_ok=True)

    for target in TARGETS:
        outpath = outdir / f"hcpya_target_map_{target}.json"
        infopath = outdir / f"hcpya_target_info_{target}.json"
        if outpath.exists():
            _logger.warning(f"Output {outpath} exists; skipping.")
            continue

        series = df[target]
        na_mask = series.isna()
        series = series.loc[~na_mask]
        na_count = int(na_mask.sum())

        if target == "Age":
            targets = series.map(AGE_MAP)
            bins = AGE_BINS
        elif target == "Gender":
            targets = series.map(GENDER_MAP)
            bins = GENDER_BINS
        else:
            targets, bins = quantize(series, num_bins=NUM_BINS)

        _, counts = np.unique(targets, return_counts=True)
        targets = targets.to_dict()

        info = {
            "target": target,
            "na_count": na_count,
            "bins": bins,
            "label_counts": counts.tolist(),
        }
        _logger.info(json.dumps(info))

        with outpath.open("w") as f:
            print(json.dumps(targets, indent=4), file=f)

        with infopath.open("w") as f:
            print(json.dumps(info, indent=4), file=f)


def quantize(series: pd.Series, num_bins: int):
    values = series.values

    qs = np.arange(1, num_bins) / num_bins
    bins = np.nanquantile(values, qs)
    bins = np.round(bins, 3).tolist()

    # right=True produces more balanced splits, and is consistent with pandas qcut
    targets = np.digitize(values, bins, right=True)
    targets = pd.Series(targets, index=series.index)
    return targets, bins


if __name__ == "__main__":
    main()
