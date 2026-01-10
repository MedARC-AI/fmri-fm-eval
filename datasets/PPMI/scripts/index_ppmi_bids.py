from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm


def main():
    root = Path("bids")
    path_list = sorted(root.rglob("*.nii.gz"))
    outpath = Path("metadata/PPMI_BIDS_index.csv")

    if outpath.exists():
        df = pd.read_csv(outpath)
        records = df.to_dict("records")
        completed = set(df["path"])
    else:
        records = []
        completed = set()

    for path in tqdm(path_list):
        relpath = str(path.relative_to(root))
        if relpath in completed:
            continue
        meta = parse_metadata(path)
        info = read_header(path)
        records.append({**meta, **info, "path": relpath})

        if len(records) % 100 == 0:
            pd.DataFrame.from_records(records).to_csv(outpath, index=False)
    pd.DataFrame.from_records(records).to_csv(outpath, index=False)


def parse_metadata(path: Path):
    # poor persons bids parser
    # sub-176185/ses-20230915/func/sub-176185_ses-20230915_task-rest_dir-AP_bold.nii.gz
    stem, ext = path.name.split(".", 1)
    stem, suffix = stem.rsplit("_", 1)
    meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
    meta = {**meta, "suffix": suffix}
    return meta


def read_header(path: Path):
    img = nib.load(path, mmap=True)
    shape = list(img.shape)
    pixdim = [round(v, 2) for v in img.header["pixdim"][1:4].tolist()]
    if len(shape) == 4:
        tr = round(float(img.header["pixdim"][4]), 2)
        num_trs = shape[-1]
        shape = shape[:3]
    else:
        tr = num_trs = None
    info = {"shape": shape, "pixdim": pixdim, "tr": tr, "num_trs": num_trs}
    return info


if __name__ == "__main__":
    main()
