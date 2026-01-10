import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm


def main():
    path_list = np.loadtxt("metadata/PPMI_BIDS_complete_paths.txt", dtype=str).tolist()

    src_dir = Path("bids")
    dst_dir = Path("bids_complete")
    shutil.copy(src_dir / "dataset_description.json", dst_dir / "dataset_description.json")

    for path in tqdm(path_list):
        path = Path(path)

        (dst_dir / path.parent).mkdir(parents=True, exist_ok=True)

        assert (src_dir / path).exists()
        if not (dst_dir / path).exists():
            (dst_dir / path).symlink_to((src_dir / path).resolve())

        path = path.parent / path.name.replace(".nii.gz", ".json")
        assert (src_dir / path).exists()
        if not (dst_dir / path).exists():
            (dst_dir / path).symlink_to((src_dir / path).resolve())


if __name__ == "__main__":
    main()
