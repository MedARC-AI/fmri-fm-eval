import urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

PARC_CACHE_DIR = Path.home() / ".cache" / "parcellations"


def fetch_schaefer(num_rois: int) -> Path:
    base_url = (
        "https://github.com/ThomasYeoLab/CBIG/raw/refs/heads/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/HCP/fslr32k/cifti/"
    )
    filename = f"Schaefer2018_{num_rois}Parcels_17Networks_order.dlabel.nii"
    path = download_file(base_url, filename, cache_dir=PARC_CACHE_DIR)
    return path


def fetch_schaefer_tian(num_rois: int, scale: int) -> Path:
    base_url = (
        "https://github.com/yetianmed/subcortex/raw/refs/heads/master/"
        "Group-Parcellation/3T/Cortex-Subcortex/"
    )
    filename = f"Schaefer2018_{num_rois}Parcels_17Networks_order_Tian_Subcortex_S{scale}.dlabel.nii"
    path = download_file(base_url, filename, cache_dir=PARC_CACHE_DIR)
    return path


def download_file(base_url: str, filename: str, cache_dir: str | Path):
    url = f"{base_url}/{filename}"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / filename
    if not cached_file.exists():
        urllib.request.urlretrieve(url, cached_file)
        assert cached_file.exists(), f"Download failed: {url}"
    return cached_file


class ParcelAverage:
    def __init__(self, parc: np.ndarray, sparse: bool = True, eps: float = 1e-6):
        self.parc = parc
        self.sparse = sparse
        self.eps = eps
        self.parc_one_hot = parc_to_one_hot(parc, sparse=sparse)

    def __call__(self, series: np.ndarray) -> np.ndarray:
        series = parcellate_timeseries(series, self.parc_one_hot, eps=self.eps)
        return series


def parc_to_one_hot(parc: np.ndarray, sparse: bool = True) -> np.ndarray:
    """Get one hot encoding of the parcellation.

    Args:
        parc: parcellation of shape (num_vertices,) with values in [0, num_rois] where 0
            is background.

    Returns:
        parc_one_hot: one hot encoding of parcellation, shape (num_vertices, num_rois).
    """
    (num_verts,) = parc.shape
    parc = parc.astype(np.int32)
    num_rois = parc.max()

    # one hot parcellation matrix, shape (num_vertices, num_rois)
    if sparse:
        mask = parc > 0
        (row_ind,) = mask.nonzero()
        col_ind = parc[mask] - 1
        values = np.ones(len(row_ind), dtype=np.float32)
        parc_one_hot = csr_array((values, (row_ind, col_ind)), shape=(num_verts, num_rois))
    else:
        parc_one_hot = parc[:, None] == np.arange(1, num_rois + 1)
        parc_one_hot = parc_one_hot.astype(np.float32)
    return parc_one_hot


def parcellate_timeseries(
    series: np.ndarray, parc_one_hot: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Extract parcellated time series.

    Args:
        series: full time series (num_samples, num_vertices)
        parc_one_hot: one hot encoding of parcellation (num_vertices, num_rois)

    Returns:
        parc_series: parcellated time series (num_samples, num_rois)
    """
    parc_one_hot = parc_one_hot.astype(series.dtype)

    # don't include verts with missing data
    valid_mask = np.std(series, axis=0) > eps
    parc_one_hot = parc_one_hot * valid_mask[:, None]

    # normalize weights to sum to 1
    # Nb, empty parcels will be all zero
    parc_counts = np.asarray(parc_one_hot.sum(axis=0))
    parc_one_hot = parc_one_hot / np.maximum(parc_counts, 1)

    # per roi averaging
    parc_series = series @ parc_one_hot
    return parc_series
