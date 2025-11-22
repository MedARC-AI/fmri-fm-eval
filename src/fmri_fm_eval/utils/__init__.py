import numpy as np
import nibabel as nib

from .cifti import get_cifti_surf_data

from .parcellation import (
    fetch_schaefer,
    fetch_schaefer_tian,
    ParcelAverage,
)

from .flatmap import (
    load_flat,
    FlatResampler,
    Surface,
    Bbox,
)


def create_flat_resampler(
    subject: str,
    hemi_padding: float,
    bbox: Bbox,
    pixel_size: float,
    roi_path: str | None,
):
    surf, mask = load_flat(subject, hemi_padding=hemi_padding)

    if roi_path:
        roi_img = nib.load(roi_path)
        roi_mask = get_cifti_surf_data(roi_img)
        roi_mask = roi_mask.flatten() > 0
        mask = mask & roi_mask

    resampler = FlatResampler(pixel_size=pixel_size, rect=bbox)
    resampler.fit(surf, mask)
    return resampler


def flat_resampler_fslr64k_224_560():
    resampler = create_flat_resampler(
        subject="32k_fs_LR",
        hemi_padding=8.0,
        bbox=(-336, 336, -122.8, 146),
        pixel_size=1.2,
        roi_path=fetch_schaefer(1000),
    )
    return resampler
