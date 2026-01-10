from typing import Protocol

import nibabel as nib
import numpy as np
from templateflow import api as tflow
from nilearn.image import resample_img

from . import nisc


class Reader(Protocol):
    def __call__(self, path: str) -> np.ndarray: ...


def fslr64k_reader() -> Reader:
    def fn(path: str):
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        return series

    return fn


def fslr91k_reader() -> Reader:
    def fn(path: str):
        series = nisc.read_cifti_data(path)
        return series

    return fn


def schaefer400_reader() -> Reader:
    parcavg = nisc.parcel_average_schaefer_fslr64k(400)

    def fn(path: str):
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        series = parcavg(series)
        return series

    return fn


def schaefer400_tians3_reader() -> Reader:
    parcavg = nisc.parcel_average_schaefer_tian_fslr91k(400, 3)

    def fn(path: str):
        series = nisc.read_cifti_data(path)
        series = parcavg(series)
        return series

    return fn


def a424_reader(cifti: bool = False) -> Reader:
    parcavg = nisc.parcel_average_a424(cifti=cifti)

    def fn(path: str):
        series = nisc.read_cifti_data(path) if cifti else nisc.read_nifti_data(path)
        series = parcavg(series)
        return series

    return fn


def schaefer400_tians3_buckner7_reader() -> Reader:
    parcavg = nisc.parcel_average_schaefer400_tians3_buckner7()

    def fn(path: str):
        series = nisc.read_nifti_data(path)
        series = parcavg(series)
        return series

    return fn


def flat_reader() -> Reader:
    resampler = nisc.flat_resampler_fslr64k_224_560()

    def fn(path: str):
        if str(path).endswith(".gii"):
            series = nisc.read_gifti_surf_data(path)
        else:
            series = nisc.read_cifti_surf_data(path)
        series = resampler.transform(series, interpolation="linear")
        series = series[:, resampler.mask_]
        return series

    return fn


def mni_cortex_reader() -> Reader:
    roi_path = nisc.fetch_schaefer(400, space="mni")
    roi_img = nib.load(roi_path)
    mask = np.ascontiguousarray(roi_img.get_fdata().T) > 0

    def fn(path: str):
        img = nib.load(path)
        img = _ensure_mni152_2mm(img)
        series = np.ascontiguousarray(img.get_fdata().T)
        series = series[:, mask]
        return series

    return fn


def mni_reader() -> Reader:
    roi_path = tflow.get(
        "MNI152NLin6Asym", desc="brain", resolution=2, suffix="mask", extension="nii.gz"
    )
    roi_img = nib.load(roi_path)
    mask = np.ascontiguousarray(roi_img.get_fdata().T) > 0

    def fn(path: str):
        img = nib.load(path)
        img = _ensure_mni152_2mm(img)
        series = np.ascontiguousarray(img.get_fdata().T)
        series = series[:, mask]
        return series

    return fn


MNI152_2MM_SHAPE = (91, 109, 91)
MNI152_2MM_AFFINE = (
    (-2.0, 0.0, 0.0, 90.0),
    (0.0, 2.0, 0.0, -126.0),
    (0.0, 0.0, 2.0, -72.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _ensure_mni152_2mm(img: nib.Nifti1Image, interpolation: str = "linear"):
    if img.shape[:3] != MNI152_2MM_SHAPE:
        img = resample_img(
            img,
            target_affine=np.array(MNI152_2MM_AFFINE),
            target_shape=MNI152_2MM_SHAPE,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True,
        )
    return img


READER_DICT = {
    "fslr64k": fslr64k_reader,
    "fslr91k": fslr91k_reader,
    "schaefer400": schaefer400_reader,
    "schaefer400_tians3": schaefer400_tians3_reader,
    "schaefer400_tians3_buckner7": schaefer400_tians3_buckner7_reader,
    "a424": a424_reader,
    "flat": flat_reader,
    "mni": mni_reader,
    "mni_cortex": mni_cortex_reader,
}


DATA_DIMS = {
    "fslr64k": 64984,
    "fslr91k": 91282,
    "schaefer400": 400,
    "schaefer400_tians3": 450,
    "schaefer400_tians3_buckner7": 457,
    "a424": 424,
    "flat": 77763,
    "mni": 228483,
    "mni_cortex": 132032,
}
