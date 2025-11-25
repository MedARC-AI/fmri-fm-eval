from typing import Protocol

import numpy as np

from . import nisc


class Reader(Protocol):
    def __call__(self, path: str) -> np.ndarray: ...


def fslr64k_reader() -> Reader:
    def fn(path: str):
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


def a424_reader() -> Reader:
    parcavg = nisc.parcel_average_a424()

    def fn(path: str):
        series = nisc.read_nifti_data(path)
        series = parcavg(series)
        return series

    return fn


def flat_reader() -> Reader:
    resampler = nisc.flat_resampler_fslr64k_224_560()

    def fn(path: str):
        series = nisc.read_cifti_surf_data(path)
        series = resampler.transform(series, interpolation="linear")
        series = series[:, resampler.mask_]
        return series

    return fn


def mni_reader() -> Reader:
    roi_path = nisc.fetch_schaefer(400, space="mni")
    mask = nisc.read_nifti_data(roi_path) > 0

    def fn(path: str):
        series = nisc.read_nifti_data(path)
        series = series[:, mask]
        return series

    return fn


READER_DICT = {
    "fslr64k": fslr64k_reader,
    "fslr91k": fslr91k_reader,
    "schaefer400": schaefer400_reader,
    "schaefer400_tians3": schaefer400_tians3_reader,
    "a424": a424_reader,
    "flat": flat_reader,
    "mni": mni_reader,
}


DATA_DIMS = {
    "fslr64k": 64984,
    "fslr91k": 91282,
    "schaefer400": 400,
    "schaefer400_tians3": 450,
    "a424": 424,
    "flat": 77763,
    "mni": 132032,
}
