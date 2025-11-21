import numpy as np
from nibabel.cifti2 import BrainModelAxis, Cifti2Image


def get_cifti_surf_data(cifti: Cifti2Image) -> np.ndarray:
    lh_data = get_cifti_struct_data(cifti, "CIFTI_STRUCTURE_CORTEX_LEFT")
    rh_data = get_cifti_struct_data(cifti, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    data = np.concatenate([lh_data, rh_data], axis=0)
    return data


def get_cifti_struct_data(cifti: Cifti2Image, struct: str) -> np.ndarray:
    """Get cifti scalar/series data for a given brain structure."""
    axis = get_brain_model_axis(cifti)
    data = cifti.get_fdata().T
    for name, indices, model in axis.iter_structures():
        if name == struct:
            num_verts = model.vertex.max() + 1
            struct_data = np.zeros((num_verts,) + data.shape[1:], dtype=data.dtype)
            struct_data[model.vertex] = data[indices]
            return struct_data
    raise ValueError(f"Invalid cifti struct {struct}")


def get_brain_model_axis(cifti: Cifti2Image) -> BrainModelAxis:
    for ii in range(cifti.ndim):
        axis = cifti.header.get_axis(ii)
        if isinstance(axis, BrainModelAxis):
            return axis
    raise ValueError("No brain model axis found in cifti")


def scale(series: np.ndarray, eps: float = 1e-6):
    mean = np.mean(series, axis=0)
    std = np.std(series, axis=0)
    valid_mask = std > eps
    series = (series - mean) / std.clip(min=eps)
    series = series * valid_mask
    mean = mean * valid_mask
    std = std * valid_mask
    return series, mean, std
