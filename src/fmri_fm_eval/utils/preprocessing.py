import numpy as np
import scipy.signal
import scipy.interpolate


def scale(series: np.ndarray, axis: int = 0, eps: float = 1e-6):
    """Standard scaling (i.e. zscore)."""
    mean = np.mean(series, axis=axis, keepdims=True)
    std = np.std(series, axis=axis, keepdims=True)
    valid_mask = std > eps
    series = (series - mean) / std.clip(min=eps)
    series = series * valid_mask
    mean = mean * valid_mask
    std = std * valid_mask
    return series, mean, std


def resample_timeseries(
    series: np.ndarray,
    tr: float,
    new_tr: float = 1.0,
    kind: str = "cubic",
    antialias: bool = True,
) -> np.ndarray:
    """Resample a time series to a target TR.

    Args:
        series: time series, shape (n_samples, dim).
        tr: repetition time, i.e. 1 / fs.
        new_tr: target repetition time.
        kind: interpolation kind
        antialias: apply an antialising filter before downsampling.

    Returns:
        Resampled time series, shape (n_new_samples, dim).
    """
    if tr == new_tr:
        return series

    fs = 1.0 / tr
    new_fs = 1.0 / new_tr

    # Anti-aliasing low-pass filter
    # Copied from scipy.signal.decimate
    if antialias and new_fs < fs:
        q = fs / new_fs
        sos = scipy.signal.cheby1(8, 0.05, 0.8 / q, output="sos")
        series = scipy.signal.sosfiltfilt(sos, series, axis=0, padtype="even")

    # Nb, this is more reliable than arange(0, duration, tr) due to floating point
    # errors.
    x = tr * np.arange(len(series))
    new_length = int(tr * len(series) / new_tr)
    new_x = new_tr * np.arange(new_length)

    interp = scipy.interpolate.interp1d(x, series, kind=kind, axis=0)
    series = interp(new_x)
    return series
