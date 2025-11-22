"""Utilities for processing surface vertex data to flat maps.

Includes utils for:
- loading pycortex flat maps
- resampling surface vertex data to flat raster grid using pycortex flat surfaces
"""

import math
from typing import Any, Literal, NamedTuple

import cortex
import numpy as np
import scipy.interpolate
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


class Surface(NamedTuple):
    points: np.ndarray
    polys: np.ndarray


def load_flat(
    subject: str = "fsaverage",
    hemi_padding: float = 8.0,
    edge_threshold: float | None = 16.0,
) -> tuple[Surface, np.ndarray]:
    """Load merged flat surface from pycortex.

    Returns tuple of surface and valid vertex mask.
    """
    maybe_download_subject(subject)
    surf_lh = cortex.db.get_surf(subject, "flat", hemisphere="lh")
    surf_rh = cortex.db.get_surf(subject, "flat", hemisphere="rh")
    surf = stack_surfaces(surf_lh, surf_rh, hemi_padding=hemi_padding)

    points, polys = surf
    # Drop points with non-zero z component (known issue of pycortex flat maps).
    # Cf https://github.com/gallantlab/pycortex/issues/497
    mask = np.abs(points[:, 2]) < 1e-5
    # Drop z coordinate.
    points = points[:, :2]
    # Drop overly stretched out triangles at edges.
    if edge_threshold is not None and edge_threshold > 0:
        lengths = triangle_longest_side((points, polys))
        polys = polys[lengths < edge_threshold]

    return Surface(points, polys), mask


def maybe_download_subject(subject: str):
    """Download pycortex subject."""
    id_to_url = {
        "32k_fs_LR": "https://figshare.com/ndownloader/files/58130806",
    }
    if subject not in cortex.db.subjects:
        cortex.download_subject(subject, url=id_to_url.get(subject))


def stack_surfaces(surf_lh: Surface, surf_rh: Surface, hemi_padding: float = 8.0) -> Surface:
    """Stack left and right surfaces along the x (left-right) axis."""
    points_lh, polys_lh = surf_lh
    points_rh, polys_rh = surf_rh

    points_lh = points_lh.copy()
    points_lh[:, 0] = points_lh[:, 0] - points_lh[:, 0].max() - hemi_padding

    points_rh = points_rh.copy()
    points_rh[:, 0] = points_rh[:, 0] - points_rh[:, 0].min() + hemi_padding

    points = np.concatenate([points_lh, points_rh])
    polys = np.concatenate([polys_lh, len(points_lh) + polys_rh])
    return Surface(points, polys)


def extract_patch(surf: Surface, mask: np.ndarray) -> Surface:
    """Extract the surface patch for a given mask."""
    points, polys = surf
    mask = mask.astype(bool)

    mask_points = points[mask]
    mask_indices = np.cumsum(mask) - 1
    poly_mask = mask[polys]
    poly_mask = np.all(poly_mask, axis=1)
    mask_polys = polys[poly_mask]
    mask_polys = mask_indices[mask_polys]
    return Surface(mask_points, mask_polys)


def triangle_area(surf: Surface) -> np.ndarray:
    """Calculate the area of each triangle."""
    points, polys = surf
    assert points.shape[1] == 2, "triangle area only implemented for 2D surfaces."
    A = points[polys[:, 0]]
    B = points[polys[:, 1]]
    C = points[polys[:, 2]]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    return 0.5 * np.abs(cross)


def triangle_longest_side(surf: Surface) -> np.ndarray:
    """Calculate the longest side length of each triangle."""
    points, polys = surf
    A = points[polys[:, 0]]
    B = points[polys[:, 1]]
    C = points[polys[:, 2]]
    ab = np.linalg.norm(B - A, axis=1)
    bc = np.linalg.norm(C - B, axis=1)
    ca = np.linalg.norm(A - C, axis=1)
    return np.maximum.reduce([ab, bc, ca])


class Bbox(NamedTuple):
    """Bounding box with format (xmin, xmax, ymin, ymax)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float


class FlatResampler:
    """Resample data from a surface mesh to a raster grid using a flat map.

    Args:
        pixel_size: size of desired pixels in original units.
        rect: Initial data bounding box (left, right, bottom, top), before any padding.
            Defaults to the bounding box of the points.
        pad_width: Width of padding in pixels. If an integer, the same padding is
            applied to all sides. If a list of tuples, the padding is applied
            independently to each side.
        pad_to_multiple: Pad the grid width and height to be a multiple of this value.
    """

    surf_: Surface
    bbox_: Bbox
    grid_: np.ndarray
    point_mask_: np.ndarray | None
    mask_: np.ndarray
    n_points_: int

    def __init__(
        self,
        pixel_size: float,
        rect: Bbox | None = None,
        pad_width: int | None = None,
        pad_to_multiple: int | None = None,
    ):
        self.pixel_size = pixel_size
        self.rect = rect
        self.pad_width = pad_width
        self.pad_to_multiple = pad_to_multiple

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixel_size={self.pixel_size}, rect={self.rect})"

    def fit(self, surf: Surface, mask: np.ndarray | None = None) -> "FlatResampler":
        if mask is not None:
            surf = extract_patch(surf, mask)
        points, polys = surf

        # Fit raster grid to the scattered points.
        grid, bbox = fit_grid(
            points,
            pixel_size=self.pixel_size,
            rect=self.rect,
            pad_width=self.pad_width,
            pad_to_multiple=self.pad_to_multiple,
        )

        # Get mask of pixels contained in the surface interior.
        tri = Triangulation(points[:, 0], points[:, 1], polys)
        trifinder = tri.get_trifinder()
        tri_indices = trifinder(grid[0], grid[1])
        img_mask = tri_indices >= 0

        # Pre-compute delaunay triangulation for linear interpolation.
        delaunay = Delaunay(points)

        # Pre-compute vertex to grid neighbor mapping for nearest interpolation.
        mask_points = grid[:, img_mask].T
        nbrs = NearestNeighbors().fit(points)
        neigh_ind = nbrs.kneighbors(mask_points, n_neighbors=1, return_distance=False)
        neigh_ind = neigh_ind.squeeze(1)

        nbrs = NearestNeighbors().fit(mask_points)
        inverse_neigh_ind = nbrs.kneighbors(points, n_neighbors=1, return_distance=False)
        inverse_neigh_ind = inverse_neigh_ind.squeeze(1)

        self.surf_ = surf
        self.bbox_ = bbox
        self.grid_ = grid
        self.point_mask_ = mask
        self.mask_ = img_mask
        self.n_points_ = len(points)

        self._delaunay = delaunay
        self._neigh_ind = neigh_ind
        self._inverse_neigh_ind = inverse_neigh_ind
        return self

    def transform(
        self,
        data: np.ndarray,
        fill_value: Any = 0,
        interpolation: Literal["nearest", "linear"] = "nearest",
    ) -> np.ndarray:
        """Transform scattered data onto regular grid.

        Args:
            data: scattered data, shape (..., n_points,).
            fill_value: value to fill pixels outside mask area.
            interpolation: interpolation method (nearest or linear)

        Returns:
            Transformed data, shape (..., height, width).
        """
        assert hasattr(self, "bbox_"), "Resampler not fit; call fit() first."
        assert interpolation in ("nearest", "linear"), "Invalid interpolation"

        if self.point_mask_ is not None:
            data = data[..., self.point_mask_]
        assert data.shape[-1] == self.n_points_, "Data does not match resampler."

        if interpolation == "linear":
            data = self._linear_interpolate(data)
        else:
            data = self._nearest_interpolate(data)
        data = np.where(self.mask_, data, fill_value)
        return data

    def inverse(self, data: np.ndarray, fill_value: Any = 0):
        data = data[..., self.mask_]
        point_data = data[..., self._inverse_neigh_ind]

        if self.point_mask_ is not None:
            leading_dims = data.shape[:-1]
            point_data_ = np.full(
                (*leading_dims, len(self.point_mask_)),
                fill_value=fill_value,
                dtype=data.dtype,
            )
            point_data_[..., self.point_mask_] = point_data
            point_data = point_data_
        return point_data

    def _linear_interpolate(self, data: np.ndarray) -> np.ndarray:
        """Resample vertex data to flat map grid using linear interpolation."""
        is_nd = data.ndim > 1
        if is_nd:
            leading_dims = data.shape[:-1]
            data = data.reshape(-1, data.shape[-1]).T
        interp = scipy.interpolate.LinearNDInterpolator(self._delaunay, data)
        xx, yy = self.grid_
        flat_data = interp(xx, yy)
        if is_nd:
            flat_data = np.transpose(flat_data, (2, 0, 1))
            flat_data = flat_data.reshape(leading_dims + self.mask_.shape)
        return flat_data

    def _nearest_interpolate(self, data: np.ndarray) -> np.ndarray:
        """Resample vertex data to flat map grid using nearest neighbor mapping."""
        leading_dims = data.shape[:-1]
        flat_data = np.zeros(leading_dims + self.mask_.shape, dtype=data.dtype)
        flat_data[..., self.mask_] = data[..., self._neigh_ind]
        return flat_data


def fit_grid(
    points: np.ndarray,
    pixel_size: float,
    rect: Bbox | None = None,
    pad_width: int | list[tuple[int, int]] | None = None,
    pad_to_multiple: int | None = None,
) -> tuple[np.ndarray, Bbox]:
    """Fit a regular grid to scattered points with desired padding and pixel size.

    Args:
        points: array of (x, y) points, shape (num_points, 2).
        pixel_size: pixel size in data units.
        rect: Initial data bounding box (left, right, bottom, top), before any padding.
            Defaults to the bounding box of the points.
        pad_width: Width of padding in pixels. If an integer, the same padding is
            applied to all sides. If a list of tuples, the padding is applied
            independently to each side.
        pad_to_multiple: Pad the grid width and height to be a multiple of this value.

    Returns:
        A tuple of the grid, shape (2, height, width), and bounding box.
    """
    if rect is None:
        xmin, ymin = np.floor(points.min(axis=0))
        xmax, ymax = np.ceil(points.max(axis=0))
    else:
        xmin, xmax, ymin, ymax = rect

    w = round((xmax - xmin) / pixel_size)
    h = round((ymax - ymin) / pixel_size)
    xmax = xmin + pixel_size * w
    ymax = ymin + pixel_size * h

    if pad_width:
        xmin, xmax, ymin, ymax = _pad_bbox((xmin, xmax, ymin, ymax), pad_width, pixel_size)

    if pad_to_multiple:
        w = round((xmax - xmin) / pixel_size)
        h = round((ymax - ymin) / pixel_size)
        padw = math.ceil(w / pad_to_multiple) * pad_to_multiple - w
        padh = math.ceil(h / pad_to_multiple) * pad_to_multiple - h
        pad_width_multiple = [
            (padh // 2, padh - padh // 2),
            (padw // 2, padw - padw // 2),
        ]
        xmin, xmax, ymin, ymax = _pad_bbox(
            (xmin, xmax, ymin, ymax),
            pad_width_multiple,
            pixel_size,
        )

    # Nb, this is more reliable than e.g. arange due to floating point errors.
    w = round((xmax - xmin) / pixel_size)
    h = round((ymax - ymin) / pixel_size)
    x = xmin + pixel_size * np.arange(w)
    y = ymax - pixel_size * np.arange(h)  # Nb, upper origin
    grid = np.stack(np.meshgrid(x, y))
    return grid, Bbox(xmin, xmax, ymin, ymax)


def _pad_bbox(
    rect: Bbox,
    pad_width: int | list[tuple[int, int]],
    pixel_size: float,
) -> Bbox:
    xmin, xmax, ymin, ymax = rect
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * 2
    ymin -= pixel_size * pad_width[0][0]
    ymax += pixel_size * pad_width[0][1]
    xmin -= pixel_size * pad_width[1][0]
    xmax += pixel_size * pad_width[1][1]
    return Bbox(xmin, xmax, ymin, ymax)
