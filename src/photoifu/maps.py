"""Reconstruct two-dimensional property maps from per-pixel tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import validate_geometry


def infer_image_shape(table: pd.DataFrame) -> tuple[int, int]:
    """Infer ``(ny, nx)`` from zero-indexed ``x`` and ``y`` pixel coordinates."""

    validate_geometry(table)
    ny = int(table["y"].max()) + 1
    nx = int(table["x"].max()) + 1
    if ny <= 0 or nx <= 0:
        raise ValueError("Could not infer a positive image shape from x/y coordinates.")
    return ny, nx


def reconstruct_map(
    table: pd.DataFrame,
    value_column: str,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Place a per-pixel column back onto a 2D image grid."""

    validate_geometry(table)
    if value_column not in table.columns:
        raise ValueError(f"Column {value_column!r} is not present in the table.")
    if image_shape is None:
        image_shape = infer_image_shape(table)

    ny, nx = image_shape
    values = pd.to_numeric(table[value_column], errors="coerce").to_numpy(dtype=float)
    x = table["x"].astype(int).to_numpy()
    y = table["y"].astype(int).to_numpy()
    in_bounds = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    if not np.all(in_bounds):
        nbad = int((~in_bounds).sum())
        raise ValueError(f"{nbad} table row(s) fall outside image_shape={image_shape}.")

    image = np.full(image_shape, np.nan, dtype=float)
    image[y, x] = values
    return image


def fill_isolated_nan_pixels(
    map_2d: np.ndarray,
    method: str = "median",
) -> tuple[np.ndarray, int]:
    """Fill isolated interior non-finite pixels using their eight finite neighbors.

    The operation is a single pass over the original array. Only pixels away from
    the image edge are considered, and only if all eight neighboring pixels are
    finite.
    """

    if method not in {"median", "mean"}:
        raise ValueError("method must be either 'median' or 'mean'.")

    source = np.asarray(map_2d, dtype=float)
    filled = source.copy()
    ny, nx = source.shape
    n_filled = 0

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if np.isfinite(source[y, x]):
                continue
            neighbors = source[y - 1 : y + 2, x - 1 : x + 2].copy()
            neighbors[1, 1] = np.nan
            values = neighbors[np.isfinite(neighbors)]
            if values.size != 8:
                continue
            filled[y, x] = float(np.median(values) if method == "median" else np.mean(values))
            n_filled += 1

    return filled, n_filled


def _table_with_log_ssfr(sed_table: pd.DataFrame) -> pd.DataFrame:
    table = sed_table.copy()
    if "log_ssfr" in table.columns:
        return table
    if "logsSFR" in table.columns:
        table["log_ssfr"] = table["logsSFR"]
        return table
    if {"logSFR", "logmass"}.issubset(table.columns):
        table["log_ssfr"] = table["logSFR"] - table["logmass"]
        return table
    return table


def build_property_maps(
    sed_table: pd.DataFrame,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Build the standard resolved SED property maps used by the example."""

    table = _table_with_log_ssfr(sed_table)
    required = ("logmass", "log_ssfr", "dust2", "gas_logz", "logSFRratio0")
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(
            "Cannot build standard property maps because the SED table is missing: "
            + ", ".join(missing)
            + ". Provide log_ssfr/logsSFR, or provide logSFR together with logmass."
        )

    if image_shape is None:
        image_shape = infer_image_shape(table)

    return {col: reconstruct_map(table, col, image_shape=image_shape) for col in required}
