"""Pixel-table and image-grid helpers for PhotoIFU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_pixel_table
from .maps import infer_image_shape, reconstruct_map


def load_pixel_photometry(path: str | Path) -> pd.DataFrame:
    """Load a PSF-matched pixel-level photometry table."""

    return load_pixel_table(path)


def infer_grid(table: pd.DataFrame) -> tuple[int, int]:
    """Infer the ``(ny, nx)`` image grid from a pixel table."""

    return infer_image_shape(table)


def table_to_map(
    table: pd.DataFrame,
    value_column: str,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Place one table column on the image grid."""

    return reconstruct_map(table, value_column, image_shape=image_shape)
