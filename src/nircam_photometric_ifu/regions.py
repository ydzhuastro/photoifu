"""Manual aperture and global-host mask helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .io import validate_geometry


def build_box_mask(
    image_shape: tuple[int, int],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    """Build a boolean mask for an inclusive pixel-coordinate box."""

    ny, nx = image_shape
    x0 = max(0, int(np.floor(xmin)))
    x1 = min(nx - 1, int(np.floor(xmax)))
    y0 = max(0, int(np.floor(ymin)))
    y1 = min(ny - 1, int(np.floor(ymax)))
    if x1 < x0 or y1 < y0:
        raise ValueError("Box bounds do not overlap the image.")

    mask = np.zeros(image_shape, dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def build_circle_mask(image_shape: tuple[int, int], x: float, y: float, r: float) -> np.ndarray:
    """Build a boolean mask for a circular aperture in pixel coordinates."""

    if r < 0:
        raise ValueError("Circle radius must be non-negative.")
    yy, xx = np.ogrid[: image_shape[0], : image_shape[1]]
    return (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(r) ** 2


def _region_items(region_config: Any):
    if isinstance(region_config, Mapping) and "regions" in region_config:
        section = region_config.get("regions", {})
    else:
        section = region_config

    if section is None:
        return
    if isinstance(section, Mapping):
        for name, spec in section.items():
            yield str(name), spec
        return
    if isinstance(section, list):
        for idx, spec in enumerate(section):
            if not isinstance(spec, Mapping):
                raise ValueError("Each region definition must be a mapping.")
            yield str(spec.get("name", f"region_{idx + 1}")), spec
        return
    raise ValueError("region_config must be a mapping, a list, or a config with a 'regions' key.")


def build_region_masks(
    image_shape: tuple[int, int],
    region_config: Mapping[str, Any] | list[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Build named masks from box or circle region definitions."""

    masks: dict[str, np.ndarray] = {}
    for name, spec in _region_items(region_config):
        if not isinstance(spec, Mapping):
            raise ValueError(f"Region {name!r} must be a mapping.")
        shape = str(spec.get("shape", "box")).lower()
        if shape == "box":
            masks[name] = build_box_mask(
                image_shape,
                xmin=spec["xmin"],
                xmax=spec["xmax"],
                ymin=spec["ymin"],
                ymax=spec["ymax"],
            )
        elif shape == "circle":
            masks[name] = build_circle_mask(
                image_shape,
                x=spec["x"],
                y=spec["y"],
                r=spec["r"],
            )
        else:
            raise ValueError(f"Region {name!r} has unsupported shape {shape!r}.")
    return masks


def assign_regions_to_pixels(
    table: pd.DataFrame,
    region_masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Return a copy of ``table`` with a semicolon-separated ``region`` column."""

    validate_geometry(table)
    output = table.copy()
    labels = np.full(len(output), "", dtype=object)
    x = output["x"].astype(int).to_numpy()
    y = output["y"].astype(int).to_numpy()

    for name, mask in region_masks.items():
        in_bounds = (y >= 0) & (y < mask.shape[0]) & (x >= 0) & (x < mask.shape[1])
        hits = np.zeros(len(output), dtype=bool)
        hits[in_bounds] = mask[y[in_bounds], x[in_bounds]]
        for idx in np.flatnonzero(hits):
            labels[idx] = f"{labels[idx]};{name}" if labels[idx] else name

    output["region"] = labels
    return output


def _coerce_exclude_masks(exclude_masks: Any) -> Iterable[np.ndarray]:
    if exclude_masks is None:
        return []
    if isinstance(exclude_masks, np.ndarray):
        return [exclude_masks]
    if isinstance(exclude_masks, Mapping):
        return list(exclude_masks.values())
    return list(exclude_masks)


def _truthy_keep_pixel(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    if np.issubdtype(values.dtype, np.number):
        return values.fillna(0).to_numpy(dtype=float) != 0
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def build_global_mask(
    sed_table: pd.DataFrame,
    image_shape: tuple[int, int],
    exclude_masks: Any = None,
) -> np.ndarray:
    """Build the full valid fitted-host comparison mask.

    The global sample is all fitted SED table pixels, optionally filtered by a
    ``keep_pixel`` column and target-level exclusion masks. It is intentionally
    independent of the manual region boxes.
    """

    validate_geometry(sed_table)
    ny, nx = image_shape
    x = sed_table["x"].astype(int).to_numpy()
    y = sed_table["y"].astype(int).to_numpy()
    valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    if "keep_pixel" in sed_table.columns:
        valid &= _truthy_keep_pixel(sed_table["keep_pixel"])

    mask = np.zeros(image_shape, dtype=bool)
    mask[y[valid], x[valid]] = True

    for exclude in _coerce_exclude_masks(exclude_masks):
        if exclude.shape != image_shape:
            raise ValueError(
                f"Exclude mask shape {exclude.shape} does not match image_shape {image_shape}."
            )
        mask &= ~exclude.astype(bool)

    return mask
