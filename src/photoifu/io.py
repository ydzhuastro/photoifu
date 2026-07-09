"""Input/output helpers and lightweight table validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_GEOMETRY_COLUMNS = ("ID", "x", "y", "pixel_index")


def _read_csv_table(path: str | Path, table_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{table_name} not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"{table_name} must be a CSV file for this public example; got {path.suffix!r}."
        )
    try:
        table = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies detailed causes.
        raise ValueError(f"Could not read {table_name} from {path}: {exc}") from exc
    if table.empty:
        raise ValueError(f"{table_name} at {path} is empty.")
    return table


def validate_geometry(table: pd.DataFrame) -> bool:
    """Validate that a per-pixel table has usable image geometry columns.

    The package assumes zero-indexed integer pixel coordinates, with ``x``
    increasing across columns and ``y`` increasing across rows.
    """

    missing = [col for col in REQUIRED_GEOMETRY_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(
            "Missing required geometry column(s): "
            + ", ".join(missing)
            + ". Expected columns are ID, x, y, and pixel_index."
        )

    for col in ("x", "y", "pixel_index"):
        values = pd.to_numeric(table[col], errors="coerce")
        if values.isna().any():
            bad = int(values.isna().sum())
            raise ValueError(f"Column {col!r} contains {bad} non-numeric or missing value(s).")
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"Column {col!r} must contain integer pixel values.")
        if (values < 0).any():
            raise ValueError(f"Column {col!r} must be non-negative.")

    if table["pixel_index"].duplicated().any():
        ndup = int(table["pixel_index"].duplicated().sum())
        raise ValueError(f"Column 'pixel_index' contains {ndup} duplicate value(s).")

    xy = table[["x", "y"]].astype(int)
    if xy.duplicated().any():
        ndup = int(xy.duplicated().sum())
        raise ValueError(f"Columns 'x' and 'y' contain {ndup} duplicate pixel coordinate(s).")

    return True


def load_pixel_table(path: str | Path) -> pd.DataFrame:
    """Load a PSF-matched NIRCam pixel photometry table from CSV."""

    table = _read_csv_table(path, "pixel photometry table")
    validate_geometry(table)
    return table


def load_sed_table(path: str | Path) -> pd.DataFrame:
    """Load a per-pixel SED output table from CSV."""

    table = _read_csv_table(path, "SED output table")
    validate_geometry(table)
    return table


def _validate_region_definition(name: str, definition: dict[str, Any]) -> None:
    if not isinstance(definition, dict):
        raise ValueError(f"Region {name!r} must be a mapping of shape parameters.")

    shape = str(definition.get("shape", "box")).lower()
    if shape == "box":
        required = ("xmin", "xmax", "ymin", "ymax")
    elif shape == "circle":
        required = ("x", "y", "r")
    else:
        raise ValueError(f"Region {name!r} has unsupported shape {shape!r}.")

    missing = [key for key in required if key not in definition]
    if missing:
        raise ValueError(f"Region {name!r} is missing required key(s): {', '.join(missing)}.")


def _iter_region_items(section: Any):
    if section is None:
        return
    if isinstance(section, dict):
        for name, definition in section.items():
            yield str(name), definition
        return
    if isinstance(section, list):
        for idx, definition in enumerate(section):
            if not isinstance(definition, dict):
                raise ValueError("Each list entry in a region section must be a mapping.")
            name = str(definition.get("name", f"region_{idx + 1}"))
            yield name, definition
        return
    raise ValueError("Region sections must be mappings or lists.")


def load_region_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML region configuration."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Region configuration not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse region YAML at {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Region configuration must be a YAML mapping.")

    regions = config.get("regions", {})
    if not regions:
        raise ValueError("Region configuration must define at least one region under 'regions'.")

    for name, definition in _iter_region_items(regions):
        _validate_region_definition(name, definition)

    for name, definition in _iter_region_items(config.get("exclude_regions", {})):
        _validate_region_definition(name, definition)

    return config
