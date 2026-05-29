"""Small statistical summaries for region-vs-host comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _finite_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def compare_region_to_global(values_region, values_global) -> dict[str, float | str | int]:
    """Compare a region distribution with the global fitted-host distribution."""

    region = _finite_values(values_region)
    global_values = _finite_values(values_global)
    median_region = float(np.median(region)) if region.size else np.nan
    median_global = float(np.median(global_values)) if global_values.size else np.nan
    delta = median_region - median_global

    p_value = np.nan
    if region.size and global_values.size:
        try:
            from scipy.stats import mannwhitneyu

            result = mannwhitneyu(region, global_values, alternative="two-sided")
            p_value = float(result.pvalue)
        except Exception:
            p_value = np.nan

    if not np.isfinite(delta):
        direction = "insufficient_data"
    elif np.isclose(delta, 0.0):
        direction = "similar"
    elif delta > 0:
        direction = "higher"
    else:
        direction = "lower"

    return {
        "n_region": int(region.size),
        "n_global": int(global_values.size),
        "median_region": median_region,
        "median_global": median_global,
        "delta_median": float(delta) if np.isfinite(delta) else np.nan,
        "p_value": p_value,
        "direction": direction,
    }


def _class_lookup(region_classes: Mapping[str, Sequence[str]] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not region_classes:
        return lookup
    for class_name, members in region_classes.items():
        for member in members:
            lookup[str(member)] = str(class_name)
    return lookup


def summarize_regions(
    property_maps: Mapping[str, np.ndarray],
    region_masks: Mapping[str, np.ndarray],
    global_mask: np.ndarray,
    region_classes: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Summarize each manual region, and optional pooled region classes."""

    rows: list[dict[str, object]] = []
    class_lookup = _class_lookup(region_classes)

    for region_name, mask in region_masks.items():
        for property_name, property_map in property_maps.items():
            stats = compare_region_to_global(property_map[mask], property_map[global_mask])
            rows.append(
                {
                    "region": region_name,
                    "region_class": class_lookup.get(region_name, ""),
                    "property": property_name,
                    **stats,
                }
            )

    if region_classes:
        for class_name, members in region_classes.items():
            pooled = np.zeros_like(global_mask, dtype=bool)
            used_members = []
            for member in members:
                if member in region_masks:
                    pooled |= region_masks[member]
                    used_members.append(str(member))
            if not used_members:
                continue
            label = f"Class {class_name}" if not str(class_name).lower().startswith("class") else str(class_name)
            for property_name, property_map in property_maps.items():
                stats = compare_region_to_global(property_map[pooled], property_map[global_mask])
                rows.append(
                    {
                        "region": label,
                        "region_class": str(class_name),
                        "property": property_name,
                        "members": ";".join(used_members),
                        **stats,
                    }
                )

    return pd.DataFrame(rows)
