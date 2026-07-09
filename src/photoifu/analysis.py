"""High-level analysis entry points for PhotoIFU examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import run_gmm_pca
from .maps import build_property_maps
from .stats import summarize_regions


def compare_regions(
    property_maps: Mapping[str, np.ndarray],
    region_masks: Mapping[str, np.ndarray],
    global_mask: np.ndarray,
    region_classes: Mapping[str, Sequence[str]] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare manual regions with all valid fitted host pixels.

    When ``output_path`` is provided, the summary table is also written as CSV.
    """

    summary = summarize_regions(
        property_maps,
        region_masks,
        global_mask,
        region_classes=region_classes,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    return summary


__all__ = ["build_property_maps", "compare_regions", "run_gmm_pca"]
