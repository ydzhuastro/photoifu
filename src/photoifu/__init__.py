"""Public PhotoIFU helpers for pixel SED maps and region analysis."""

from .analysis import compare_regions
from .clustering import build_feature_matrix, run_gmm_pca
from .cube import infer_grid, load_pixel_photometry, table_to_map
from .io import load_pixel_table, load_region_config, load_sed_table, validate_geometry
from .maps import (
    build_property_maps,
    fill_isolated_nan_pixels,
    infer_image_shape,
    reconstruct_map,
)
from .regions import (
    assign_regions_to_pixels,
    build_box_mask,
    build_circle_mask,
    build_global_mask,
    build_region_masks,
)
from .stats import compare_region_to_global, summarize_regions

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "assign_regions_to_pixels",
    "build_box_mask",
    "build_circle_mask",
    "build_feature_matrix",
    "build_global_mask",
    "build_property_maps",
    "build_region_masks",
    "compare_regions",
    "compare_region_to_global",
    "fill_isolated_nan_pixels",
    "infer_grid",
    "infer_image_shape",
    "load_pixel_photometry",
    "load_pixel_table",
    "load_region_config",
    "load_sed_table",
    "reconstruct_map",
    "run_gmm_pca",
    "summarize_regions",
    "table_to_map",
    "validate_geometry",
]
