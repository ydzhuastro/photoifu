#!/usr/bin/env python
"""Run the synthetic NIRCam photometric-IFU public example."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
mpl_config_dir = Path(tempfile.gettempdir()) / "nircam_photometric_ifu_mplconfig"
mpl_config_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.path.insert(0, str(ROOT / "src"))

from nircam_photometric_ifu.clustering import run_gmm_pca
from nircam_photometric_ifu.io import load_pixel_table, load_region_config, load_sed_table
from nircam_photometric_ifu.maps import build_property_maps, infer_image_shape
from nircam_photometric_ifu.plotting import (
    plot_gmm_pca,
    plot_region_comparisons,
    plot_resolved_maps,
)
from nircam_photometric_ifu.regions import (
    assign_regions_to_pixels,
    build_global_mask,
    build_region_masks,
)
from nircam_photometric_ifu.stats import summarize_regions


def report(path: Path) -> None:
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    data_dir = ROOT / "examples" / "data"
    pixel_table = load_pixel_table(data_dir / "example_cut_image.csv")
    sed_table = load_sed_table(data_dir / "example_all_params.csv")
    region_config = load_region_config(data_dir / "example_regions.yaml")

    image_shape = infer_image_shape(pixel_table)
    property_maps = build_property_maps(sed_table, image_shape=image_shape)

    maps_path = plot_resolved_maps(
        pixel_table,
        property_maps,
        OUTPUT_DIR / "example_sed_maps.png",
        image_shape=image_shape,
    )
    report(maps_path)

    features = ["logmass", "log_ssfr", "dust2", "gas_logz", "logSFRratio0"]
    cluster_table = run_gmm_pca(sed_table, features=features, k=6, random_state=12)

    region_masks = build_region_masks(image_shape, region_config)
    cluster_table = assign_regions_to_pixels(cluster_table, region_masks)
    cluster_csv = OUTPUT_DIR / "example_cluster_pixels.csv"
    cluster_table.to_csv(cluster_csv, index=False)
    report(cluster_csv)

    gmm_path = plot_gmm_pca(
        pixel_table,
        cluster_table,
        OUTPUT_DIR / "example_gmm_pca.png",
        image_shape=image_shape,
    )
    report(gmm_path)

    exclude_masks = {}
    if region_config.get("exclude_regions"):
        exclude_masks = build_region_masks(image_shape, {"regions": region_config["exclude_regions"]})
    global_mask = build_global_mask(sed_table, image_shape=image_shape, exclude_masks=exclude_masks)
    region_classes = region_config.get("region_classes", {})

    summary = summarize_regions(
        property_maps,
        region_masks,
        global_mask,
        region_classes=region_classes,
    )
    summary_csv = OUTPUT_DIR / "example_region_summary.csv"
    summary.to_csv(summary_csv, index=False)
    report(summary_csv)

    region_path = plot_region_comparisons(
        property_maps,
        region_masks,
        global_mask,
        OUTPUT_DIR / "example_region_comparisons.png",
        region_classes=region_classes,
    )
    report(region_path)

    print("Done. Clustering used SED-derived physical features only; x/y were not inputs.")


if __name__ == "__main__":
    main()
