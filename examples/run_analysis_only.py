#!/usr/bin/env python
"""Run the public PhotoIFU analysis-only 206183 example."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "examples" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
mpl_config_dir = Path(tempfile.gettempdir()) / "photoifu_mplconfig"
mpl_config_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.path.insert(0, str(ROOT / "src"))

from photoifu.analysis import build_property_maps, compare_regions, run_gmm_pca
from photoifu.cube import infer_grid, load_pixel_photometry
from photoifu.io import load_sed_table
from photoifu.plotting import plot_gmm_pca, plot_region_comparisons, plot_sed_maps
from photoifu.regions import assign_region_labels, build_global_mask, build_region_masks, load_regions


def report(path: Path) -> None:
    print(f"Wrote {path.relative_to(ROOT)}")


def run_analysis(
    sed_table_path: Path | None = None,
    output_prefix: str = "206183",
) -> dict[str, Path]:
    data_dir = ROOT / "examples" / "data"
    pixel_table = load_pixel_photometry(data_dir / "cut_image_206183.csv")
    sed_table = load_sed_table(sed_table_path or data_dir / "all_params_206183.csv")
    region_config = load_regions(data_dir / "regions_206183.yaml")

    image_shape = infer_grid(pixel_table)
    property_maps = build_property_maps(sed_table, image_shape=image_shape)

    paths: dict[str, Path] = {}
    paths["maps"] = plot_sed_maps(
        pixel_table,
        property_maps,
        OUTPUT_DIR / f"{output_prefix}_sed_maps.png",
        image_shape=image_shape,
    )
    report(paths["maps"])

    features = ["logmass", "log_ssfr", "dust2", "gas_logz", "logSFRratio0"]
    finite_rows = sed_table[features].apply(lambda col: col.notna()).all(axis=1)
    n_components = min(6, max(1, int(finite_rows.sum()) - 1))
    cluster_table = run_gmm_pca(sed_table, features=features, k=n_components, random_state=12)

    region_masks = build_region_masks(image_shape, region_config)
    cluster_table = assign_region_labels(cluster_table, region_masks)
    paths["clusters"] = OUTPUT_DIR / f"{output_prefix}_cluster_pixels.csv"
    cluster_table.to_csv(paths["clusters"], index=False)
    report(paths["clusters"])

    paths["gmm_pca"] = plot_gmm_pca(
        pixel_table,
        cluster_table,
        OUTPUT_DIR / f"{output_prefix}_gmm_pca.png",
        image_shape=image_shape,
    )
    report(paths["gmm_pca"])

    exclude_masks = {}
    if region_config.get("exclude_regions"):
        exclude_masks = build_region_masks(image_shape, {"regions": region_config["exclude_regions"]})
    global_mask = build_global_mask(sed_table, image_shape=image_shape, exclude_masks=exclude_masks)
    region_classes = region_config.get("region_classes", {})

    paths["summary"] = OUTPUT_DIR / f"{output_prefix}_region_summary.csv"
    compare_regions(
        property_maps,
        region_masks,
        global_mask,
        region_classes=region_classes,
        output_path=paths["summary"],
    )
    report(paths["summary"])

    paths["regions"] = plot_region_comparisons(
        property_maps,
        region_masks,
        global_mask,
        OUTPUT_DIR / f"{output_prefix}_region_comparisons.png",
        region_classes=region_classes,
    )
    report(paths["regions"])

    print("Done. GMM/PCA used SED-derived physical features only; x/y were not inputs.")
    return paths


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
