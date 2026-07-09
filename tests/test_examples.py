from pathlib import Path

import numpy as np

from photoifu.analysis import build_property_maps, compare_regions, run_gmm_pca
from photoifu.cube import infer_grid, load_pixel_photometry
from photoifu.io import load_sed_table
from photoifu.regions import build_global_mask, build_region_masks, load_regions


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"


def test_public_example_analysis_smoke(tmp_path):
    pixel_table = load_pixel_photometry(DATA / "cut_image_206183.csv")
    sed_table = load_sed_table(DATA / "all_params_206183.csv")
    regions = load_regions(DATA / "regions_206183.yaml")

    image_shape = infer_grid(pixel_table)
    property_maps = build_property_maps(sed_table, image_shape=image_shape)
    assert {"logmass", "log_ssfr", "dust2", "gas_logz", "logSFRratio0"} <= set(property_maps)
    assert np.isfinite(property_maps["logmass"]).any()

    clustered = run_gmm_pca(sed_table, k=3, random_state=1)
    assert "GMM_cluster" in clustered.columns

    region_masks = build_region_masks(image_shape, regions)
    global_mask = build_global_mask(sed_table, image_shape)
    summary = compare_regions(
        property_maps,
        region_masks,
        global_mask,
        region_classes=regions.get("region_classes", {}),
        output_path=tmp_path / "region_summary.csv",
    )
    assert not summary.empty
    assert (tmp_path / "region_summary.csv").exists()
