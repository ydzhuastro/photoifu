#!/usr/bin/env python
"""Run the public PhotoIFU 206183 workflow from pixel photometry to analysis."""

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

from photoifu.cube import infer_grid, load_pixel_photometry
from photoifu.sed import FSPS_MESSAGE, SedFittingDependencyError, fit_sed_table

from run_analysis_only import run_analysis


def main() -> None:
    data_dir = ROOT / "examples" / "data"
    pixel_table = load_pixel_photometry(data_dir / "cut_image_206183.csv")
    image_shape = infer_grid(pixel_table)
    print(f"Loaded {len(pixel_table)} pixel rows on a {image_shape[1]} x {image_shape[0]} grid.")

    public_fit_path = OUTPUT_DIR / "206183_all_params_public_fit.csv"
    try:
        fit_sed_table(pixel_table, output_path=public_fit_path, max_pixels=40)
        print(f"Wrote {public_fit_path.relative_to(ROOT)}")
        print("Public SED-fitting demo ran with python-fsps/FSPS on a small pixel subset.")
    except SedFittingDependencyError:
        print(FSPS_MESSAGE)
        print("Continuing with the cached all_params_206183.csv analysis table.")

    run_analysis(sed_table_path=data_dir / "all_params_206183.csv", output_prefix="206183")


if __name__ == "__main__":
    main()
