# PhotoIFU

`photoifu` treats PSF-matched multi-band imaging as a low-resolution photometric integral field unit. Each spatial pixel has a coarse SED. The package provides tools to create or load pixel-level photometry tables, run a lightweight public SED-fitting example, reconstruct resolved physical-property maps, identify pixel populations with GMM/PCA, and compare gas-selected apertures with the full fitted host distribution.

## What this repo does

`photoifu` is organized around three steps:

1. **Create pixel data products**  
   Build a pixel-level photometry table from PSF-matched imaging, or load an existing table.
2. **Fit pixel SEDs**  
   Run a lightweight public FSPS-based SED-fitting example that writes an `all_params`-style table. This public fitter is intended for demonstration and reproducibility of the workflow structure, not as the private production pipeline used in the paper.
3. **Analyze resolved SED outputs**  
   Reconstruct maps, run robust-scaled PCA and Gaussian-mixture clustering in physical-property space, apply manual regions, and compare selected apertures with all valid fitted host pixels.

## What this repo does not include yet

This repo does not include the production fitting machinery.

The public SED fitter included here is a lightweight reference implementation. A faster production implementation is planned for a later release. If you would like to apply the full workflow before then, please contact the maintainer; I am happy to discuss running the code or helping set up a comparable workflow.

Emission-line maps require per-pixel best-fit spectra and are not produced by the default cached-table example.

## Installation

```bash
python -m pip install -e .
```

For analysis-only use, the core dependencies are NumPy, pandas, scikit-learn, matplotlib, and PyYAML. SciPy is optional and enables Mann-Whitney p-values in the region summaries.

The SED-fitting demo may require optional FSPS dependencies:

```bash
python -m pip install -e ".[sed]"
```

## Quickstart: analysis-only demo

```bash
python examples/run_analysis_only.py
```

This uses the included compact 206183 example tables and writes:

- `examples/outputs/206183_sed_maps.png`
- `examples/outputs/206183_gmm_pca.png`
- `examples/outputs/206183_region_comparisons.png`
- `examples/outputs/206183_cluster_pixels.csv`
- `examples/outputs/206183_region_summary.csv`

## End-to-end demo

```bash
python examples/run_end_to_end_206183.py
```

This demonstrates the full public workflow from pixel photometry to SED-fitting output and analysis products. The public fitter runs on a small representative subset by default. If FSPS is unavailable, the script falls back to the cached `examples/data/all_params_206183.csv` table and continues with the analysis-only part.

When the SED-fitting step runs, it also writes:

- `examples/outputs/206183_all_params_public_fit.csv`

## Expected inputs

### Pixel photometry table

CSV columns include:

- `ID`
- `x`, `y`
- `pixel_index`
- NIRCam flux columns, e.g. `F090W`, `F115W`, `F150W`, `F200W`, `F277W`, `F356W`, `F444W`
- matching uncertainty columns such as `e_F090W`
- `keep_pixel`

The photometry should already be PSF matched and placed on a common grid.

### Per-pixel SED output table

CSV columns include:

- `ID`
- `x`, `y`
- `pixel_index`
- `logmass`
- `log_ssfr` or `logsSFR`
- `dust2`
- `gas_logz`
- `logSFRratio0`

If `log_ssfr` is absent, `photoifu` can compute it from `logSFR - logmass` when both columns are present.

### Region YAML

Manual regions can be boxes or circles:

```yaml
regions:
  A:
    shape: box
    xmin: 31
    xmax: 44
    ymin: 31
    ymax: 43
    class: A
```

The global comparison sample is not the union of the manual regions. It is all valid fitted host pixels after any target-level exclusions.

## Example data

The included 206183 example data are compact CSV-level products intended to demonstrate the public workflow. They do not include private mosaics. They are sufficient to run the analysis and, where optional dependencies are available, the lightweight public SED-fitting demo.

Included files:

- `examples/data/cut_image_206183.csv`
- `examples/data/all_params_206183.csv`
- `examples/data/regions_206183.yaml`

## Public API

```python
from photoifu.io import load_pixel_table, load_sed_table
from photoifu.cube import infer_grid, load_pixel_photometry, table_to_map
from photoifu.sed import fit_sed_table
from photoifu.analysis import build_property_maps, run_gmm_pca, compare_regions
from photoifu.plotting import plot_sed_maps, plot_gmm_pca, plot_region_comparisons
from photoifu.regions import load_regions, assign_region_labels
```

## Clustering design

The GMM/PCA step deliberately excludes spatial information. The clustering features are SED-derived quantities such as `logmass`, `log_ssfr`, `dust2`, `gas_logz`, and `logSFRratio0`. Pixel coordinates are used only to reconstruct maps and relate clusters or manual regions back to the image.

## Citation

If you use this workflow, please cite Zhu et al. submitted / in preparation until the final citation is available.

## License

MIT. See `LICENSE`.
