# nircam-photometric-ifu

Lightweight public example code for the workflow behind "NIRCam as a Photometric Integral Field Unit: Resolved Imprints of Galactic Feedback." Given PSF-matched NIRCam photometry and per-pixel SED outputs, the code reconstructs resolved maps, identifies physical pixel populations with GMM/PCA, and compares feedback-associated apertures with the full host distribution.

## What This Repo Does

- Loads a pixel photometry table and a per-pixel SED output table.
- Reconstructs 2D maps of stellar mass, sSFR, dust, metallicity, and recent-SFH proxy values.
- Runs robust-scaled PCA and Gaussian-mixture clustering in physical-property space.
- Applies simple manual box or circle regions.
- Compares representative feedback/channel apertures against all valid fitted host pixels.
- Makes compact Matplotlib figures and CSV summaries.

## What It Does Not Do

This is not a complete SED-fitting pipeline. It does not run Prospector, Parrot, Nautilus, FSPS, or any private production fitting machinery. It starts from tables that you have already produced.

A public version of the full SED-fitting pipeline will be released in a future version of this repo.

## Installation

From this directory:

```bash
python -m pip install -e .
```

The example can also run directly from the source tree without installation:

```bash
cd nircam-photometric-ifu
python examples/run_example.py
```

## Expected Inputs

### Pixel Photometry Table

CSV columns:

- `ID`
- `x`, `y`
- `pixel_index`
- `F090W`, `F115W`, `F150W`, `F200W`, `F277W`, `F356W`, `F444W`
- matching uncertainty columns such as `e_F090W`, `e_F115W`, etc.
- `keep_pixel`

Coordinates are zero-indexed pixel coordinates. The photometry should already be PSF-matched and placed on a common grid.

### Per-Pixel SED Output Table

CSV columns:

- `ID`
- `x`, `y`
- `pixel_index`
- `logmass`
- `log_ssfr` or `logsSFR`
- `dust2`
- `gas_logz`
- `logSFRratio0`

If `log_ssfr` is absent, the code can compute it from `logSFR - logmass` only when both columns are present.

### Region YAML

Manual regions can be boxes or circles. The example uses box apertures:

```yaml
regions:
  A1:
    shape: box
    xmin: 31
    xmax: 44
    ymin: 31
    ymax: 43
    class: A
```

Manual regions are representative apertures. The global comparison sample is not the union of these boxes; it is all valid fitted host pixels after any target-level exclusions.

## Example Data

The files in `examples/data/` are synthetic and small. They mimic a 60 x 60 NIRCam cutout with a masked host galaxy, a lower-dust channel-like feature, a recent-SFH-enhanced region, and a mild metallicity gradient. They are for demonstrating the workflow only and are not unpublished science data.

Running the example writes:

- `outputs/example_sed_maps.png`
- `outputs/example_gmm_pca.png`
- `outputs/example_region_comparisons.png`
- `outputs/example_cluster_pixels.csv`
- `outputs/example_region_summary.csv`

## Clustering Design

The GMM/PCA step deliberately excludes spatial information. The clustering features are SED-derived quantities such as `logmass`, `log_ssfr`, `dust2`, `gas_logz`, and `logSFRratio0`. Pixel coordinates are used later only to reconstruct maps and relate clusters back to the image.

## Citation

If you use this workflow, please cite Zhu et al. in prep / submitted once available.

## License

MIT. See `LICENSE`.
