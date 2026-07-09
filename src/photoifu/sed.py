"""Lightweight public SED-fitting demonstration.

This module is intentionally small and transparent. It is not the private
production fitting machinery used for the paper; it exists so the public
example can show how pixel photometry becomes an ``all_params``-style table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import validate_geometry

FSPS_MESSAGE = (
    "The SED-fitting example requires python-fsps/FSPS. Install the optional "
    "sed dependencies or use the provided all_params_206183.csv for analysis-only mode."
)

FILTER_WAVELENGTH_UM = {
    "F070W": 0.70,
    "F090W": 0.90,
    "F115W": 1.15,
    "F150W": 1.50,
    "F182M": 1.82,
    "F200W": 2.00,
    "F210M": 2.10,
    "F277W": 2.77,
    "F335M": 3.35,
    "F356W": 3.56,
    "F410M": 4.10,
    "F444W": 4.44,
}


class SedFittingDependencyError(ImportError):
    """Raised when the optional SED-fitting dependencies are unavailable."""


@dataclass(frozen=True)
class _GridPoint:
    age_gyr: float
    dust2: float
    gas_logz: float
    logSFRratio0: float
    template: np.ndarray


def _require_fsps():
    try:
        import fsps
    except ImportError as exc:
        raise SedFittingDependencyError(FSPS_MESSAGE) from exc
    return fsps


def _flux_columns(table: pd.DataFrame) -> list[str]:
    columns = [col for col in table.columns if col in FILTER_WAVELENGTH_UM]
    if len(columns) < 3:
        raise ValueError("SED fitting requires at least three recognized NIRCam flux columns.")
    return sorted(columns, key=lambda col: FILTER_WAVELENGTH_UM[col])


def _uncertainty_column(flux_column: str, table: pd.DataFrame) -> str | None:
    for candidate in (f"e_{flux_column}", f"{flux_column}_err", f"{flux_column}_unc"):
        if candidate in table.columns:
            return candidate
    return None


def _selected_pixels(table: pd.DataFrame, max_pixels: int | None) -> pd.DataFrame:
    selected = table.copy()
    if "keep_pixel" in selected.columns:
        keep = selected["keep_pixel"]
        if keep.dtype == bool:
            selected = selected.loc[keep]
        else:
            selected = selected.loc[pd.to_numeric(keep, errors="coerce").fillna(0) != 0]
    if max_pixels is not None and len(selected) > max_pixels:
        index = np.linspace(0, len(selected) - 1, int(max_pixels)).round().astype(int)
        selected = selected.iloc[index]
    return selected.reset_index(drop=True)


def _top_hat_photometry(wave_angstrom: np.ndarray, spectrum: np.ndarray, bands: list[str], redshift: float) -> np.ndarray:
    values = []
    rest_um = wave_angstrom / 1.0e4 * (1.0 + redshift)
    for band in bands:
        center = FILTER_WAVELENGTH_UM[band]
        width = 0.18 * center
        hit = (rest_um >= center - width / 2.0) & (rest_um <= center + width / 2.0)
        if not np.any(hit):
            values.append(np.nan)
        else:
            values.append(float(np.nanmedian(spectrum[hit])))
    values = np.asarray(values, dtype=float)
    norm = np.nanmedian(np.abs(values))
    if not np.isfinite(norm) or norm <= 0:
        norm = 1.0
    return values / norm


def _build_fsps_grid(bands: list[str], redshift: float) -> list[_GridPoint]:
    fsps = _require_fsps()
    try:
        stellar_population = fsps.StellarPopulation(zcontinuous=1, sfh=4, dust_type=2)
    except Exception as exc:  # pragma: no cover - depends on local FSPS runtime setup.
        raise SedFittingDependencyError(FSPS_MESSAGE) from exc
    grid: list[_GridPoint] = []
    ages = (0.08, 0.25, 0.75, 1.8)
    dust_values = (0.05, 0.25, 0.55)
    metallicities = (-0.8, -0.35, 0.0)
    recent_values = (0.35, 0.0, -0.35)

    for age_gyr in ages:
        for dust2 in dust_values:
            for gas_logz in metallicities:
                for recent in recent_values:
                    stellar_population.params["dust2"] = float(dust2)
                    stellar_population.params["logzsol"] = float(gas_logz)
                    try:
                        wave, spectrum = stellar_population.get_spectrum(tage=float(age_gyr), peraa=True)
                    except Exception as exc:  # pragma: no cover - depends on local FSPS runtime setup.
                        raise SedFittingDependencyError(FSPS_MESSAGE) from exc
                    template = _top_hat_photometry(wave, spectrum, bands, redshift=redshift)
                    if np.isfinite(template).all():
                        grid.append(
                            _GridPoint(
                                age_gyr=float(age_gyr),
                                dust2=float(dust2),
                                gas_logz=float(gas_logz),
                                logSFRratio0=float(recent),
                                template=template,
                            )
                        )
    if not grid:
        raise RuntimeError("FSPS grid generation produced no finite model photometry.")
    return grid


def _fit_one_pixel(row: pd.Series, bands: list[str], grid: list[_GridPoint]) -> dict[str, float]:
    flux = pd.to_numeric(row[bands], errors="coerce").to_numpy(dtype=float)
    errors = []
    for band in bands:
        err_col = _uncertainty_column(band, row.to_frame().T)
        errors.append(float(row[err_col]) if err_col else np.nan)
    sigma = np.asarray(errors, dtype=float)
    finite_sigma = np.isfinite(sigma) & (sigma > 0)
    if finite_sigma.any():
        fallback_sigma = float(np.nanmedian(sigma[finite_sigma]))
    else:
        fallback_sigma = max(float(np.nanmedian(np.abs(flux))) * 0.1, 1.0e-3)
    sigma = np.where(finite_sigma, sigma, fallback_sigma)

    finite = np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
    if finite.sum() < 3:
        return {
            "logmass": np.nan,
            "log_ssfr": np.nan,
            "dust2": np.nan,
            "gas_logz": np.nan,
            "logSFRratio0": np.nan,
            "logSFR": np.nan,
        }

    best = None
    best_chi2 = np.inf
    best_scale = np.nan
    for point in grid:
        model = point.template
        numerator = np.sum(flux[finite] * model[finite] / sigma[finite] ** 2)
        denominator = np.sum(model[finite] ** 2 / sigma[finite] ** 2)
        if denominator <= 0:
            continue
        scale = max(numerator / denominator, 1.0e-12)
        chi2 = float(np.sum(((flux[finite] - scale * model[finite]) / sigma[finite]) ** 2))
        if chi2 < best_chi2:
            best = point
            best_chi2 = chi2
            best_scale = scale

    if best is None:
        raise RuntimeError("No finite FSPS grid model could be fit to a selected pixel.")

    logmass = 7.0 + np.log10(max(best_scale, 1.0e-12))
    log_ssfr = -8.7 - 0.55 * np.log10(max(best.age_gyr, 0.03)) + 0.25 * best.logSFRratio0
    log_sfr = logmass + log_ssfr
    return {
        "logmass": float(logmass),
        "log_ssfr": float(log_ssfr),
        "dust2": float(best.dust2),
        "gas_logz": float(best.gas_logz),
        "logSFRratio0": float(best.logSFRratio0),
        "logSFR": float(log_sfr),
        "fit_chi2": float(best_chi2),
    }


def fit_sed_table(
    pixel_table: pd.DataFrame,
    output_path: str | Path | None = None,
    max_pixels: int | None = 40,
    redshift: float = 7.0,
) -> pd.DataFrame:
    """Fit a small public FSPS demonstration grid to a pixel photometry table.

    By default only a representative subset of valid pixels is fit so the public
    example stays light. Pass ``max_pixels=None`` to fit every kept pixel.
    """

    validate_geometry(pixel_table)
    bands = _flux_columns(pixel_table)
    selected = _selected_pixels(pixel_table, max_pixels=max_pixels)
    if selected.empty:
        raise ValueError("No valid pixels were selected for SED fitting.")

    grid = _build_fsps_grid(bands, redshift=redshift)
    rows = []
    for _, row in selected.iterrows():
        fitted = _fit_one_pixel(row, bands, grid)
        rows.append(
            {
                "ID": row["ID"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "pixel_index": int(row["pixel_index"]),
                "keep_pixel": row.get("keep_pixel", 1),
                **fitted,
            }
        )

    output = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
    return output


__all__ = ["FSPS_MESSAGE", "SedFittingDependencyError", "fit_sed_table"]
