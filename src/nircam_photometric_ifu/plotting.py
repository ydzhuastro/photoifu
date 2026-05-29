"""Matplotlib plotting utilities for the public example workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from .maps import infer_image_shape, reconstruct_map
from .stats import compare_region_to_global


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=float)
    lo, hi = np.nanpercentile(finite, [1, 99.5])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(channel, dtype=float)
    scaled = np.clip((channel - lo) / (hi - lo), 0, 1)
    scaled = np.arcsinh(4.0 * scaled) / np.arcsinh(4.0)
    return np.nan_to_num(scaled, nan=0.0)


def make_rgb_image(
    pixel_table: pd.DataFrame,
    image_shape: tuple[int, int] | None = None,
    bands: Sequence[str] = ("F444W", "F277W", "F150W"),
) -> np.ndarray:
    """Build a simple display RGB image from three NIRCam flux columns."""

    if image_shape is None:
        image_shape = infer_image_shape(pixel_table)
    missing = [band for band in bands if band not in pixel_table.columns]
    if missing:
        raise ValueError("Pixel table is missing RGB band column(s): " + ", ".join(missing))

    channels = [reconstruct_map(pixel_table, band, image_shape=image_shape) for band in bands]
    rgb = np.dstack([_stretch_channel(channel) for channel in channels])
    return np.clip(rgb, 0, 1)


def _copy_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="0.88")
    return cmap


def plot_resolved_maps(
    pixel_table: pd.DataFrame,
    property_maps: Mapping[str, np.ndarray],
    output_path: str | Path,
    image_shape: tuple[int, int] | None = None,
) -> Path:
    """Save a resolved image-plus-property map figure."""

    if image_shape is None:
        image_shape = infer_image_shape(pixel_table)
    output_path = _ensure_parent(output_path)

    ordered = [name for name in ("logmass", "log_ssfr", "dust2", "gas_logz", "logSFRratio0") if name in property_maps]
    fig, axes = plt.subplots(1, len(ordered) + 1, figsize=(3.1 * (len(ordered) + 1), 3.4), constrained_layout=True)

    axes[0].imshow(make_rgb_image(pixel_table, image_shape=image_shape), origin="lower")
    axes[0].set_title("synthetic NIRCam")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    cmaps = {
        "logmass": "magma",
        "log_ssfr": "viridis",
        "dust2": "cividis",
        "gas_logz": "coolwarm",
        "logSFRratio0": "plasma",
    }
    labels = {
        "logmass": "log mass",
        "log_ssfr": "log sSFR",
        "dust2": "dust2",
        "gas_logz": "gas log Z",
        "logSFRratio0": "recent SFH",
    }
    for ax, name in zip(axes[1:], ordered):
        image = np.ma.masked_invalid(property_maps[name])
        im = ax.imshow(image, origin="lower", cmap=_copy_cmap(cmaps.get(name, "viridis")))
        ax.set_title(labels.get(name, name))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_gmm_pca(
    pixel_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    output_path: str | Path,
    image_shape: tuple[int, int] | None = None,
    region_column: str = "region",
) -> Path:
    """Save the image, cluster map, and PCA diagnostic panels."""

    if image_shape is None:
        image_shape = infer_image_shape(pixel_table)
    output_path = _ensure_parent(output_path)
    cluster_map = reconstruct_map(cluster_table, "GMM_cluster", image_shape=image_shape)

    valid_cluster = np.isfinite(pd.to_numeric(cluster_table["GMM_cluster"], errors="coerce"))
    clusters = sorted(pd.to_numeric(cluster_table.loc[valid_cluster, "GMM_cluster"]).astype(int).unique())
    n_clusters = max(len(clusters), 1)
    cmap = plt.get_cmap("tab10", max(n_clusters, 3))
    bounds = np.arange(-0.5, n_clusters + 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(make_rgb_image(pixel_table, image_shape=image_shape), origin="lower")
    cvals = pd.to_numeric(cluster_table["GMM_cluster"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(cvals)
    ax.scatter(
        cluster_table.loc[finite, "x"],
        cluster_table.loc[finite, "y"],
        c=cvals[finite],
        cmap=cmap,
        norm=norm,
        s=7,
        alpha=0.52,
        linewidths=0,
    )
    ax.set_title("image with GMM overlay")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    im = ax.imshow(np.ma.masked_invalid(cluster_map), origin="lower", cmap=cmap, norm=norm)
    ax.set_title("GMM cluster map")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=range(n_clusters))
    cbar.set_label("cluster")

    ax = axes[1, 0]
    ax.scatter(
        cluster_table.loc[finite, "PCA1"],
        cluster_table.loc[finite, "PCA2"],
        c=cvals[finite],
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.8,
        linewidths=0,
    )
    ax.set_title("PCA colored by GMM")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    ax = axes[1, 1]
    ax.scatter(cluster_table["PCA1"], cluster_table["PCA2"], color="0.82", s=6, linewidths=0)
    if region_column in cluster_table.columns and (cluster_table[region_column] != "").any():
        region_labels = [label for label in cluster_table[region_column].dropna().unique() if label != ""]
        region_colors = plt.get_cmap("Set1", max(len(region_labels), 3))
        for idx, label in enumerate(region_labels):
            hit = cluster_table[region_column].astype(str).str.contains(str(label), regex=False)
            ax.scatter(
                cluster_table.loc[hit, "PCA1"],
                cluster_table.loc[hit, "PCA2"],
                s=16,
                color=region_colors(idx),
                label=label,
                linewidths=0,
                alpha=0.88,
            )
        ax.legend(frameon=False, fontsize=8, markerscale=1.6)
    ax.set_title("PCA with manual regions")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 1e-3:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def _find_class_a_members(region_classes: Mapping[str, Sequence[str]] | None) -> list[str]:
    if not region_classes:
        return []
    for class_name, members in region_classes.items():
        normalized = str(class_name).lower().replace("class", "").strip()
        if normalized == "a":
            return [str(member) for member in members]
    return []


def plot_region_comparisons(
    property_maps: Mapping[str, np.ndarray],
    region_masks: Mapping[str, np.ndarray],
    global_mask: np.ndarray,
    output_path: str | Path,
    region_classes: Mapping[str, Sequence[str]] | None = None,
    properties: Sequence[str] = ("logSFRratio0", "log_ssfr", "dust2", "gas_logz"),
) -> Path:
    """Save violin-style region distributions compared with the global host."""

    output_path = _ensure_parent(output_path)
    properties = [prop for prop in properties if prop in property_maps]
    if not properties:
        raise ValueError("None of the requested properties are present in property_maps.")

    region_names = list(region_masks.keys())
    class_a_members = _find_class_a_members(region_classes)
    class_a_mask = np.zeros_like(global_mask, dtype=bool)
    for member in class_a_members:
        if member in region_masks:
            class_a_mask |= region_masks[member]

    fig, axes = plt.subplots(1, len(properties), figsize=(3.4 * len(properties), 3.8), constrained_layout=True)
    if len(properties) == 1:
        axes = [axes]
    rng = np.random.default_rng(5)

    for ax, prop in zip(axes, properties):
        image = property_maps[prop]
        global_values = image[global_mask]
        global_values = global_values[np.isfinite(global_values)]
        global_median = np.nanmedian(global_values) if global_values.size else np.nan

        data = []
        positions = []
        for pos, name in enumerate(region_names, start=1):
            vals = image[region_masks[name]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                data.append(vals)
                positions.append(pos)

        if data:
            parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.78)
            for body in parts["bodies"]:
                body.set_facecolor("0.78")
                body.set_edgecolor("0.35")
                body.set_alpha(0.8)
            for key in ("cmins", "cmaxes", "cbars", "cmedians"):
                if key in parts:
                    parts[key].set_color("0.25")

        for pos, name in enumerate(region_names, start=1):
            vals = image[region_masks[name]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                jitter = rng.uniform(-0.08, 0.08, size=vals.size)
                ax.scatter(np.full(vals.size, pos) + jitter, vals, s=6, alpha=0.35, linewidths=0)

        if np.isfinite(global_median):
            ax.axhline(global_median, color="black", linestyle="--", linewidth=1.1, label="global median")

        if class_a_mask.any():
            stats = compare_region_to_global(image[class_a_mask], image[global_mask])
            ax.text(
                0.03,
                0.96,
                "Class A " + _format_pvalue(float(stats["p_value"])),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

        ax.set_title(prop)
        ax.set_xticks(range(1, len(region_names) + 1))
        ax.set_xticklabels(region_names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.22)

    axes[0].set_ylabel("property value")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
