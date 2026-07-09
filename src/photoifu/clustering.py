"""Feature-space PCA and Gaussian-mixture clustering for fitted pixels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ("logmass", "log_ssfr", "dust2", "gas_logz", "logSFRratio0")
SPATIAL_COLUMNS = {"x", "y", "pixel_index", "ID", "id", "ra", "dec"}


def _with_derived_log_ssfr(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    if "log_ssfr" in output.columns:
        return output
    if "logsSFR" in output.columns:
        output["log_ssfr"] = output["logsSFR"]
    elif {"logSFR", "logmass"}.issubset(output.columns):
        output["log_ssfr"] = output["logSFR"] - output["logmass"]
    return output


def _check_features(features: Sequence[str]) -> list[str]:
    features = list(features)
    forbidden = [feature for feature in features if feature in SPATIAL_COLUMNS]
    if forbidden:
        raise ValueError(
            "Spatial or identifier columns cannot be used for clustering: "
            + ", ".join(forbidden)
            + ". Use physical SED-derived quantities only."
        )
    if len(features) < 2:
        raise ValueError("At least two non-spatial features are required for PCA/GMM clustering.")
    return features


def build_feature_matrix(sed_table: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Return a finite feature matrix indexed to the input SED table rows."""

    features = _check_features(features)
    table = _with_derived_log_ssfr(sed_table)
    missing = [feature for feature in features if feature not in table.columns]
    if missing:
        raise ValueError(
            "SED table is missing requested clustering feature(s): " + ", ".join(missing)
        )

    feature_matrix = table.loc[:, features].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(feature_matrix.to_numpy(dtype=float)).all(axis=1)
    feature_matrix = feature_matrix.loc[finite].astype(float)
    if feature_matrix.empty:
        raise ValueError("No rows have finite values for all requested clustering features.")
    return feature_matrix


def _select_k_by_bic(
    scaled_features: np.ndarray,
    k_range: Sequence[int],
    random_state: int,
    covariance_type: str,
):
    from sklearn.mixture import GaussianMixture

    best_model = None
    best_bic = np.inf
    best_k = None
    bic_values: dict[int, float] = {}
    for candidate_k in k_range:
        if candidate_k < 1:
            continue
        model = GaussianMixture(
            n_components=int(candidate_k),
            covariance_type=covariance_type,
            random_state=random_state,
        )
        model.fit(scaled_features)
        bic = float(model.bic(scaled_features))
        bic_values[int(candidate_k)] = bic
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_k = int(candidate_k)
    if best_model is None:
        raise ValueError("k_range did not contain any positive candidate k values.")
    return best_k, best_model, bic_values


def _reorder_labels(
    labels: np.ndarray,
    table: pd.DataFrame,
    row_index: pd.Index,
) -> tuple[np.ndarray, dict[int, int]]:
    order_col = "logSFRratio0" if "logSFRratio0" in table.columns else None
    if order_col is None and "log_ssfr" in table.columns:
        order_col = "log_ssfr"
    if order_col is None:
        ordered = sorted(np.unique(labels))
    else:
        tmp = pd.DataFrame(
            {
                "cluster": labels,
                "order_value": pd.to_numeric(table.loc[row_index, order_col], errors="coerce"),
            },
            index=row_index,
        )
        medians = tmp.groupby("cluster")["order_value"].median().sort_values()
        ordered = list(medians.index)

    mapping = {int(old_label): int(new_label) for new_label, old_label in enumerate(ordered)}
    return np.array([mapping[int(label)] for label in labels], dtype=int), mapping


def run_gmm_pca(
    sed_table: pd.DataFrame,
    features: Sequence[str] | None = None,
    k: int | str = 6,
    auto_k: bool = False,
    k_range: Sequence[int] = tuple(range(2, 9)),
    random_state: int = 7,
    covariance_type: str = "full",
) -> pd.DataFrame:
    """Run robust scaling, PCA projection, and GMM clustering without spatial inputs."""

    try:
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import RobustScaler
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs.
        raise ImportError(
            "run_gmm_pca requires scikit-learn. Install with `pip install scikit-learn`."
        ) from exc

    if features is None:
        features = DEFAULT_FEATURES
    table = _with_derived_log_ssfr(sed_table)
    feature_matrix = build_feature_matrix(table, features)
    row_index = feature_matrix.index

    scaler = RobustScaler()
    scaled = scaler.fit_transform(feature_matrix.to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=random_state)
    pca_xy = pca.fit_transform(scaled)

    if auto_k or k == "auto":
        selected_k, model, bic_values = _select_k_by_bic(
            scaled,
            k_range=k_range,
            random_state=random_state,
            covariance_type=covariance_type,
        )
    else:
        selected_k = int(k)
        if selected_k < 1:
            raise ValueError("k must be a positive integer or 'auto'.")
        model = GaussianMixture(
            n_components=selected_k,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        model.fit(scaled)
        bic_values = {selected_k: float(model.bic(scaled))}

    labels = model.predict(scaled)
    probabilities = model.predict_proba(scaled).max(axis=1)
    labels, label_mapping = _reorder_labels(labels, table, row_index)

    output = table.copy()
    output["PCA1"] = np.nan
    output["PCA2"] = np.nan
    output["GMM_cluster"] = pd.Series(pd.array([pd.NA] * len(output), dtype="Int64"), index=output.index)
    output["cluster_probability"] = np.nan
    output.loc[row_index, "PCA1"] = pca_xy[:, 0]
    output.loc[row_index, "PCA2"] = pca_xy[:, 1]
    output.loc[row_index, "GMM_cluster"] = pd.array(labels, dtype="Int64")
    output.loc[row_index, "cluster_probability"] = probabilities

    output.attrs["features"] = list(features)
    output.attrs["gmm_k"] = selected_k
    output.attrs["gmm_bic"] = bic_values
    output.attrs["cluster_label_mapping"] = label_mapping
    output.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return output
