from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .ml_pipeline import (
    DEFAULT_FEATURE_COLUMNS,
    _append_all_row,
    _apply_calibration,
    _build_calibration_factors,
    _build_true_valuation_targets,
    _estimate_ibnr_addon,
    _stack_training_snapshots,
    build_claim_snapshot_dataset,
    summarize_ml_compare,
)

@dataclass
class _NNTrajectoryTwoPartOffsetModel:
    """Trajectory-aware two-part model with residual severity around an offset prior."""

    prob_scaler: object | None
    prob_model: object | None
    prob_const: float
    sev_scaler: object | None
    sev_model: object | None
    sev_const: float
    sev_residual_clip: float
    sev_pred_cap: float
    prior_state: dict | None
    use_log_prior_feature: bool


def _augment_features_with_dev_age(d: pd.DataFrame, feat_cols: list[str]) -> Tuple[pd.DataFrame, list[str]]:
    """Append engineered development-age features for DL models."""
    out = d.copy()
    if "AY" in out.columns and "valuation_year" in out.columns:
        dev_age = pd.to_numeric(out["valuation_year"], errors="coerce") - pd.to_numeric(out["AY"], errors="coerce")
        dev_age = dev_age.fillna(0.0).clip(lower=0.0)
    else:
        dev_age = pd.Series(0.0, index=out.index)

    out["__dev_age"] = dev_age.astype(float)
    out["__dev_age_sq"] = np.square(out["__dev_age"])
    out["__dev_age_inv"] = 1.0 / (1.0 + out["__dev_age"])
    out["__dev_age_log1p"] = np.log1p(out["__dev_age"])
    out["__maturity_band"] = np.select(
        [
            out["__dev_age"] <= 2,
            out["__dev_age"] <= 5,
        ],
        [
            0.0,   # immature
            1.0,   # middle
        ],
        default=2.0,  # mature
    ).astype(float)

    feat_cols_aug = [c for c in feat_cols if c in out.columns]
    for c in ["__dev_age", "__dev_age_sq", "__dev_age_inv", "__dev_age_log1p", "__maturity_band"]:
        if c not in feat_cols_aug:
            feat_cols_aug.append(c)
    return out, feat_cols_aug


def _collect_trajectory_feature_columns(d: pd.DataFrame) -> list[str]:
    return sorted([c for c in d.columns if str(c).startswith("__traj_")])


def _extract_trajectory_features_for_valuation(
    raw_output_df: pd.DataFrame,
    valuation_year: int,
    scale_divisor: float,
    n_lags: int = 6,
) -> pd.DataFrame:
    """Build valuation-safe lagged payment/open-path features from raw yearly simulation output."""
    raw = raw_output_df.copy()
    if not {"Id", "AY"}.issubset(raw.columns):
        return pd.DataFrame(columns=["Id"])

    raw["Id"] = pd.to_numeric(raw["Id"], errors="coerce").astype("Int64")
    raw["AY"] = pd.to_numeric(raw["AY"], errors="coerce")

    pay_cols = sorted([c for c in raw.columns if str(c).startswith("Pay")], key=lambda x: int(str(x)[3:]))
    open_cols = sorted([c for c in raw.columns if str(c).startswith("Open")], key=lambda x: int(str(x)[4:]))
    if not pay_cols:
        return pd.DataFrame(columns=["Id"])

    n_dev = len(pay_cols)
    for c in pay_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    pay_mat = raw[pay_cols].to_numpy(dtype=float)

    if open_cols:
        for c in open_cols:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
        open_mat = raw[open_cols].to_numpy(dtype=float)
    else:
        open_mat = np.zeros_like(pay_mat)

    ay = pd.to_numeric(raw["AY"], errors="coerce").to_numpy(dtype=float)
    dev = np.maximum(np.minimum(int(valuation_year) - ay, n_dev - 1), 0).astype(int)
    row_idx = np.arange(len(raw))
    col_idx = np.arange(n_dev)[None, :]
    observed_mask = col_idx <= dev[:, None]

    observed_pay = np.where(observed_mask, pay_mat, 0.0)
    observed_open = np.where(observed_mask, open_mat, 0.0)
    paid_to_val = observed_pay.sum(axis=1)
    eps = 1e-9

    data: dict[str, object] = {"Id": raw["Id"]}
    recent_sum = np.zeros(len(raw), dtype=float)
    older_sum = np.zeros(len(raw), dtype=float)

    for lag in range(int(max(n_lags, 1))):
        idx = dev - lag
        valid = idx >= 0
        idx_safe = np.where(valid, idx, 0)

        pay_lag = np.where(valid, pay_mat[row_idx, idx_safe], 0.0)
        open_lag = np.where(valid, open_mat[row_idx, idx_safe], 0.0)
        share_lag = np.where(paid_to_val > eps, pay_lag / np.maximum(paid_to_val, eps), 0.0)
        pay_ind_lag = (pay_lag > 0.0).astype(float)

        data[f"__traj_pay_lag_{lag}"] = pay_lag / float(scale_divisor)
        data[f"__traj_pay_share_lag_{lag}"] = share_lag
        data[f"__traj_pay_ind_lag_{lag}"] = pay_ind_lag
        data[f"__traj_open_lag_{lag}"] = open_lag

        if lag < 3:
            recent_sum += pay_lag
        elif lag < 6:
            older_sum += pay_lag

    pos_mask = observed_pay > 0.0
    first_pos_idx = np.where(pos_mask, col_idx, n_dev).min(axis=1)
    has_pos = pos_mask.any(axis=1)
    first_pos_idx = np.where(has_pos, first_pos_idx, -1)
    years_since_first_pay = np.where(has_pos, dev - first_pos_idx, dev + 1).astype(float)

    data["__traj_recent_pay_sum"] = recent_sum / float(scale_divisor)
    data["__traj_older_pay_sum"] = older_sum / float(scale_divisor)
    data["__traj_recent_to_older"] = np.where(older_sum > eps, recent_sum / np.maximum(older_sum, eps), 0.0)
    data["__traj_recent_open_sum"] = (
        pd.Series(data.get("__traj_open_lag_0", 0.0), index=raw.index, dtype=float)
        + pd.Series(data.get("__traj_open_lag_1", 0.0), index=raw.index, dtype=float)
        + pd.Series(data.get("__traj_open_lag_2", 0.0), index=raw.index, dtype=float)
    )
    data["__traj_years_since_first_pay"] = years_since_first_pay

    if int(max(n_lags, 1)) >= 4:
        data["__traj_pay_trend_2v2"] = (
            pd.Series(data.get("__traj_pay_lag_0", 0.0), index=raw.index, dtype=float)
            + pd.Series(data.get("__traj_pay_lag_1", 0.0), index=raw.index, dtype=float)
            - pd.Series(data.get("__traj_pay_lag_2", 0.0), index=raw.index, dtype=float)
            - pd.Series(data.get("__traj_pay_lag_3", 0.0), index=raw.index, dtype=float)
        )
    else:
        data["__traj_pay_trend_2v2"] = np.zeros(len(raw), dtype=float)

    return pd.DataFrame(data)


def _augment_with_trajectory_features_multi_snapshot(
    d: pd.DataFrame,
    raw_output_df: pd.DataFrame | None,
    scale_divisor: float,
    n_lags: int = 6,
) -> pd.DataFrame:
    """Attach trajectory features for each valuation-year snapshot in d."""
    if raw_output_df is None or d.empty or "Id" not in d.columns or "valuation_year" not in d.columns:
        return d.copy()

    frames = []
    for val in sorted(pd.to_numeric(d["valuation_year"], errors="coerce").dropna().astype(int).unique().tolist()):
        part = d.loc[pd.to_numeric(d["valuation_year"], errors="coerce").astype("Int64") == int(val)].copy()
        if part.empty:
            continue
        traj = _extract_trajectory_features_for_valuation(
            raw_output_df=raw_output_df,
            valuation_year=int(val),
            scale_divisor=float(scale_divisor),
            n_lags=int(n_lags),
        )
        part = part.merge(traj, on="Id", how="left")
        frames.append(part)

    if not frames:
        return d.copy()

    out = pd.concat(frames, ignore_index=True)
    for c in _collect_trajectory_feature_columns(out):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _build_monotone_positive_offset_prior(d_pos: pd.DataFrame) -> dict:
    """Create age/type positive-severity priors with monotone age profiles and credibility fallback."""
    if d_pos.empty or "y" not in d_pos.columns:
        return {"global": {}, "type": {}, "global_floor": 0.0, "type_floor": {}}

    tmp = d_pos.loc[:, ["Type", "AY", "valuation_year", "y", "paid_to_val_truth"]].copy()
    tmp["Type"] = pd.to_numeric(tmp["Type"], errors="coerce").astype(int)
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp["paid_to_val_truth"] = pd.to_numeric(tmp["paid_to_val_truth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp["dev_age"] = (
        pd.to_numeric(tmp["valuation_year"], errors="coerce") - pd.to_numeric(tmp["AY"], errors="coerce")
    ).fillna(0.0).clip(lower=0.0).astype(int)
    tmp = tmp.loc[tmp["y"] > 1e-9].copy()
    if tmp.empty:
        return {"global": {}, "type": {}, "global_floor": 0.0, "type_floor": {}}

    eps = 1.0
    tmp["ratio"] = tmp["y"] / (tmp["paid_to_val_truth"] + eps)

    def _mono_table(df_in: pd.DataFrame) -> dict[int, dict[str, float]]:
        if df_in.empty:
            return {}
        grp = (
            df_in.groupby("dev_age", as_index=False)
            .agg(
                n=("y", "size"),
                mean_y=("y", "mean"),
                mean_ratio=("ratio", "mean"),
            )
            .sort_values("dev_age")
            .reset_index(drop=True)
        )
        grp["mean_y"] = np.minimum.accumulate(grp["mean_y"].to_numpy(dtype=float))
        grp["mean_ratio"] = np.minimum.accumulate(grp["mean_ratio"].to_numpy(dtype=float))
        return {
            int(row.dev_age): {
                "n": float(row.n),
                "mean_y": float(max(row.mean_y, 0.0)),
                "mean_ratio": float(max(row.mean_ratio, 0.0)),
            }
            for row in grp.itertuples(index=False)
        }

    global_tbl = _mono_table(tmp)
    type_tbl = {int(typ): _mono_table(grp.copy()) for typ, grp in tmp.groupby("Type")}
    type_floor = {int(typ): float(grp["y"].mean()) for typ, grp in tmp.groupby("Type")}
    global_floor = float(tmp["y"].mean())
    return {
        "global": global_tbl,
        "type": type_tbl,
        "global_floor": global_floor,
        "type_floor": type_floor,
    }


def _lookup_monotone_prior_row(table: dict[int, dict[str, float]], age: int) -> dict[str, float]:
    if not table:
        return {"n": 0.0, "mean_y": 0.0, "mean_ratio": 0.0}
    ages = sorted(table.keys())
    age_use = min(max(int(age), ages[0]), ages[-1])
    return table.get(age_use, table[ages[-1]])


def _predict_positive_offset_prior(
    prior_state: dict | None,
    d_local: pd.DataFrame,
    credibility_n: float = 150.0,
    mean_weight: float = 0.55,
) -> np.ndarray:
    if prior_state is None or d_local.empty:
        return np.zeros(len(d_local), dtype=float)

    global_tbl = prior_state.get("global", {})
    type_tbl = prior_state.get("type", {})
    global_floor = float(prior_state.get("global_floor", 0.0))
    type_floor = prior_state.get("type_floor", {})

    dev_age = (
        pd.to_numeric(d_local["valuation_year"], errors="coerce") - pd.to_numeric(d_local["AY"], errors="coerce")
    ).fillna(0.0).clip(lower=0.0).astype(int)
    typ_arr = pd.to_numeric(d_local["Type"], errors="coerce").fillna(-1).astype(int)
    paid_obs = pd.to_numeric(d_local["paid_to_val_truth"], errors="coerce").fillna(0.0).clip(lower=0.0)

    out = np.zeros(len(d_local), dtype=float)
    for i, (age, typ, paid) in enumerate(zip(dev_age.tolist(), typ_arr.tolist(), paid_obs.tolist())):
        g = _lookup_monotone_prior_row(global_tbl, age)
        t = _lookup_monotone_prior_row(type_tbl.get(int(typ), {}), age)
        n_type = float(t.get("n", 0.0))
        w = min(max(n_type / float(max(credibility_n, 1.0)), 0.0), 1.0)
        mean_y = w * float(t.get("mean_y", 0.0)) + (1.0 - w) * float(g.get("mean_y", 0.0))
        mean_ratio = w * float(t.get("mean_ratio", 0.0)) + (1.0 - w) * float(g.get("mean_ratio", 0.0))
        floor_val = w * float(type_floor.get(int(typ), global_floor)) + (1.0 - w) * float(global_floor)
        prior_val = float(mean_weight) * mean_y + (1.0 - float(mean_weight)) * (mean_ratio * float(paid))
        out[i] = max(prior_val, 0.15 * max(floor_val, 0.0))
    return np.maximum(out, 0.0)


def _build_monotone_age_caps(d_train: pd.DataFrame, quantile: float = 0.95) -> dict[int, float]:
    """Build non-increasing reserve-ratio caps by development age."""
    if d_train.empty or "paid_to_val_truth" not in d_train.columns:
        return {}
    if "AY" not in d_train.columns or "valuation_year" not in d_train.columns:
        return {}

    tmp = d_train.loc[:, ["AY", "valuation_year", "y", "paid_to_val_truth"]].copy()
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp["paid_to_val_truth"] = pd.to_numeric(tmp["paid_to_val_truth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp["dev_age"] = (pd.to_numeric(tmp["valuation_year"], errors="coerce") - pd.to_numeric(tmp["AY"], errors="coerce")).fillna(0.0)
    tmp["dev_age"] = tmp["dev_age"].clip(lower=0.0).astype(int)

    eps = 1e-6
    tmp["ratio"] = tmp["y"] / (tmp["paid_to_val_truth"] + eps)
    grp = tmp.groupby("dev_age", as_index=False).agg(cap=("ratio", lambda s: float(np.nanquantile(s, quantile))))
    if grp.empty:
        return {}

    grp = grp.sort_values("dev_age").reset_index(drop=True)
    raw_caps = grp["cap"].to_numpy(dtype=float)
    mono_caps = np.minimum.accumulate(raw_caps)
    mono_caps = np.clip(mono_caps, 0.0, np.nanmax(raw_caps) if len(raw_caps) else 0.0)

    return {int(age): float(cap) for age, cap in zip(grp["dev_age"].tolist(), mono_caps.tolist())}


def _apply_monotone_age_caps(preds: np.ndarray, d_pred: pd.DataFrame, age_caps: dict[int, float]) -> np.ndarray:
    """Cap predicted reserve by non-increasing age ratio profile."""
    out = np.maximum(np.asarray(preds, dtype=float), 0.0)
    if len(out) == 0 or not age_caps:
        return out
    if "AY" not in d_pred.columns or "valuation_year" not in d_pred.columns or "paid_to_val_truth" not in d_pred.columns:
        return out

    dev_age = (
        pd.to_numeric(d_pred["valuation_year"], errors="coerce")
        - pd.to_numeric(d_pred["AY"], errors="coerce")
    ).fillna(0.0).clip(lower=0.0).astype(int)
    paid_obs = pd.to_numeric(d_pred["paid_to_val_truth"], errors="coerce").fillna(0.0).clip(lower=0.0)

    known_ages = sorted(age_caps.keys())
    if not known_ages:
        return out
    min_age, max_age = known_ages[0], known_ages[-1]

    for i, age in enumerate(dev_age.tolist()):
        age_use = min(max(int(age), min_age), max_age)
        cap_ratio = float(age_caps.get(age_use, age_caps[max_age]))
        cap_val = float(max(cap_ratio, 0.0) * float(paid_obs.iloc[i]))
        out[i] = min(out[i], cap_val)
    return np.maximum(out, 0.0)


def fit_trajectory_two_part_offset_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    d_train: pd.DataFrame,
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
    alpha: float = 3e-4,
    learning_rate_init: float = 8e-4,
    max_iter: int = 260,
    random_state: int = 42,
    residual_clip: float = 2.5,
    pred_cap_quantile: float = 0.995,
    offset_credibility_n: float = 150.0,
    offset_mean_weight: float = 0.55,
    use_log_prior_feature: bool = True,
) -> _NNTrajectoryTwoPartOffsetModel:
    """Trajectory-aware two-part model: payment-occurrence head + severity residual around a monotone offset prior."""
    try:
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for the DL pipeline. Install with: pip install scikit-learn"
        ) from exc

    X_arr = np.asarray(X_train, dtype=float)
    y = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(y) == 0:
        return _NNTrajectoryTwoPartOffsetModel(
            prob_scaler=None,
            prob_model=None,
            prob_const=0.0,
            sev_scaler=None,
            sev_model=None,
            sev_const=0.0,
            sev_residual_clip=float(residual_clip),
            sev_pred_cap=0.0,
            prior_state=None,
            use_log_prior_feature=bool(use_log_prior_feature),
        )

    y_pos = (y > 1e-9).astype(int)
    pos_rate = float(np.mean(y_pos)) if len(y_pos) else 0.0

    if np.all(y_pos == y_pos[0]):
        prob_scaler = None
        prob_model = None
    else:
        min_class_n = int(min(np.bincount(y_pos))) if len(np.unique(y_pos)) >= 2 else 0
        use_prob_early_stopping = bool(len(y_pos) >= 50 and min_class_n >= 5)
        prob_scaler = StandardScaler()
        X_prob = prob_scaler.fit_transform(X_arr)
        prob_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            early_stopping=use_prob_early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=int(random_state),
        )
        prob_model.fit(X_prob, y_pos)

    pos_mask = y_pos.astype(bool)
    sev_const = float(np.mean(y[pos_mask])) if np.any(pos_mask) else 0.0
    if int(pos_mask.sum()) < 30:
        return _NNTrajectoryTwoPartOffsetModel(
            prob_scaler=prob_scaler,
            prob_model=prob_model,
            prob_const=pos_rate,
            sev_scaler=None,
            sev_model=None,
            sev_const=sev_const,
            sev_residual_clip=float(residual_clip),
            sev_pred_cap=max(sev_const, 0.0),
            prior_state=None,
            use_log_prior_feature=bool(use_log_prior_feature),
        )

    d_pos = d_train.iloc[np.flatnonzero(pos_mask)].copy()
    prior_state = _build_monotone_positive_offset_prior(d_pos)
    prior_pos = _predict_positive_offset_prior(
        prior_state,
        d_pos,
        credibility_n=float(offset_credibility_n),
        mean_weight=float(offset_mean_weight),
    )

    sev_X = X_train.iloc[np.flatnonzero(pos_mask)].copy()
    if bool(use_log_prior_feature):
        sev_X["__log_prior_offset"] = np.log1p(prior_pos)

    eps = 1e-9
    residual_target = np.log((y[pos_mask] + eps) / (prior_pos + eps))
    residual_target = np.clip(residual_target, -float(residual_clip), float(residual_clip))

    use_sev_early_stopping = bool(int(pos_mask.sum()) >= 50)
    sev_scaler = StandardScaler()
    X_sev = sev_scaler.fit_transform(np.asarray(sev_X, dtype=float))
    sev_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        learning_rate_init=float(learning_rate_init),
        max_iter=int(max_iter),
        early_stopping=use_sev_early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=int(random_state),
    )
    sev_model.fit(X_sev, residual_target)

    raw_pred_train = np.maximum(
        prior_pos * np.exp(np.clip(sev_model.predict(X_sev), -float(residual_clip), float(residual_clip))),
        0.0,
    )
    pred_cap = float(np.nanquantile(raw_pred_train, float(pred_cap_quantile)) * 2.0)
    pred_cap = max(pred_cap, float(np.nanmax(raw_pred_train)) if len(raw_pred_train) > 0 else 0.0)

    return _NNTrajectoryTwoPartOffsetModel(
        prob_scaler=prob_scaler,
        prob_model=prob_model,
        prob_const=pos_rate,
        sev_scaler=sev_scaler,
        sev_model=sev_model,
        sev_const=sev_const,
        sev_residual_clip=float(residual_clip),
        sev_pred_cap=float(pred_cap),
        prior_state=prior_state,
        use_log_prior_feature=bool(use_log_prior_feature),
    )


def predict_trajectory_two_part_offset_model(
    model: _NNTrajectoryTwoPartOffsetModel,
    X: pd.DataFrame,
    d_local: pd.DataFrame,
    offset_credibility_n: float = 150.0,
    offset_mean_weight: float = 0.55,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if len(X_arr) == 0:
        return np.zeros(0, dtype=float)

    if model.prob_model is None or model.prob_scaler is None:
        p_open = np.full(len(X_arr), float(np.clip(model.prob_const, 0.0, 1.0)), dtype=float)
    else:
        X_prob = model.prob_scaler.transform(X_arr)
        p_open = np.clip(model.prob_model.predict_proba(X_prob)[:, 1], 0.0, 1.0)

    if model.sev_model is None or model.sev_scaler is None or model.prior_state is None:
        sev = np.full(len(X_arr), max(float(model.sev_const), 0.0), dtype=float)
    else:
        prior = _predict_positive_offset_prior(
            model.prior_state,
            d_local,
            credibility_n=float(offset_credibility_n),
            mean_weight=float(offset_mean_weight),
        )
        sev_X = X.copy()
        if bool(model.use_log_prior_feature):
            sev_X["__log_prior_offset"] = np.log1p(prior)
        X_sev = model.sev_scaler.transform(np.asarray(sev_X, dtype=float))
        sev_resid = np.clip(
            model.sev_model.predict(X_sev),
            -float(model.sev_residual_clip),
            float(model.sev_residual_clip),
        )
        sev = np.maximum(prior * np.exp(sev_resid), 0.0)
        if np.isfinite(model.sev_pred_cap) and float(model.sev_pred_cap) > 0.0:
            sev = np.minimum(sev, float(model.sev_pred_cap))

    return np.maximum(p_open * sev, 0.0)


def _fit_predict_with_credibility_blend_nn(
    d_train: pd.DataFrame,
    d_pred: pd.DataFrame,
    feat_cols: list[str],
    min_rows_type_model: int,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    random_state: int,
    residual_clip: float,
    pred_cap_quantile: float,
    model_variant: str,
    add_age_features: bool,
    monotone_postprocess: bool,
    raw_output_df: Optional[pd.DataFrame] = None,
    scale_divisor: float = 1000.0,
    trajectory_n_lags: int = 6,
) -> np.ndarray:
    if str(model_variant) != "trajectory_two_part_offset":
        raise ValueError(
            "Only model_variant='trajectory_two_part_offset' is supported in the streamlined DL pipeline."
        )

    d_train_work = d_train.copy()
    d_pred_work = d_pred.copy()

    d_train_work = _augment_with_trajectory_features_multi_snapshot(
        d_train_work,
        raw_output_df=raw_output_df,
        scale_divisor=float(scale_divisor),
        n_lags=int(trajectory_n_lags),
    )
    d_pred_work = _augment_with_trajectory_features_multi_snapshot(
        d_pred_work,
        raw_output_df=raw_output_df,
        scale_divisor=float(scale_divisor),
        n_lags=int(trajectory_n_lags),
    )

    feat_cols_work = [c for c in feat_cols if c in d_train_work.columns and c in d_pred_work.columns]
    for c in _collect_trajectory_feature_columns(d_train_work):
        if c in d_pred_work.columns and c not in feat_cols_work:
            feat_cols_work.append(c)
    if add_age_features:
        d_train_work, feat_cols_work = _augment_features_with_dev_age(d_train_work, feat_cols_work)
        d_pred_work, feat_cols_work = _augment_features_with_dev_age(d_pred_work, feat_cols_work)

    if not feat_cols_work:
        return np.zeros(len(d_pred_work), dtype=float)

    def _fit_model_local(d_local: pd.DataFrame):
        return fit_trajectory_two_part_offset_model(
            d_local[feat_cols_work],
            d_local["y"].astype(float),
            d_train=d_local,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=float(alpha),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            random_state=int(random_state),
            residual_clip=float(residual_clip),
            pred_cap_quantile=float(pred_cap_quantile),
        )

    def _predict_model_local(model_local, d_local: pd.DataFrame) -> np.ndarray:
        return predict_trajectory_two_part_offset_model(
            model_local,
            d_local[feat_cols_work],
            d_local=d_local,
        )

    pooled_model = _fit_model_local(d_train_work)
    pooled_pred = _predict_model_local(pooled_model, d_pred_work)
    final_pred = pooled_pred.copy()

    for typ in sorted(d_pred_work["Type"].dropna().astype(int).unique().tolist()):
        pred_mask = d_pred_work["Type"].eq(typ)
        train_mask = d_train_work["Type"].eq(typ)
        n_type = int(train_mask.sum())
        if n_type < int(min_rows_type_model):
            continue

        type_model = _fit_model_local(d_train_work.loc[train_mask].copy())
        type_pred = _predict_model_local(type_model, d_pred_work.loc[pred_mask].copy())
        alpha_type = min(max((n_type - min_rows_type_model) / float(max(min_rows_type_model, 1)), 0.0), 1.0)
        alpha_type = 0.20 + 0.40 * alpha_type
        final_pred[pred_mask.to_numpy()] = alpha_type * type_pred + (1.0 - alpha_type) * pooled_pred[pred_mask.to_numpy()]

    if monotone_postprocess:
        age_caps = _build_monotone_age_caps(d_train_work)
        final_pred = _apply_monotone_age_caps(final_pred, d_pred_work, age_caps)

    return np.maximum(final_pred, 0.0)


def run_claim_level_dl_rolling_origin_backtest(
    full_paid_df: pd.DataFrame,
    full_claims_df: pd.DataFrame,
    valuation_years: Iterable[int],
    train_snapshot_years: Optional[Iterable[int]] = None,
    scale_divisor: float = 1000.0,
    feature_cols: Optional[list[str]] = None,
    method_name_rbns: str = "DL-Claim-NN-RBNS-Cal",
    method_name_hybrid: str = "DL-Claim-NN+IBNR-Cal",
    min_train_rows: int = 300,
    raw_output_df: Optional[pd.DataFrame] = None,
    lookback_valuations: int = 4,
    max_train_rows: int = 220_000,
    min_rows_type_model: int = 2_000,
    eval_recent_ay_window: Optional[int] = 3,
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    max_iter: int = 240,
    random_state: int = 42,
    residual_clip: float = 2.5,
    pred_cap_quantile: float = 0.995,
    model_variant: str = "trajectory_two_part_offset",
    add_age_features: bool = False,
    monotone_postprocess: bool = False,
    calibration_min_mix: float = 0.35,
    ibnr_calibration_min_mix: float = 0.35,
    trajectory_n_lags: int = 6,
    n_jobs_valuation: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling-origin DL pipeline on the same target slice as the ML benchmark."""
    used_features = feature_cols if feature_cols is not None else list(DEFAULT_FEATURE_COLUMNS)
    valuation_years = [int(v) for v in valuation_years]
    extra_train_years = [int(y) for y in (train_snapshot_years or [])]
    training_pool = sorted(set(valuation_years) | set(extra_train_years))
    if str(model_variant) != "trajectory_two_part_offset":
        raise ValueError(
            "Only model_variant='trajectory_two_part_offset' is supported in the streamlined DL pipeline."
        )

    rbns_compare_rows = []
    rbns_summary_rows = []
    hybrid_compare_rows = []
    hybrid_summary_rows = []

    def _process_single_valuation(v: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        d_test_full = build_claim_snapshot_dataset(
            full_paid_df=full_paid_df,
            full_claims_df=full_claims_df,
            valuation_year=int(v),
            scale_divisor=scale_divisor,
            include_unreported=False,
            raw_output_df=raw_output_df,
        )
        if d_test_full.empty:
            return None

        train_cutoff = int(v) - 3
        if eval_recent_ay_window is None:
            d_test = d_test_full.copy()
        else:
            d_test = d_test_full.loc[d_test_full["AY"] > int(v) - int(eval_recent_ay_window)].copy()
        if d_test.empty:
            return None

        d_train = _stack_training_snapshots(
            full_paid_df=full_paid_df,
            full_claims_df=full_claims_df,
            valuation_years=training_pool,
            current_valuation=int(v),
            train_cutoff=int(train_cutoff),
            scale_divisor=scale_divisor,
            raw_output_df=raw_output_df,
            lookback_valuations=int(lookback_valuations),
            max_train_rows=int(max_train_rows),
        )
        if d_train.empty or len(d_train) < int(min_train_rows):
            return None

        calib_val = int(d_train["valuation_year"].max()) if "valuation_year" in d_train.columns else None
        if calib_val is not None and (d_train["valuation_year"] < calib_val).sum() >= int(min_train_rows):
            d_fit = d_train.loc[d_train["valuation_year"] < calib_val].copy()
            d_calib = d_train.loc[d_train["valuation_year"] == calib_val].copy()
        else:
            d_fit = d_train.copy()
            d_calib = pd.DataFrame()

        feat_cols = [c for c in used_features if c in d_fit.columns and c in d_test.columns]
        if not feat_cols:
            return None

        raw_test_pred = _fit_predict_with_credibility_blend_nn(
            d_train=d_fit,
            d_pred=d_test,
            feat_cols=feat_cols,
            min_rows_type_model=int(min_rows_type_model),
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=float(alpha),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            random_state=int(random_state),
            residual_clip=float(residual_clip),
            pred_cap_quantile=float(pred_cap_quantile),
            model_variant=str(model_variant),
            add_age_features=bool(add_age_features),
            monotone_postprocess=bool(monotone_postprocess),
            raw_output_df=raw_output_df,
            scale_divisor=float(scale_divisor),
            trajectory_n_lags=int(trajectory_n_lags),
        )

        if not d_calib.empty:
            raw_calib_pred = _fit_predict_with_credibility_blend_nn(
                d_train=d_fit,
                d_pred=d_calib,
                feat_cols=feat_cols,
                min_rows_type_model=int(min_rows_type_model),
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=float(alpha),
                learning_rate_init=float(learning_rate_init),
                max_iter=int(max_iter),
                random_state=int(random_state),
                residual_clip=float(residual_clip),
                pred_cap_quantile=float(pred_cap_quantile),
                model_variant=str(model_variant),
                add_age_features=bool(add_age_features),
                monotone_postprocess=bool(monotone_postprocess),
            )
            calib_eval = d_calib.loc[:, ["Type", "y"]].copy()
            calib_eval["pred_rbns"] = raw_calib_pred
            g_fac, t_fac = _build_calibration_factors(calib_eval, pred_col="pred_rbns", true_col="y", min_rows_type=500)
            preds = _apply_calibration(
                raw_test_pred,
                d_test,
                global_factor=g_fac,
                type_factors=t_fac,
                min_mix=float(calibration_min_mix),
            )
        else:
            preds = raw_test_pred

        d_eval = d_test.loc[:, ["Id", "AY", "Type", "lob", "y", "paid_to_val_truth"]].copy()
        d_eval["pred_rbns"] = preds
        d_eval["latest_observed_claim"] = d_eval["paid_to_val_truth"]
        d_eval["valuation_pay_year"] = int(v)

        rbns_cmp = (
            d_eval.groupby(["AY", "Type", "lob"], as_index=False)
            .agg(
                latest_observed=("latest_observed_claim", "sum"),
                pred_rbns=("pred_rbns", "sum"),
                true_rbns=("y", "sum"),
            )
        )
        rbns_cmp["ultimate"] = rbns_cmp["pred_rbns"]
        rbns_cmp["true_ultimate"] = rbns_cmp["true_rbns"]
        rbns_cmp["error"] = rbns_cmp["pred_rbns"] - rbns_cmp["true_rbns"]
        rbns_cmp["abs_error"] = rbns_cmp["error"].abs()
        rbns_cmp["pct_error"] = np.where(rbns_cmp["true_rbns"].abs() > 1e-9, 100.0 * rbns_cmp["error"] / rbns_cmp["true_rbns"], np.nan)
        rbns_cmp["method"] = method_name_rbns
        rbns_cmp["valuation_pay_year"] = int(v)
        rbns_cmp = _append_all_row(rbns_cmp, value_col="pred_rbns", true_col="true_rbns")
        rbns_smry = summarize_ml_compare(rbns_cmp)

        all_truth = _build_true_valuation_targets(
            full_paid_df=full_paid_df,
            full_claims_df=full_claims_df,
            valuation_year=int(v),
            scale_divisor=scale_divisor,
        )
        if eval_recent_ay_window is None:
            key_mask = all_truth["AY"] <= int(v)
        else:
            key_mask = (all_truth["AY"] > int(v) - int(eval_recent_ay_window)) & (all_truth["AY"] <= int(v))
        key_df = all_truth.loc[key_mask, ["AY", "Type", "lob"]].drop_duplicates().copy()
        if key_df.empty:
            return rbns_cmp, rbns_smry, pd.DataFrame(), pd.DataFrame()

        latest_by_key = (
            all_truth.loc[key_mask]
            .groupby(["AY", "Type", "lob"], as_index=False)
            .agg(
                latest_observed=("latest_observed", "sum"),
                true_total_reserve=("true_total_reserve", "sum"),
                true_ultimate=("Ultimate", "sum"),
            )
        )

        rbns_by_key = rbns_cmp.loc[rbns_cmp["lob"] != "All", ["AY", "Type", "lob", "pred_rbns"]].copy()
        ibnr_addon = _estimate_ibnr_addon(
            full_claims_df=full_claims_df,
            valuation_year=int(v),
            train_cutoff=int(train_cutoff),
            keys_df=key_df,
            scale_divisor=scale_divisor,
        )

        if not d_calib.empty:
            calib_year = int(calib_val)
            calib_cutoff = calib_year - 3
            calib_truth = _build_true_valuation_targets(
                full_paid_df=full_paid_df,
                full_claims_df=full_claims_df,
                valuation_year=int(calib_year),
                scale_divisor=scale_divisor,
            )
            if eval_recent_ay_window is None:
                calib_mask = calib_truth["AY"] <= int(calib_year)
            else:
                calib_mask = (calib_truth["AY"] > int(calib_year) - int(eval_recent_ay_window)) & (calib_truth["AY"] <= int(calib_year))
            calib_keys = calib_truth.loc[calib_mask, ["AY", "Type", "lob"]].drop_duplicates().copy()
            if not calib_keys.empty:
                calib_ibnr_pred = _estimate_ibnr_addon(
                    full_claims_df=full_claims_df,
                    valuation_year=int(calib_year),
                    train_cutoff=int(calib_cutoff),
                    keys_df=calib_keys,
                    scale_divisor=scale_divisor,
                )
                calib_ibnr_true = (
                    calib_truth.loc[calib_mask]
                    .groupby(["AY", "Type", "lob"], as_index=False)
                    .agg(true_ibnr=("true_ibnr", "sum"))
                )
                ibnr_calib = calib_keys.merge(calib_ibnr_pred, on=["AY", "Type", "lob"], how="left").merge(calib_ibnr_true, on=["AY", "Type", "lob"], how="left")
                ibnr_calib["pred_ibnr"] = pd.to_numeric(ibnr_calib["pred_ibnr"], errors="coerce").fillna(0.0)
                ibnr_calib["true_ibnr"] = pd.to_numeric(ibnr_calib["true_ibnr"], errors="coerce").fillna(0.0)
                g_ifac, t_ifac = _build_calibration_factors(ibnr_calib, pred_col="pred_ibnr", true_col="true_ibnr", min_rows_type=3)
                ibnr_addon["pred_ibnr"] = _apply_calibration(
                    ibnr_addon["pred_ibnr"].to_numpy(),
                    ibnr_addon,
                    g_ifac,
                    t_ifac,
                    min_mix=float(ibnr_calibration_min_mix),
                )

        hybrid_cmp = (
            key_df.merge(latest_by_key, on=["AY", "Type", "lob"], how="left")
            .merge(rbns_by_key, on=["AY", "Type", "lob"], how="left")
            .merge(ibnr_addon[["AY", "Type", "lob", "pred_ibnr"]], on=["AY", "Type", "lob"], how="left")
        )
        hybrid_cmp["latest_observed"] = pd.to_numeric(hybrid_cmp["latest_observed"], errors="coerce").fillna(0.0)
        hybrid_cmp["true_total_reserve"] = pd.to_numeric(hybrid_cmp["true_total_reserve"], errors="coerce").fillna(0.0)
        hybrid_cmp["pred_rbns"] = pd.to_numeric(hybrid_cmp["pred_rbns"], errors="coerce").fillna(0.0)
        hybrid_cmp["pred_ibnr"] = pd.to_numeric(hybrid_cmp["pred_ibnr"], errors="coerce").fillna(0.0)

        hybrid_cmp["pred_reserve"] = hybrid_cmp["pred_rbns"] + hybrid_cmp["pred_ibnr"]
        hybrid_cmp["ultimate"] = hybrid_cmp["latest_observed"] + hybrid_cmp["pred_reserve"]
        hybrid_cmp["true_ultimate"] = pd.to_numeric(hybrid_cmp["true_ultimate"], errors="coerce").fillna(
            hybrid_cmp["latest_observed"] + hybrid_cmp["true_total_reserve"]
        )
        hybrid_cmp["error"] = hybrid_cmp["ultimate"] - hybrid_cmp["true_ultimate"]
        hybrid_cmp["abs_error"] = hybrid_cmp["error"].abs()
        hybrid_cmp["pct_error"] = np.where(hybrid_cmp["true_ultimate"].abs() > 1e-9, 100.0 * hybrid_cmp["error"] / hybrid_cmp["true_ultimate"], np.nan)
        hybrid_cmp["method"] = method_name_hybrid
        hybrid_cmp["valuation_pay_year"] = int(v)
        hybrid_cmp = _append_all_row(hybrid_cmp, value_col="pred_reserve", true_col="true_total_reserve")
        hybrid_smry = summarize_ml_compare(hybrid_cmp)
        return rbns_cmp, rbns_smry, hybrid_cmp, hybrid_smry

    worker_count = int(os.cpu_count() or 1) if n_jobs_valuation is None else int(n_jobs_valuation)
    worker_count = max(1, worker_count)
    if valuation_years:
        worker_count = min(worker_count, len(valuation_years))

    if worker_count == 1:
        results = [_process_single_valuation(v) for v in valuation_years]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_process_single_valuation, valuation_years))

    for result in results:
        if result is None:
            continue
        rbns_cmp, rbns_smry, hybrid_cmp, hybrid_smry = result
        if not rbns_cmp.empty:
            rbns_compare_rows.append(rbns_cmp)
        if not rbns_smry.empty:
            rbns_summary_rows.append(rbns_smry)
        if not hybrid_cmp.empty:
            hybrid_compare_rows.append(hybrid_cmp)
        if not hybrid_smry.empty:
            hybrid_summary_rows.append(hybrid_smry)

    rbns_compare_tbl = pd.concat(rbns_compare_rows, ignore_index=True) if rbns_compare_rows else pd.DataFrame()
    rbns_summary_tbl = pd.concat(rbns_summary_rows, ignore_index=True) if rbns_summary_rows else pd.DataFrame()
    hybrid_compare_tbl = pd.concat(hybrid_compare_rows, ignore_index=True) if hybrid_compare_rows else pd.DataFrame()
    hybrid_summary_tbl = pd.concat(hybrid_summary_rows, ignore_index=True) if hybrid_summary_rows else pd.DataFrame()

    return rbns_compare_tbl, rbns_summary_tbl, hybrid_compare_tbl, hybrid_summary_tbl
