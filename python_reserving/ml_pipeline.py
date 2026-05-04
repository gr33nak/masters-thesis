from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor


FORBIDDEN_LEAKAGE_COLUMNS = {
    "Ultimate",
    "CumPaid",
    "PayCount",
    "Status",
    "SetMonth",
    "SetDelMonths",
}

# Intentionally exclude absolute calendar-year markers and unstable open/report flags.
# The selected default set reflects the stable ablation winner used in Section 12.
DEFAULT_FEATURE_COLUMNS = [
    "dev_at_val",
    "Type",
    "AQ",
    "cc",
    "inj_part",
    "Age",
    "RepDelayYears",
    "rep_lag_years",
    "paid_to_val",
    "n_pay_to_val",
    "last_pay_lag_years",
    "last_obs_pay",
    "cum_paid_last_1",
    "cum_paid_last_2",
    "cum_paid_last_3",
    "share_pay_years",
    "mean_pos_pay",
    "max_pos_pay",
    "zero_streak_to_val",
    "paid_per_observed_year",
]


def _safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _safe_divide(a, b, default: float = 0.0):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full_like(a_arr, float(default), dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def _extract_raw_path_features(
    raw_output_df: pd.DataFrame,
    valuation_year: int,
    scale_divisor: float,
) -> pd.DataFrame:
    raw = raw_output_df.copy()
    raw["Id"] = pd.to_numeric(raw["Id"], errors="coerce").astype("Int64")
    raw["AY"] = pd.to_numeric(raw["AY"], errors="coerce")

    pay_cols = sorted([c for c in raw.columns if str(c).startswith("Pay")], key=lambda x: int(str(x)[3:]))
    open_cols = sorted([c for c in raw.columns if str(c).startswith("Open")], key=lambda x: int(str(x)[4:]))
    if not pay_cols:
        raise KeyError("Expected Pay00..PayNN columns in raw_output_df.")

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
    pos_mask = observed_pay > 0.0

    paid_to_val = observed_pay.sum(axis=1)
    n_pay_to_val = pos_mask.sum(axis=1).astype(float)

    latest_open_ind = open_mat[row_idx, dev].astype(float)
    last_obs_pay = observed_pay[row_idx, dev]

    last_pos_idx = np.where(pos_mask, col_idx, -1).max(axis=1)
    last_pay_lag_years = np.where(last_pos_idx >= 0, dev - last_pos_idx, dev + 1).astype(float)
    has_reported_payment = (n_pay_to_val > 0).astype(float)

    latest_idx = dev
    prev1_idx = np.maximum(dev - 1, 0)
    prev2_idx = np.maximum(dev - 2, 0)

    cum_paid_last_1 = observed_pay[row_idx, latest_idx]
    cum_paid_last_2 = observed_pay[row_idx, latest_idx] + np.where(dev >= 1, observed_pay[row_idx, prev1_idx], 0.0)
    cum_paid_last_3 = (
        observed_pay[row_idx, latest_idx]
        + np.where(dev >= 1, observed_pay[row_idx, prev1_idx], 0.0)
        + np.where(dev >= 2, observed_pay[row_idx, prev2_idx], 0.0)
    )

    observed_periods = dev + 1
    share_pay_years = _safe_divide(n_pay_to_val, observed_periods, default=0.0)

    pos_pay_sum = np.where(pos_mask, observed_pay, 0.0).sum(axis=1)
    mean_pos_pay = np.zeros(len(raw), dtype=float)
    max_pos_pay = np.zeros(len(raw), dtype=float)
    has_pos = n_pay_to_val > 0
    if np.any(has_pos):
        mean_pos_pay[has_pos] = pos_pay_sum[has_pos] / n_pay_to_val[has_pos]
        max_pos_pay[has_pos] = np.where(pos_mask[has_pos], observed_pay[has_pos], 0.0).max(axis=1)

    zero_streak = np.zeros(len(raw), dtype=float)
    for j in range(n_dev):
        active = dev >= j
        is_zero = observed_pay[:, j] <= 0.0
        zero_streak = np.where(active & is_zero, zero_streak + 1.0, np.where(active, 0.0, zero_streak))

    open_years_to_val = ((open_mat > 0.0) & observed_mask).sum(axis=1).astype(float)
    ever_open_to_val = (open_years_to_val > 0).astype(float)

    return pd.DataFrame(
        {
            "Id": raw["Id"],
            "paid_to_val": paid_to_val / float(scale_divisor),
            "n_pay_to_val": n_pay_to_val,
            "last_pay_lag_years": last_pay_lag_years,
            "latest_open_ind": latest_open_ind,
            "has_reported_payment": has_reported_payment,
            "last_obs_pay": last_obs_pay / float(scale_divisor),
            "cum_paid_last_1": cum_paid_last_1 / float(scale_divisor),
            "cum_paid_last_2": cum_paid_last_2 / float(scale_divisor),
            "cum_paid_last_3": cum_paid_last_3 / float(scale_divisor),
            "share_pay_years": share_pay_years,
            "mean_pos_pay": mean_pos_pay / float(scale_divisor),
            "max_pos_pay": max_pos_pay / float(scale_divisor),
            "zero_streak_to_val": zero_streak,
            "open_years_to_val": open_years_to_val,
            "ever_open_to_val": ever_open_to_val,
        }
    )


def _extract_paid_agg_features(
    full_paid_df: pd.DataFrame,
    valuation_year: int,
    scale_divisor: float,
) -> pd.DataFrame:
    paid = full_paid_df.copy()
    paid["Id"] = pd.to_numeric(paid["Id"], errors="coerce").astype("Int64")
    paid["PayYear"] = pd.to_numeric(paid["PayYear"], errors="coerce").astype("Int64")
    paid["DY"] = pd.to_numeric(paid["DY"], errors="coerce").astype("Int64")
    paid["Paid"] = pd.to_numeric(paid["Paid"], errors="coerce").fillna(0.0)
    paid["OpenInd"] = pd.to_numeric(paid.get("OpenInd", 0.0), errors="coerce").fillna(0.0)

    obs = paid.loc[paid["PayYear"].notna() & (paid["PayYear"] <= int(valuation_year))].copy()
    if obs.empty:
        return pd.DataFrame(columns=["Id"] + [c for c in DEFAULT_FEATURE_COLUMNS if c not in {"Type"}])

    obs = obs.sort_values(["Id", "DY"]).copy()
    obs["pay_pos"] = (obs["Paid"] > 0.0).astype(float)

    base = obs.groupby("Id", as_index=False).agg(
        paid_to_val=("Paid", "sum"),
        n_pay_to_val=("pay_pos", "sum"),
        open_years_to_val=("OpenInd", "sum"),
        ever_open_to_val=("OpenInd", lambda s: float((s > 0).any())),
    )
    base["paid_to_val"] = base["paid_to_val"] / float(scale_divisor)

    last_positive = (
        obs.loc[obs["Paid"] > 0.0]
        .groupby("Id", as_index=False)
        .agg(last_pay_year_to_val=("PayYear", "max"))
    )

    latest = (
        obs.sort_values(["Id", "PayYear", "DY"])
        .groupby("Id", as_index=False)
        .tail(1)
        .loc[:, ["Id", "Paid", "OpenInd"]]
        .rename(columns={"Paid": "last_obs_pay", "OpenInd": "latest_open_ind"})
    )
    latest["last_obs_pay"] = latest["last_obs_pay"] / float(scale_divisor)

    last_k = []
    for k in [1, 2, 3]:
        tmp = (
            obs.groupby("Id", group_keys=False)
            .tail(k)
            .groupby("Id", as_index=False)["Paid"]
            .sum()
            .rename(columns={"Paid": f"cum_paid_last_{k}"})
        )
        tmp[f"cum_paid_last_{k}"] = tmp[f"cum_paid_last_{k}"] / float(scale_divisor)
        last_k.append(tmp)

    counts = obs.groupby("Id").size().rename("n_obs").reset_index()
    pos_only = obs.loc[obs["Paid"] > 0.0].copy()
    if pos_only.empty:
        pos_stats = pd.DataFrame(columns=["Id", "mean_pos_pay", "max_pos_pay"])
    else:
        pos_stats = pos_only.groupby("Id", as_index=False)["Paid"].agg(mean_pos_pay="mean", max_pos_pay="max")
        pos_stats["mean_pos_pay"] = pos_stats["mean_pos_pay"] / float(scale_divisor)
        pos_stats["max_pos_pay"] = pos_stats["max_pos_pay"] / float(scale_divisor)

    zero_streak_rows = []
    for _id, grp in obs.groupby("Id", sort=False):
        vals = grp.sort_values("DY")["Paid"].to_numpy(dtype=float)
        streak = 0.0
        for val in vals[::-1]:
            if val <= 0.0:
                streak += 1.0
            else:
                break
        zero_streak_rows.append({"Id": _id, "zero_streak_to_val": streak})
    zero_streak = pd.DataFrame(zero_streak_rows)

    out = base.merge(last_positive, on="Id", how="left").merge(latest, on="Id", how="left").merge(counts, on="Id", how="left")
    for tmp in last_k:
        out = out.merge(tmp, on="Id", how="left")
    out = out.merge(pos_stats, on="Id", how="left").merge(zero_streak, on="Id", how="left")

    out["n_obs"] = pd.to_numeric(out["n_obs"], errors="coerce").fillna(0.0)
    out["share_pay_years"] = np.where(out["n_obs"] > 0, out["n_pay_to_val"] / out["n_obs"], 0.0)
    out["has_reported_payment"] = (out["n_pay_to_val"] > 0).astype(float)
    return out


def build_claim_snapshot_dataset(
    full_paid_df: pd.DataFrame,
    full_claims_df: pd.DataFrame,
    valuation_year: int,
    scale_divisor: float = 1000.0,
    include_unreported: bool = False,
    raw_output_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    claims = full_claims_df.copy()
    claims["Id"] = pd.to_numeric(claims["Id"], errors="coerce").astype("Int64")
    claims["AY"] = pd.to_numeric(claims["AY"], errors="coerce").astype("Int64")
    if "RepAY" in claims.columns:
        claims["RepYear"] = pd.to_numeric(claims["RepAY"], errors="coerce").astype("Int64")
    else:
        claims["RepYear"] = pd.Series(pd.NA, index=claims.index, dtype="Int64")

    claims = claims.loc[claims["AY"].notna() & claims["Type"].notna()].copy()
    claims = claims.loc[claims["AY"] <= int(valuation_year)].copy()

    if not include_unreported:
        claims = claims.loc[claims["RepYear"].notna() & (claims["RepYear"] <= int(valuation_year))].copy()

    if claims.empty:
        return claims

    claims["Type"] = pd.to_numeric(claims["Type"], errors="coerce").astype(int)

    if raw_output_df is not None and {"Id", "AY"}.issubset(raw_output_df.columns):
        feat = _extract_raw_path_features(raw_output_df=raw_output_df, valuation_year=int(valuation_year), scale_divisor=float(scale_divisor))
    else:
        feat = _extract_paid_agg_features(full_paid_df=full_paid_df, valuation_year=int(valuation_year), scale_divisor=float(scale_divisor))

    paid_hist = full_paid_df.copy()
    paid_hist["Id"] = pd.to_numeric(paid_hist["Id"], errors="coerce").astype("Int64")
    paid_hist["PayYear"] = pd.to_numeric(paid_hist["PayYear"], errors="coerce").astype("Int64")
    paid_hist["Paid"] = pd.to_numeric(paid_hist["Paid"], errors="coerce").fillna(0.0)

    paid_to_val_truth = (
        paid_hist.loc[paid_hist["PayYear"].notna() & (paid_hist["PayYear"] <= int(valuation_year))]
        .groupby("Id", as_index=False)
        .agg(paid_to_val_truth=("Paid", "sum"))
    )
    paid_to_val_truth["paid_to_val_truth"] = paid_to_val_truth["paid_to_val_truth"] / float(scale_divisor)

    out = claims.merge(feat, on="Id", how="left").merge(paid_to_val_truth, on="Id", how="left")

    out["Age"] = _safe_numeric(out, "Age", 0.0)
    out["RepDelayYears"] = _safe_numeric(out, "RepDelayYears", 0.0)
    out["AQ"] = _safe_numeric(out, "AQ", 0.0)
    out["cc"] = _safe_numeric(out, "cc", 0.0)
    out["inj_part"] = _safe_numeric(out, "inj_part", 0.0)

    numeric_fill_cols = [
        "paid_to_val",
        "n_pay_to_val",
        "last_pay_lag_years",
        "latest_open_ind",
        "has_reported_payment",
        "last_obs_pay",
        "cum_paid_last_1",
        "cum_paid_last_2",
        "cum_paid_last_3",
        "share_pay_years",
        "mean_pos_pay",
        "max_pos_pay",
        "zero_streak_to_val",
        "open_years_to_val",
        "ever_open_to_val",
        "paid_to_val_truth",
    ]
    for c in numeric_fill_cols:
        out[c] = _safe_numeric(out, c, 0.0)

    out["paid_to_val"] = out["paid_to_val"].fillna(out["paid_to_val_truth"])
    out["valuation_year"] = int(valuation_year)
    out["dev_at_val"] = np.maximum(int(valuation_year) - pd.to_numeric(out["AY"], errors="coerce"), 0.0)
    out["rep_lag_years"] = np.where(
        out["RepYear"].notna(),
        np.maximum(int(valuation_year) - pd.to_numeric(out["RepYear"], errors="coerce"), 0.0),
        0.0,
    )
    out["last_pay_lag_years"] = np.where(
        np.isfinite(pd.to_numeric(out["last_pay_lag_years"], errors="coerce")),
        pd.to_numeric(out["last_pay_lag_years"], errors="coerce"),
        out["dev_at_val"] + 1.0,
    )

    obs_years = np.maximum(out["dev_at_val"] + 1.0, 1.0)
    out["paid_per_observed_year"] = out["paid_to_val"] / obs_years
    out["paid_to_val_x_open_years"] = out["paid_to_val"] * out["open_years_to_val"]
    out["dev_x_rep_lag"] = out["dev_at_val"] * out["rep_lag_years"]

    if "Ultimate" in out.columns:
        out["ultimate_true"] = pd.to_numeric(out["Ultimate"], errors="coerce").fillna(0.0) / float(scale_divisor)
        out["outstanding_true"] = np.maximum(out["ultimate_true"] - out["paid_to_val_truth"], 0.0)
        out["y"] = out["outstanding_true"]
    else:
        out["ultimate_true"] = np.nan
        out["outstanding_true"] = np.nan
        out["y"] = np.nan

    out["lob"] = out["Type"].map(lambda t: f"Type_{int(t)}")

    for c in DEFAULT_FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0

    return out


def fit_xgb_claim_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    occ_params: Optional[dict] = None,
    sev_params: Optional[dict] = None,
    pos_count_threshold: int = 300,
):
    X_train = X_train.copy()
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0.0)

    y_pos = (y_train > 1e-9).astype(int)
    if y_pos.nunique() < 2:
        return {
            "mode": "degenerate",
            "p_const": float(y_pos.iloc[0]) if len(y_pos) > 0 else 0.0,
            "sev_const": float(y_train[y_train > 0.0].mean()) if (y_train > 0.0).any() else 0.0,
        }

    occ_defaults = {
        "n_estimators": 320,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.25,
        "reg_lambda": 5.0,
        "min_child_weight": 8.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
    }
    if occ_params:
        occ_defaults.update(occ_params)
    occ = XGBClassifier(**occ_defaults)
    occ.fit(X_train, y_pos)

    pos_count = int(y_pos.sum())
    if pos_count < int(pos_count_threshold):
        return {
            "mode": "two_stage",
            "occ_model": occ,
            "sev_model": None,
            "sev_const": float(y_train[y_train > 0.0].mean()) if pos_count > 0 else 0.0,
        }

    sev_defaults = {
        "n_estimators": 360,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.25,
        "reg_lambda": 5.0,
        "min_child_weight": 10.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 4,
    }
    if sev_params:
        sev_defaults.update(sev_params)
    sev = XGBRegressor(**sev_defaults)
    sev.fit(X_train.loc[y_pos > 0, :], np.log1p(y_train.loc[y_pos > 0]))

    return {
        "mode": "two_stage",
        "occ_model": occ,
        "sev_model": sev,
        "sev_const": float(y_train[y_train > 0.0].mean()) if pos_count > 0 else 0.0,
    }


def predict_xgb_claim_model(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict):
        mode = model.get("mode", "two_stage")

        if mode == "degenerate":
            p_const = float(model.get("p_const", 0.0))
            sev_const = max(float(model.get("sev_const", 0.0)), 0.0)
            return np.full(len(X), p_const * sev_const, dtype=float)

        occ_model = model.get("occ_model")
        sev_model = model.get("sev_model")
        p_occ = occ_model.predict_proba(X)[:, 1] if occ_model is not None else np.zeros(len(X), dtype=float)
        if sev_model is None:
            sev_pred = np.full(len(X), max(float(model.get("sev_const", 0.0)), 0.0), dtype=float)
        else:
            sev_pred = np.maximum(np.expm1(sev_model.predict(X)), 0.0)
        return np.maximum(p_occ * sev_pred, 0.0)

    return np.maximum(model.predict(X), 0.0)


def summarize_ml_compare(compare_df: pd.DataFrame) -> pd.DataFrame:
    if compare_df.empty:
        return pd.DataFrame()
    return (
        compare_df.groupby(["valuation_pay_year", "lob", "method"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("error", "mean"),
            mape=("pct_error", lambda s: float(np.nanmean(np.abs(s)))),
        )
    )


def _append_all_row(compare_df: pd.DataFrame, value_col: str, true_col: str) -> pd.DataFrame:
    if compare_df.empty:
        return compare_df

    grp = (
        compare_df.groupby(["AY", "valuation_pay_year", "method"], as_index=False)
        .agg(
            latest_observed=("latest_observed", "sum"),
            pred_value=(value_col, "sum"),
            true_value=(true_col, "sum"),
        )
    )
    grp["ultimate"] = grp["latest_observed"] + grp["pred_value"]
    grp["true_ultimate"] = grp["latest_observed"] + grp["true_value"]
    grp["error"] = grp["ultimate"] - grp["true_ultimate"]
    grp["abs_error"] = grp["error"].abs()
    grp["pct_error"] = np.where(grp["true_ultimate"].abs() > 1e-9, 100.0 * grp["error"] / grp["true_ultimate"], np.nan)
    grp[value_col] = grp["pred_value"]
    grp[true_col] = grp["true_value"]
    grp["lob"] = "All"
    grp = grp.drop(columns=["pred_value", "true_value"])
    return pd.concat([compare_df, grp], ignore_index=True, sort=False)


def _build_true_valuation_targets(
    full_paid_df: pd.DataFrame,
    full_claims_df: pd.DataFrame,
    valuation_year: int,
    scale_divisor: float,
) -> pd.DataFrame:
    claims = full_claims_df.copy()
    claims["Id"] = pd.to_numeric(claims["Id"], errors="coerce").astype("Int64")
    claims["AY"] = pd.to_numeric(claims["AY"], errors="coerce").astype("Int64")
    claims["Type"] = pd.to_numeric(claims["Type"], errors="coerce").astype("Int64")
    claims["RepAY"] = pd.to_numeric(claims.get("RepAY", pd.NA), errors="coerce").astype("Int64")
    claims["Ultimate"] = pd.to_numeric(claims.get("Ultimate", 0.0), errors="coerce").fillna(0.0) / float(scale_divisor)

    paid = full_paid_df.copy()
    paid["Id"] = pd.to_numeric(paid["Id"], errors="coerce").astype("Int64")
    paid["PayYear"] = pd.to_numeric(paid["PayYear"], errors="coerce").astype("Int64")
    paid["Paid"] = pd.to_numeric(paid["Paid"], errors="coerce").fillna(0.0)

    paid_to_val = (
        paid.loc[paid["PayYear"].notna() & (paid["PayYear"] <= int(valuation_year))]
        .groupby("Id", as_index=False)
        .agg(latest_observed=("Paid", "sum"))
    )
    paid_to_val["latest_observed"] = paid_to_val["latest_observed"] / float(scale_divisor)

    out = claims.merge(paid_to_val, on="Id", how="left")
    out["latest_observed"] = pd.to_numeric(out["latest_observed"], errors="coerce").fillna(0.0)
    out["true_total_reserve"] = np.maximum(out["Ultimate"] - out["latest_observed"], 0.0)
    out["reported_ind"] = (out["RepAY"].notna() & (out["RepAY"] <= int(valuation_year))).astype(float)
    out["true_rbns"] = np.where(out["reported_ind"] > 0, out["true_total_reserve"], 0.0)
    out["true_ibnr"] = np.where(out["reported_ind"] > 0, 0.0, out["Ultimate"])
    out["lob"] = out["Type"].map(lambda t: f"Type_{int(t)}")
    return out


def _estimate_ibnr_addon(
    full_claims_df: pd.DataFrame,
    valuation_year: int,
    train_cutoff: int,
    keys_df: pd.DataFrame,
    scale_divisor: float,
) -> pd.DataFrame:
    claims = full_claims_df.copy()
    claims["AY"] = pd.to_numeric(claims["AY"], errors="coerce").astype("Int64")
    claims["Type"] = pd.to_numeric(claims["Type"], errors="coerce").astype("Int64")
    claims["RepAY"] = pd.to_numeric(claims.get("RepAY", pd.NA), errors="coerce").astype("Int64")
    claims["RepDelayYears"] = pd.to_numeric(claims.get("RepDelayYears", 0.0), errors="coerce").fillna(0.0)
    claims["Ultimate"] = pd.to_numeric(claims.get("Ultimate", 0.0), errors="coerce").fillna(0.0) / float(scale_divisor)

    train = claims.loc[claims["AY"].notna() & (claims["AY"] <= int(train_cutoff))].copy()
    global_train = train.copy()

    rows = []
    for row in keys_df.itertuples(index=False):
        ay = int(row.AY)
        typ = int(row.Type)
        lob = row.lob
        dev = max(int(valuation_year) - ay, 0)

        train_type = train.loc[train["Type"] == typ].copy()
        credible_type = len(train_type) >= 300
        ref = train_type if credible_type else global_train
        if ref.empty:
            rows.append({"AY": ay, "Type": typ, "lob": lob, "pred_ibnr": 0.0})
            continue

        reported_now = claims.loc[
            claims["AY"].eq(ay)
            & claims["Type"].eq(typ)
            & claims["RepAY"].notna()
            & (claims["RepAY"] <= int(valuation_year))
        ].copy()
        reported_count = float(len(reported_now))

        p_rep_type = float((ref["RepDelayYears"] <= dev).mean())
        p_rep_global = float((global_train["RepDelayYears"] <= dev).mean()) if not global_train.empty else p_rep_type
        weight = min(max(len(train_type) / 1000.0, 0.0), 1.0) if len(train_type) > 0 else 0.0
        p_reported = weight * p_rep_type + (1.0 - weight) * p_rep_global
        p_reported = min(max(p_reported, 0.10), 0.995)

        late_type = train_type.loc[train_type["RepDelayYears"] > dev, "Ultimate"]
        late_global = global_train.loc[global_train["RepDelayYears"] > dev, "Ultimate"]
        sev_type = float(late_type.mean()) if not late_type.empty else np.nan
        sev_global = float(late_global.mean()) if not late_global.empty else float(global_train["Ultimate"].mean())
        late_severity = sev_type if np.isfinite(sev_type) and len(late_type) >= 50 else sev_global
        late_severity = max(float(late_severity), 0.0)

        est_total_count = reported_count / p_reported
        ibnr_count = max(est_total_count - reported_count, 0.0)
        pred_ibnr = ibnr_count * late_severity

        rows.append(
            {
                "AY": ay,
                "Type": typ,
                "lob": lob,
                "pred_ibnr": float(pred_ibnr),
            }
        )

    return pd.DataFrame(rows)


def _stack_training_snapshots(
    full_paid_df: pd.DataFrame,
    full_claims_df: pd.DataFrame,
    valuation_years: list[int],
    current_valuation: int,
    train_cutoff: int,
    scale_divisor: float,
    raw_output_df: Optional[pd.DataFrame],
    lookback_valuations: int,
    max_train_rows: int,
) -> pd.DataFrame:
    # valuation_years here is the training-snapshot pool — any year strictly
    # less than current_valuation is a candidate, regardless of whether it is
    # also an evaluation year upstream.
    prior_vals = sorted({int(x) for x in valuation_years if int(x) < int(current_valuation)})
    if lookback_valuations is not None and lookback_valuations > 0:
        prior_vals = prior_vals[-int(lookback_valuations):]

    frames = []
    for s in prior_vals:
        snap = build_claim_snapshot_dataset(
            full_paid_df=full_paid_df,
            full_claims_df=full_claims_df,
            valuation_year=int(s),
            scale_divisor=scale_divisor,
            include_unreported=False,
            raw_output_df=raw_output_df,
        )
        if snap.empty:
            continue
        snap = snap.loc[snap["AY"] <= int(train_cutoff)].copy()
        if snap.empty:
            continue
        frames.append(snap)

    if not frames:
        return pd.DataFrame()

    train_df = pd.concat(frames, ignore_index=True)
    train_df = train_df.sort_values(["valuation_year", "AY", "Id"]).drop_duplicates(subset=["valuation_year", "Id"], keep="last")

    if max_train_rows is not None and len(train_df) > int(max_train_rows):
        rng = np.random.default_rng(42)
        y = pd.to_numeric(train_df["y"], errors="coerce").fillna(0.0)
        pos_idx = train_df.index[y > 0.0]
        zero_idx = train_df.index[y <= 0.0]
        keep_pos = min(len(pos_idx), int(max_train_rows * 0.60))
        keep_zero = max(int(max_train_rows) - keep_pos, 0)
        chosen_pos = rng.choice(pos_idx.to_numpy(), size=keep_pos, replace=False) if keep_pos < len(pos_idx) else pos_idx.to_numpy()
        chosen_zero = rng.choice(zero_idx.to_numpy(), size=min(keep_zero, len(zero_idx)), replace=False) if keep_zero < len(zero_idx) else zero_idx.to_numpy()
        chosen = np.concatenate([chosen_pos, chosen_zero])
        train_df = train_df.loc[np.sort(chosen)].copy()

    return train_df.reset_index(drop=True)


def _fit_predict_with_credibility_blend(
    d_train: pd.DataFrame,
    d_pred: pd.DataFrame,
    feat_cols: list[str],
    min_rows_type_model: int,
    occ_params: Optional[dict] = None,
    sev_params: Optional[dict] = None,
    pos_count_threshold: int = 300,
) -> np.ndarray:
    pooled_model = fit_xgb_claim_model(
        d_train[feat_cols],
        d_train["y"].astype(float),
        occ_params=occ_params,
        sev_params=sev_params,
        pos_count_threshold=int(pos_count_threshold),
    )
    pooled_pred = predict_xgb_claim_model(pooled_model, d_pred[feat_cols])
    final_pred = pooled_pred.copy()

    for typ in sorted(d_pred["Type"].dropna().astype(int).unique().tolist()):
        pred_mask = d_pred["Type"].eq(typ)
        train_mask = d_train["Type"].eq(typ)
        n_type = int(train_mask.sum())
        if n_type < int(min_rows_type_model):
            continue

        type_model = fit_xgb_claim_model(
            d_train.loc[train_mask, feat_cols],
            d_train.loc[train_mask, "y"].astype(float),
            occ_params=occ_params,
            sev_params=sev_params,
            pos_count_threshold=int(pos_count_threshold),
        )
        type_pred = predict_xgb_claim_model(type_model, d_pred.loc[pred_mask, feat_cols])
        alpha = min(max((n_type - min_rows_type_model) / float(max(min_rows_type_model, 1)), 0.0), 1.0)
        alpha = 0.20 + 0.40 * alpha
        final_pred[pred_mask.to_numpy()] = alpha * type_pred + (1.0 - alpha) * pooled_pred[pred_mask.to_numpy()]

    return np.maximum(final_pred, 0.0)


def _build_calibration_factors(
    calib_df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    min_rows_type: int = 500,
) -> tuple[float, dict[int, float]]:
    eps = 1e-9
    pred_sum = float(pd.to_numeric(calib_df[pred_col], errors="coerce").fillna(0.0).sum())
    true_sum = float(pd.to_numeric(calib_df[true_col], errors="coerce").fillna(0.0).sum())
    global_factor = true_sum / max(pred_sum, eps) if pred_sum > eps else 1.0
    global_factor = float(np.clip(global_factor, 0.35, 2.25))

    type_factors: dict[int, float] = {}
    for typ, grp in calib_df.groupby("Type"):
        if len(grp) < int(min_rows_type):
            continue
        p = float(pd.to_numeric(grp[pred_col], errors="coerce").fillna(0.0).sum())
        t = float(pd.to_numeric(grp[true_col], errors="coerce").fillna(0.0).sum())
        if p <= eps:
            continue
        fac = t / p
        fac = float(np.clip(fac, 0.35, 2.25))
        type_factors[int(typ)] = fac
    return global_factor, type_factors


def _apply_calibration(
    preds: np.ndarray,
    d_pred: pd.DataFrame,
    global_factor: float,
    type_factors: dict[int, float],
    min_mix: float = 0.25,
) -> np.ndarray:
    out = np.asarray(preds, dtype=float).copy()
    type_arr = pd.to_numeric(d_pred["Type"], errors="coerce").fillna(-1).astype(int).to_numpy()
    for i, typ in enumerate(type_arr):
        fac = type_factors.get(int(typ), global_factor)
        # blend type factor toward global factor to avoid instability
        blended = (1.0 - min_mix) * global_factor + min_mix * fac
        out[i] = max(out[i] * blended, 0.0)
    return out


def run_claim_level_rolling_origin_backtest(
    full_paid_df: pd.DataFrame,
    full_claims_df: pd.DataFrame,
    valuation_years: Iterable[int],
    train_snapshot_years: Optional[Iterable[int]] = None,
    scale_divisor: float = 1000.0,
    feature_cols: Optional[list[str]] = None,
    method_name_rbns: str = "ML-Claim-XGB-RBNS-Cal",
    method_name_hybrid: str = "ML-Claim-XGB+IBNR-Cal",
    min_train_rows: int = 300,
    raw_output_df: Optional[pd.DataFrame] = None,
    lookback_valuations: int = 4,
    max_train_rows: int = 250_000,
    min_rows_type_model: int = 2_000,
    train_cutoff_lag: int = 3,
    eval_recent_ay_window: Optional[int] = 3,
    occ_params: Optional[dict] = None,
    sev_params: Optional[dict] = None,
    pos_count_threshold: int = 300,
    calibration_min_mix: float = 0.35,
    ibnr_calibration_min_mix: float = 0.35,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    used_features = feature_cols if feature_cols is not None else list(DEFAULT_FEATURE_COLUMNS)
    valuation_years = [int(v) for v in valuation_years]
    extra_train_years = [int(y) for y in (train_snapshot_years or [])]
    training_pool = sorted(set(valuation_years) | set(extra_train_years))
    train_cutoff_lag = max(int(train_cutoff_lag), 0)

    rbns_compare_rows = []
    rbns_summary_rows = []
    hybrid_compare_rows = []
    hybrid_summary_rows = []

    for v in valuation_years:
        d_test_full = build_claim_snapshot_dataset(
            full_paid_df=full_paid_df,
            full_claims_df=full_claims_df,
            valuation_year=int(v),
            scale_divisor=scale_divisor,
            include_unreported=False,
            raw_output_df=raw_output_df,
        )
        if d_test_full.empty:
            continue

        train_cutoff = int(v) - int(train_cutoff_lag)
        if eval_recent_ay_window is None:
            d_test = d_test_full.copy()
        else:
            d_test = d_test_full.loc[d_test_full["AY"] > int(v) - int(eval_recent_ay_window)].copy()
        if d_test.empty:
            continue

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
            continue

        # Use the most recent prior valuation snapshot as an out-of-time calibration window.
        calib_val = int(d_train["valuation_year"].max()) if "valuation_year" in d_train.columns else None
        if calib_val is not None and (d_train["valuation_year"] < calib_val).sum() >= int(min_train_rows):
            d_fit = d_train.loc[d_train["valuation_year"] < calib_val].copy()
            d_calib = d_train.loc[d_train["valuation_year"] == calib_val].copy()
        else:
            d_fit = d_train.copy()
            d_calib = pd.DataFrame()

        feat_cols = [c for c in used_features if c in d_fit.columns and c in d_test.columns]
        if not feat_cols:
            continue

        raw_test_pred = _fit_predict_with_credibility_blend(
            d_train=d_fit,
            d_pred=d_test,
            feat_cols=feat_cols,
            min_rows_type_model=int(min_rows_type_model),
            occ_params=occ_params,
            sev_params=sev_params,
            pos_count_threshold=int(pos_count_threshold),
        )

        if not d_calib.empty:
            raw_calib_pred = _fit_predict_with_credibility_blend(
                d_train=d_fit,
                d_pred=d_calib,
                feat_cols=feat_cols,
                min_rows_type_model=int(min_rows_type_model),
                occ_params=occ_params,
                sev_params=sev_params,
                pos_count_threshold=int(pos_count_threshold),
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
        rbns_compare_rows.append(rbns_cmp)

        rbns_smry = summarize_ml_compare(rbns_cmp)
        if not rbns_smry.empty:
            rbns_summary_rows.append(rbns_smry)

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
            continue

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

        # Calibrate the simple IBNR block using the most recent prior valuation if available.
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
        hybrid_compare_rows.append(hybrid_cmp)

        hybrid_smry = summarize_ml_compare(hybrid_cmp)
        if not hybrid_smry.empty:
            hybrid_summary_rows.append(hybrid_smry)

    rbns_compare_tbl = pd.concat(rbns_compare_rows, ignore_index=True) if rbns_compare_rows else pd.DataFrame()
    rbns_summary_tbl = pd.concat(rbns_summary_rows, ignore_index=True) if rbns_summary_rows else pd.DataFrame()
    hybrid_compare_tbl = pd.concat(hybrid_compare_rows, ignore_index=True) if hybrid_compare_rows else pd.DataFrame()
    hybrid_summary_tbl = pd.concat(hybrid_summary_rows, ignore_index=True) if hybrid_summary_rows else pd.DataFrame()

    return rbns_compare_tbl, rbns_summary_tbl, hybrid_compare_tbl, hybrid_summary_tbl
