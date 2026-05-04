from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .synthetic_loader import pay_columns_from_output, true_ultimate_by_ay_from_output


def _lob_key_to_label(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "All"
    return f"Type_{int(v)}"


def evaluate_triangle_predictions(
    pred_tbl: pd.DataFrame,
    true_ultimate,
    valuation_year: int,
    claim_type: Optional[int],
) -> pd.DataFrame:
    if pred_tbl is None or pred_tbl.empty:
        return pd.DataFrame()

    cmp = pred_tbl.copy()
    cmp["AY"] = pd.to_numeric(cmp["AY"], errors="coerce").astype("Int64")

    if "lob_key" not in cmp.columns:
        cmp["lob_key"] = pd.NA if claim_type is None else int(claim_type)

    if isinstance(true_ultimate, pd.DataFrame):
        truth = true_ultimate.copy()
        truth["AY"] = pd.to_numeric(truth["AY"], errors="coerce").astype("Int64")
        if "lob_key" not in truth.columns:
            truth["lob_key"] = pd.NA if claim_type is None else int(claim_type)
        truth["true_ultimate"] = pd.to_numeric(truth["true_ultimate"], errors="coerce")
    else:
        s = pd.to_numeric(true_ultimate, errors="coerce")
        s.index = pd.to_numeric(s.index, errors="coerce").astype("Int64")
        truth = pd.DataFrame(
            {
                "AY": s.index,
                "lob_key": pd.NA if claim_type is None else int(claim_type),
                "true_ultimate": s.values,
            }
        )

    cmp["_lob_merge_key"] = cmp["lob_key"].map(_lob_key_to_label)
    truth["_lob_merge_key"] = truth["lob_key"].map(_lob_key_to_label)
    cmp = cmp.merge(
        truth[["AY", "_lob_merge_key", "true_ultimate"]],
        on=["AY", "_lob_merge_key"],
        how="left",
    )
    cmp = cmp.drop(columns=["_lob_merge_key"])

    cmp["latest_observed"] = pd.to_numeric(cmp["latest_observed"], errors="coerce").fillna(0.0)
    cmp["ultimate"] = pd.to_numeric(cmp["ultimate"], errors="coerce").fillna(0.0)
    cmp["pred_reserve"] = pd.to_numeric(cmp["pred_reserve"], errors="coerce").fillna(0.0)
    cmp["true_ultimate"] = pd.to_numeric(cmp["true_ultimate"], errors="coerce").fillna(0.0)

    cmp["true_total_reserve"] = cmp["true_ultimate"] - cmp["latest_observed"]
    cmp["error"] = cmp["ultimate"] - cmp["true_ultimate"]
    cmp["abs_error"] = cmp["error"].abs()
    cmp["pct_error"] = np.where(
        cmp["true_ultimate"].abs() > 1e-9,
        100.0 * cmp["error"] / cmp["true_ultimate"],
        np.nan,
    )
    cmp["valuation_pay_year"] = int(valuation_year)
    cmp["lob"] = cmp["lob_key"].map(_lob_key_to_label)
    return cmp


def summarize_triangle_compare(compare_tbl: pd.DataFrame) -> pd.DataFrame:
    if compare_tbl is None or compare_tbl.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "lob",
                "valuation_pay_year",
                "mae",
                "rmse",
                "bias",
                "mape",
                "n_obs",
            ]
        )

    d = compare_tbl.copy()
    d["error"] = pd.to_numeric(d["error"], errors="coerce").fillna(0.0)
    d["abs_error"] = pd.to_numeric(d["abs_error"], errors="coerce").fillna(0.0)
    d["pct_error"] = pd.to_numeric(d["pct_error"], errors="coerce")

    out = (
        d.groupby(["method", "lob", "valuation_pay_year"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(pd.to_numeric(s, errors="coerce").fillna(0.0)))))),
            bias=("error", "mean"),
            mape=("pct_error", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
            n_obs=("AY", "count"),
        )
        .sort_values(["valuation_pay_year", "method"])
        .reset_index(drop=True)
    )
    return out


def _prepare_claim_level_cumulative_output(
    output_df: pd.DataFrame,
    claim_type: Optional[int],
    scale_divisor: float,
) -> tuple[pd.DataFrame, list[str], int]:
    d = output_df.copy()
    d["AY"] = pd.to_numeric(d.get("AY"), errors="coerce")
    d["Type"] = pd.to_numeric(d.get("Type"), errors="coerce")
    d = d.loc[d["AY"].notna()].copy()

    if claim_type is not None:
        d = d.loc[d["Type"] == float(int(claim_type))].copy()

    pay_cols = pay_columns_from_output(d)
    max_dev = int(max(int(str(c).replace("Pay", "")) for c in pay_cols))
    for c in pay_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0) / float(scale_divisor)

    running = np.zeros(len(d), dtype=float)
    c_cols: list[str] = []
    for c in pay_cols:
        dy = int(str(c).replace("Pay", ""))
        running = running + pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        cname = f"C{dy}"
        d[cname] = running
        c_cols.append(cname)

    d["AY"] = pd.to_numeric(d["AY"], errors="coerce").astype(int)
    d = d.reset_index(drop=True)
    return d, c_cols, max_dev


def _encode_claim_level_features(d: pd.DataFrame) -> pd.DataFrame:
    age = pd.to_numeric(d.get("Age"), errors="coerce")
    if age.notna().any():
        age_min = float(age.min())
        age_max = float(age.max())
        if age_max > age_min:
            age_scaled = 2.0 * (age - age_min) / (age_max - age_min) - 1.0
        else:
            age_scaled = pd.Series(0.0, index=d.index, dtype=float)
    else:
        age_scaled = pd.Series(0.0, index=d.index, dtype=float)

    X = pd.DataFrame({"age_transformed": pd.to_numeric(age_scaled, errors="coerce").fillna(0.0)}, index=d.index)

    cat_cols = [c for c in ["Type", "AQ", "cc", "inj_part"] if c in d.columns]
    for c in cat_cols:
        v = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype(int)
        dum = pd.get_dummies(v, prefix=c, dtype=float)
        X = pd.concat([X, dum], axis=1)

    if X.shape[1] == 0:
        X["const"] = 1.0
    return X.reset_index(drop=True)


def _predict_claim_chainladder_style_for_valuation(
    claims_cum: pd.DataFrame,
    X_all: pd.DataFrame,
    c_cols: Sequence[str],
    valuation_year: int,
    claim_type: Optional[int],
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    random_state: int,
    min_train_rows: int,
    method_name: str,
) -> pd.DataFrame:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for R-style triangle DL replication. Install with: pip install tensorflow"
        ) from exc

    d_raw = claims_cum.copy().reset_index(drop=True)
    for c in ["AY", *list(c_cols)]:
        d_raw[c] = pd.to_numeric(d_raw.get(c), errors="coerce")

    if len(c_cols) > 0:
        nonneg_mask = d_raw[list(c_cols)].min(axis=1, skipna=True) >= 0.0
        d_raw = d_raw.loc[nonneg_mask].copy()

    d_raw = d_raw.loc[pd.to_numeric(d_raw[c_cols[-1]], errors="coerce").fillna(0.0) > 0.0].copy()
    if d_raw.empty:
        return pd.DataFrame(columns=["AY", "latest_observed", "ultimate", "pred_reserve", "method"])

    if "RepDel" in d_raw.columns:
        rep_delay = pd.to_numeric(d_raw["RepDel"], errors="coerce")
    elif "RepDelayYears" in d_raw.columns:
        rep_delay = pd.to_numeric(d_raw["RepDelayYears"], errors="coerce")
    elif "RepAY" in d_raw.columns:
        rep_delay = pd.to_numeric(d_raw["RepAY"], errors="coerce") - pd.to_numeric(d_raw["AY"], errors="coerce")
    else:
        rep_delay = pd.Series(0.0, index=d_raw.index, dtype=float)
    rep_delay = pd.to_numeric(rep_delay, errors="coerce").fillna(0.0)

    reported_mask = (pd.to_numeric(d_raw["AY"], errors="coerce").fillna(-1).astype(int) + np.floor(rep_delay).astype(int)) <= int(valuation_year)
    d_source = d_raw.loc[reported_mask].copy().reset_index(drop=True)
    if d_source.empty:
        return pd.DataFrame(columns=["AY", "latest_observed", "ultimate", "pred_reserve", "method"])

    X_source = _encode_claim_level_features(d_source)
    d_work = d_source.copy().reset_index(drop=True)
    X = X_source.copy().reset_index(drop=True)

    lob_col = "__lob_key__"
    if claim_type is None and "Type" in d_work.columns:
        d_work[lob_col] = pd.to_numeric(d_work["Type"], errors="coerce").fillna(0).astype(int)
        d_source[lob_col] = pd.to_numeric(d_source["Type"], errors="coerce").fillna(0).astype(int)
    else:
        type_val = int(claim_type) if claim_type is not None else 1
        d_work[lob_col] = type_val
        d_source[lob_col] = type_val

    ay = pd.to_numeric(d_work["AY"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)

    for j, c in enumerate(c_cols):
        m_future = ay + int(j) > int(valuation_year)
        if bool(np.any(m_future)):
            d_work.loc[m_future, c] = 0.0

    observed_snapshot = d_work[["AY", lob_col, *list(c_cols)]].copy()

    hidden_arch = [int(u) for u in hidden_layer_sizes if int(u) > 0]
    if len(hidden_arch) == 0:
        hidden_arch = [20]
    l2_penalty = float(max(alpha, 0.0))
    lr = float(max(learning_rate_init, 1e-6))
    min_rows_nonzero = int(max(min_train_rows, 10))

    for j in range(1, len(c_cols)):
        prev_col = c_cols[j - 1]
        cur_col = c_cols[j]

        c_prev = pd.to_numeric(d_work[prev_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        c_cur = pd.to_numeric(d_work[cur_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        estimate = (c_prev > 0.0) & (ay >= int(valuation_year) - int(j) + 1)
        non_estimate = (c_prev > 0.0) & (ay < int(valuation_year) - int(j) + 1)

        if not bool(np.any(estimate)):
            continue

        if int(np.sum(non_estimate)) >= min_rows_nonzero:
            w_train = np.sqrt(np.clip(c_prev[non_estimate], 0.0, None)).reshape(-1, 1)
            w_est = np.sqrt(np.clip(c_prev[estimate], 0.0, None)).reshape(-1, 1)
            y_train = (c_cur[non_estimate] / np.clip(w_train.reshape(-1), 1e-9, None)).reshape(-1, 1)

            x_train = X.loc[non_estimate, :].to_numpy(dtype=np.float32)
            x_est = X.loc[estimate, :].to_numpy(dtype=np.float32)

            keras.backend.clear_session()
            tf.keras.utils.set_random_seed(int(random_state + j))

            feat_in = keras.Input(shape=(int(x_train.shape[1]),), name="features")
            net = feat_in
            for units in hidden_arch:
                net = layers.Dense(
                    units=int(units),
                    activation="tanh",
                    kernel_regularizer=keras.regularizers.l2(l2_penalty) if l2_penalty > 0.0 else None,
                )(net)
                net = layers.Dropout(rate=0.30)(net)
            net = layers.Dense(units=1, activation="exponential")(net)

            vol_in = keras.Input(shape=(1,), name="volumes")
            offset = layers.Dense(
                units=1,
                activation="linear",
                use_bias=False,
                trainable=False,
                kernel_initializer=keras.initializers.Ones(),
            )(vol_in)

            merged = layers.Multiply()([net, offset])
            model = keras.Model(inputs=[feat_in, vol_in], outputs=merged)
            model.compile(
                loss="mse",
                optimizer=keras.optimizers.RMSprop(learning_rate=lr),
            )

            fit_kwargs = {
                "x": [x_train, w_train.astype(np.float32)],
                "y": y_train.astype(np.float32),
                "epochs": int(max_iter),
                "batch_size": int(min(10000, max(32, len(y_train)))),
                "verbose": 0,
            }
            if len(y_train) >= 20:
                fit_kwargs["validation_split"] = 0.1
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=12,
                    restore_best_weights=True,
                )
            ] if len(y_train) >= 20 else []

            model.fit(**fit_kwargs, callbacks=callbacks)

            pred_raw = model.predict([x_est, w_est.astype(np.float32)], verbose=0).reshape(-1)
            pred = pred_raw * w_est.reshape(-1)
            pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
            d_work.loc[estimate, cur_col] = pred
        else:
            if int(np.sum(non_estimate)) > 0:
                ratio_fb = float(np.sum(c_cur[non_estimate]) / max(np.sum(c_prev[non_estimate]), 1e-9))
            else:
                ratio_fb = 1.0
            d_work.loc[estimate, cur_col] = np.clip(c_prev[estimate] * ratio_fb, 0.0, None)

    nonzero_cum = (
        d_work[["AY", lob_col, *list(c_cols)]]
        .copy()
        .groupby(["AY", lob_col], as_index=True)
        .sum()
        .sort_index()
    )
    if nonzero_cum.empty:
        return pd.DataFrame(columns=["AY", "latest_observed", "ultimate", "pred_reserve", "method"])

    max_dev = int(len(c_cols) - 1)
    eval_ays = sorted({int(a) for a in pd.to_numeric(d_work["AY"], errors="coerce").dropna().astype(int).tolist() if int(a) <= int(valuation_year)})
    lob_values = sorted({int(l) for l in pd.to_numeric(d_work[lob_col], errors="coerce").dropna().astype(int).tolist()})

    zero_cum = nonzero_cum.copy()
    zero_cum.loc[:, :] = 0.0

    d_hist = d_source[["AY", lob_col, *list(c_cols)]].copy()
    d_hist["AY"] = pd.to_numeric(d_hist["AY"], errors="coerce").fillna(-1).astype(int)
    d_hist[lob_col] = pd.to_numeric(d_hist[lob_col], errors="coerce").fillna(0).astype(int)
    for c in c_cols:
        d_hist[c] = pd.to_numeric(d_hist[c], errors="coerce").fillna(0.0)

    for lob in lob_values:
        df_help = d_hist.loc[d_hist[lob_col] == int(lob)].copy()
        if df_help.empty:
            continue

        for ay_eval in eval_ays:
            idx = (int(ay_eval), int(lob))
            if idx not in nonzero_cum.index:
                continue

            latest_dev = int(np.clip(int(valuation_year) - int(ay_eval), 0, max_dev))
            if latest_dev >= max_dev:
                continue

            prev_col = c_cols[latest_dev]
            next_col = c_cols[latest_dev + 1]

            hist = df_help.loc[df_help["AY"] < int(ay_eval)].copy()
            if hist.empty:
                continue

            den1 = float(pd.to_numeric(hist[prev_col], errors="coerce").fillna(0.0).sum())
            x_zero = hist.loc[pd.to_numeric(hist[prev_col], errors="coerce").fillna(0.0) <= 1e-12].copy()
            num1 = float(pd.to_numeric(x_zero[next_col], errors="coerce").fillna(0.0).sum())

            g1 = float(num1 / den1) if den1 > 0.0 else 0.0
            cf_zero = float(pd.to_numeric(nonzero_cum.loc[idx, next_col], errors="coerce")) * g1
            zero_cum.loc[idx, next_col] = cf_zero

            for dev in range(latest_dev + 2, max_dev + 1):
                obs = x_zero.loc[x_zero["AY"] < int(valuation_year) - int(dev) + 1].copy()
                den = float(pd.to_numeric(obs[c_cols[dev - 1]], errors="coerce").fillna(0.0).sum())
                num = float(pd.to_numeric(obs[c_cols[dev]], errors="coerce").fillna(0.0).sum())
                g = float(num / den) if den > 0.0 else 1.0
                cf_zero = cf_zero * g
                zero_cum.loc[idx, c_cols[dev]] = cf_zero

    total_cum = (nonzero_cum + zero_cum).sort_index()

    latest_claim = observed_snapshot[list(c_cols)].max(axis=1)
    latest_by_ay_lob = (
        pd.DataFrame(
            {
                "AY": pd.to_numeric(observed_snapshot["AY"], errors="coerce").astype(int),
                "lob_key": pd.to_numeric(observed_snapshot[lob_col], errors="coerce").astype(int),
                "latest_observed_claim": pd.to_numeric(latest_claim, errors="coerce").fillna(0.0),
            }
        )
        .groupby(["AY", "lob_key"], as_index=False)["latest_observed_claim"]
        .sum()
        .rename(columns={"latest_observed_claim": "latest_observed"})
    )

    ult_by_ay_lob = (
        total_cum[[c_cols[-1]]]
        .reset_index()
        .rename(columns={c_cols[-1]: "ultimate", lob_col: "lob_key"})
    )
    ult_by_ay_lob["AY"] = pd.to_numeric(ult_by_ay_lob["AY"], errors="coerce").astype(int)
    ult_by_ay_lob["lob_key"] = pd.to_numeric(ult_by_ay_lob["lob_key"], errors="coerce").astype(int)

    out_per_lob = latest_by_ay_lob.merge(ult_by_ay_lob, on=["AY", "lob_key"], how="inner")
    out_per_lob = out_per_lob.loc[pd.to_numeric(out_per_lob["AY"], errors="coerce") <= int(valuation_year)].copy()

    if claim_type is None:
        distinct_lobs = out_per_lob["lob_key"].nunique()
        if distinct_lobs > 1:
            out_portfolio = (
                out_per_lob.groupby("AY", as_index=False)[["latest_observed", "ultimate"]].sum()
            )
            out_portfolio["lob_key"] = pd.NA
            out_per_lob_obj = out_per_lob.astype({"lob_key": "object"})
            out = pd.concat([out_per_lob_obj, out_portfolio], ignore_index=True)
        else:
            out = out_per_lob.copy()
            out["lob_key"] = pd.NA
    else:
        out = out_per_lob

    out["pred_reserve"] = (
        pd.to_numeric(out["ultimate"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["latest_observed"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    out["method"] = str(method_name)
    out = out.sort_values(["lob_key", "AY"], na_position="last").reset_index(drop=True)
    return out


def run_triangle_dl_rolling_origin_backtest(
    output_df: pd.DataFrame,
    valuation_years: Iterable[int],
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
    lookback_valuations: int = 4,
    min_train_rows: int = 80,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    alpha: float = 2e-4,
    learning_rate_init: float = 8e-4,
    max_iter: int = 320,
    random_state: int = 42,
    factor_floor: float = 1.0,
    factor_cap: float = 3.0,
    credibility_k: float = 25.0,
    eval_recent_ay_window: Optional[int] = None,
    method_name: str = "DL-Triangle-LinkRatio-NN",
    fallback_safeguard_ratio: float = 1.10,
    include_current_valuation_in_training: bool = True,
    use_chain_ladder_guardrail: bool = False,
    cl_guardrail_lower: float = 0.95,
    cl_guardrail_upper: float = 1.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin NN benchmark using claim-level chain-ladder-style training logic.

    The extra keyword parameters are kept for notebook-call compatibility.
    """

    vals = sorted({int(v) for v in valuation_years})

    if claim_type is None:
        types_available = sorted(
            int(t) for t in pd.to_numeric(output_df.get("Type"), errors="coerce").dropna().unique()
        )
        truth_frames = []
        for t in types_available:
            s_t = true_ultimate_by_ay_from_output(
                output_df=output_df,
                claim_type=int(t),
                scale_divisor=float(scale_divisor),
            )
            truth_frames.append(
                pd.DataFrame(
                    {
                        "AY": pd.to_numeric(s_t.index, errors="coerce"),
                        "lob_key": int(t),
                        "true_ultimate": pd.to_numeric(s_t.values, errors="coerce"),
                    }
                )
            )
        s_all = true_ultimate_by_ay_from_output(
            output_df=output_df,
            claim_type=None,
            scale_divisor=float(scale_divisor),
        )
        truth_all = pd.DataFrame(
            {
                "AY": pd.to_numeric(s_all.index, errors="coerce"),
                "lob_key": pd.NA,
                "true_ultimate": pd.to_numeric(s_all.values, errors="coerce"),
            }
        )
        truth_frames.append(truth_all)
        true_ult = pd.concat(truth_frames, ignore_index=True) if truth_frames else truth_all
    else:
        s = true_ultimate_by_ay_from_output(
            output_df=output_df,
            claim_type=claim_type,
            scale_divisor=float(scale_divisor),
        )
        true_ult = pd.DataFrame(
            {
                "AY": pd.to_numeric(s.index, errors="coerce"),
                "lob_key": int(claim_type),
                "true_ultimate": pd.to_numeric(s.values, errors="coerce"),
            }
        )

    claims_cum, c_cols, _ = _prepare_claim_level_cumulative_output(
        output_df=output_df,
        claim_type=claim_type,
        scale_divisor=float(scale_divisor),
    )
    if claims_cum.empty or len(c_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    X_all = _encode_claim_level_features(claims_cum)
    compare_rows: list[pd.DataFrame] = []

    for v in vals:
        pred_tbl = _predict_claim_chainladder_style_for_valuation(
            claims_cum=claims_cum,
            X_all=X_all,
            c_cols=c_cols,
            valuation_year=int(v),
            claim_type=claim_type,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=float(alpha),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            random_state=int(random_state),
            min_train_rows=int(min_train_rows),
            method_name=str(method_name),
        )
        if pred_tbl.empty:
            continue

        if eval_recent_ay_window is not None:
            pred_tbl = pred_tbl.loc[pred_tbl["AY"] > int(v) - int(eval_recent_ay_window)].copy()
            if pred_tbl.empty:
                continue

        cmp = evaluate_triangle_predictions(
            pred_tbl=pred_tbl,
            true_ultimate=true_ult,
            valuation_year=int(v),
            claim_type=claim_type,
        )
        if cmp.empty:
            continue

        compare_rows.append(cmp)

    compare_long = pd.concat(compare_rows, ignore_index=True) if compare_rows else pd.DataFrame()
    summary_long = summarize_triangle_compare(compare_long)
    return compare_long, summary_long
