from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class MethodResult:
    method: str
    reserve_by_ay: pd.Series
    ultimate_by_ay: pd.Series
    latest_by_ay: pd.Series
    full_incremental: pd.DataFrame


@dataclass
class BootstrapResult:
    method: str
    reserve_samples_by_ay: pd.DataFrame

    @property
    def reserve_samples_total(self) -> pd.Series:
        return self.reserve_samples_by_ay.sum(axis=1)


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _latest_observed_dev(row: pd.Series) -> int:
    obs = np.where(~row.isna().to_numpy())[0]
    if len(obs) == 0:
        raise ValueError("Each accident year row must contain at least one observed development value.")
    return int(obs[-1])


def _observed_cumulative(incremental_triangle: pd.DataFrame) -> pd.DataFrame:
    tri = _ensure_numeric(incremental_triangle)
    obs_mask = ~tri.isna()
    cum = tri.fillna(0.0).cumsum(axis=1)
    return cum.where(obs_mask)


def _estimate_odp_dispersion(observed: pd.DataFrame, fitted: pd.DataFrame, parameter_count: int) -> float:
    obs = _ensure_numeric(observed)
    fit = _ensure_numeric(fitted).reindex(index=obs.index, columns=obs.columns)

    mask = (~obs.isna()) & (~fit.isna())
    if not mask.to_numpy().any():
        return 1.0

    y = obs.where(mask).stack().to_numpy(dtype=float)
    mu = np.maximum(fit.where(mask).stack().to_numpy(dtype=float), 1e-9)
    pearson = np.square(y - mu) / mu
    dof = max(len(pearson) - int(parameter_count), 1)
    phi = float(pearson.sum() / dof)
    return phi if np.isfinite(phi) and phi > 0 else 1.0


def _gamma_process_draw(rng: np.random.Generator, mean: float, dispersion: float) -> float:
    m = max(float(mean), 0.0)
    phi = max(float(dispersion), 0.0)

    if m <= 0.0:
        return 0.0
    if phi <= 1e-12:
        return m

    shape = m / phi
    scale = phi
    return float(rng.gamma(shape=shape, scale=scale))


def _estimate_chain_ladder_dispersion(cumulative_triangle: pd.DataFrame, link_ratios: pd.Series) -> float:
    cum = _ensure_numeric(cumulative_triangle)
    dy_cols = list(cum.columns)

    pearson_terms = []
    for ay in cum.index:
        row = cum.loc[ay]
        k = _latest_observed_dev(row)
        for j in range(0, k):
            c_prev = float(row.iloc[j])
            c_next = float(row.iloc[j + 1])
            if not np.isfinite(c_prev) or not np.isfinite(c_next) or c_prev <= 0.0:
                continue

            mean_inc = max(c_prev * (float(link_ratios.loc[dy_cols[j]]) - 1.0), 0.0)
            act_inc = c_next - c_prev
            pearson_terms.append(np.square(act_inc - mean_inc) / max(c_prev, 1e-9))

    if not pearson_terms:
        return 1.0

    dof = max(len(pearson_terms) - len(link_ratios), 1)
    phi = float(np.sum(pearson_terms) / dof)
    return phi if np.isfinite(phi) and phi > 0 else 1.0


def _estimate_chain_ladder_dispersion_by_dev(
    cumulative_triangle: pd.DataFrame,
    link_ratios: pd.Series,
) -> pd.Series:
    cum = _ensure_numeric(cumulative_triangle)
    dy_cols = list(cum.columns)

    out = {}
    fallback = _estimate_chain_ladder_dispersion(cum, link_ratios)

    for j, dev in enumerate(dy_cols[:-1]):
        vals = []
        for ay in cum.index:
            row = cum.loc[ay]
            k = _latest_observed_dev(row)
            if j > k - 1:
                continue

            c_prev = float(row.iloc[j])
            c_next = float(row.iloc[j + 1])
            if not np.isfinite(c_prev) or not np.isfinite(c_next) or c_prev <= 0.0:
                continue

            mean_inc = max(c_prev * (float(link_ratios.loc[dev]) - 1.0), 0.0)
            act_inc = c_next - c_prev
            term = np.square(act_inc - mean_inc) / max(c_prev, 1e-9)
            if np.isfinite(term):
                vals.append(float(term))

        if len(vals) >= 2:
            phi_j = float(np.sum(vals) / max(len(vals) - 1, 1))
            out[dev] = phi_j if np.isfinite(phi_j) and phi_j > 0 else fallback
        elif len(vals) == 1:
            out[dev] = max(vals[0], 1e-9)
        else:
            out[dev] = fallback

    return pd.Series(out, name="phi_by_dev")


def incremental_to_cumulative(incremental_triangle: pd.DataFrame) -> pd.DataFrame:
    tri = _ensure_numeric(incremental_triangle)
    return tri.cumsum(axis=1)


def cumulative_to_incremental(cumulative_triangle: pd.DataFrame) -> pd.DataFrame:
    tri = _ensure_numeric(cumulative_triangle)
    shifted = tri.shift(axis=1).fillna(0.0)
    return tri - shifted


def build_incremental_triangle(
    df: pd.DataFrame,
    accident_year_col: str,
    development_year_col: str,
    amount_col: str,
    lob_col: Optional[str] = None,
    lob_value: Optional[str] = None,
    max_development_year: Optional[int] = None,
) -> pd.DataFrame:
    """Builds an incremental paid triangle from transaction-level data.

    The function expects one row per payment (or at least payment aggregate that can be summed)
    with explicit accident and development year columns.
    """

    data = df.copy()
    if lob_col is not None and lob_value is not None:
        data = data.loc[data[lob_col] == lob_value].copy()

    required = [accident_year_col, development_year_col, amount_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns for triangle build: {missing}")

    data[accident_year_col] = pd.to_numeric(data[accident_year_col], errors="coerce").astype("Int64")
    data[development_year_col] = pd.to_numeric(data[development_year_col], errors="coerce").astype("Int64")
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce")

    data = data.dropna(subset=[accident_year_col, development_year_col, amount_col])

    if max_development_year is not None:
        data = data.loc[data[development_year_col] <= max_development_year].copy()

    tri = (
        data.groupby([accident_year_col, development_year_col], as_index=False)[amount_col]
        .sum()
        .pivot(index=accident_year_col, columns=development_year_col, values=amount_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    tri.index.name = "AY"
    tri.columns.name = "DY"
    tri = _ensure_numeric(tri)

    return tri


def selected_link_ratios(cumulative_triangle: pd.DataFrame) -> pd.Series:
    cum = _ensure_numeric(cumulative_triangle)
    j = cum.shape[1]
    factors = []

    for col in range(j - 1):
        c0 = cum.iloc[:, col]
        c1 = cum.iloc[:, col + 1]
        mask = (~c0.isna()) & (~c1.isna()) & (c0 > 0)

        if mask.sum() == 0:
            factors.append(1.0)
        else:
            factors.append(float(c1[mask].sum() / c0[mask].sum()))

    out = pd.Series(factors, index=cum.columns[:-1], name="f_j")
    return out


def cdf_from_link_ratios(link_ratios: pd.Series, final_dev_label: int) -> pd.Series:
    cdf_values: Dict[int, float] = {int(final_dev_label): 1.0}
    labels = list(link_ratios.index)

    running = 1.0
    for dev in reversed(labels):
        running *= float(link_ratios.loc[dev])
        cdf_values[int(dev)] = running

    return pd.Series(cdf_values).sort_index().rename("CDF_to_ultimate")


def _leading_zero_count(values: pd.Series) -> int:
    count = 0
    for v in values:
        if pd.isna(v):
            break
        if float(v) == 0.0:
            count += 1
        else:
            break
    return count


def fit_chain_ladder(
    incremental_triangle: pd.DataFrame,
    drop_first_ay_if_initial_zeros: bool = True,
    initial_zero_count_threshold: int = 2,
) -> MethodResult:
    """Chain-Ladder with ENTER-style assumptions.

    Assumptions mirrored from the ENTER tool guidance:
    - empty cells are treated as zero (not blank)
    - optional omission of first AY if it starts with several zero development cells
    """

    tri = _ensure_numeric(incremental_triangle)

    if tri.shape[0] < 2 or tri.shape[1] < 2:
        raise ValueError("Triangle must have at least 2 AY and 2 DY for Chain-Ladder.")

    if drop_first_ay_if_initial_zeros:
        first_ay = tri.index[0]
        zero_count = _leading_zero_count(tri.loc[first_ay])
        if zero_count >= initial_zero_count_threshold and tri.shape[0] > 2:
            tri = tri.iloc[1:, :].copy()

    cum = _observed_cumulative(tri)
    n_cols = cum.shape[1]

    factors = []
    for c in range(n_cols - 1):
        c0 = cum.iloc[:, c]
        c1 = cum.iloc[:, c + 1]
        mask = (~c0.isna()) & (~c1.isna()) & (c0 > 0)

        if mask.sum() == 0:
            factors.append(1.0)
        else:
            factors.append(float(c1[mask].sum() / c0[mask].sum()))

    link = pd.Series(factors, index=cum.columns[:-1], name="f_j")

    full_cum = cum.copy()
    for ay in full_cum.index:
        k = _latest_observed_dev(cum.loc[ay])
        for c in range(k + 1, n_cols):
            prev_col = full_cum.columns[c - 1]
            f = float(link.loc[prev_col])
            full_cum.loc[ay, full_cum.columns[c]] = float(full_cum.loc[ay, prev_col]) * f

    latest = pd.Series(
        {ay: float(cum.loc[ay].iloc[_latest_observed_dev(cum.loc[ay])]) for ay in cum.index},
        name="latest_observed_cumulative",
    )
    ultimate = full_cum.iloc[:, -1].rename("ultimate")
    reserve = (ultimate - latest).rename("reserve")

    full_inc = cumulative_to_incremental(full_cum)

    return MethodResult(
        method="Chain-Ladder",
        reserve_by_ay=reserve,
        ultimate_by_ay=ultimate,
        latest_by_ay=latest,
        full_incremental=full_inc,
    )


def fit_bornhuetter_ferguson(
    incremental_triangle: pd.DataFrame,
    apriori_ultimate_by_ay: pd.Series,
    method_name: str = "Bornhuetter-Ferguson",
) -> MethodResult:
    tri = _ensure_numeric(incremental_triangle)
    cum = incremental_to_cumulative(tri)

    f = selected_link_ratios(cum)
    cdf = cdf_from_link_ratios(f, final_dev_label=int(cum.columns[-1]))
    reported = (1.0 / cdf).rename("reported_pct")

    apriori = apriori_ultimate_by_ay.reindex(cum.index)
    if apriori.isna().any():
        missing = list(apriori[apriori.isna()].index)
        raise ValueError(
            "Missing apriori ultimate for AY values: "
            f"{missing}. Provide apriori_ultimate_by_ay for all AY rows."
        )

    full_cum = cum.copy()
    latest = pd.Series(index=cum.index, dtype=float, name="latest_observed_cumulative")
    ultimate = pd.Series(index=cum.index, dtype=float, name="ultimate")

    dev_cols = list(cum.columns)

    for ay in cum.index:
        row = cum.loc[ay]
        k = _latest_observed_dev(row)
        k_dev = int(dev_cols[k])

        latest_ay = float(row.iloc[k])
        latest.loc[ay] = latest_ay

        rep_k = float(reported.loc[k_dev])
        apriori_ay = float(apriori.loc[ay])

        reserve_ay = (1.0 - rep_k) * apriori_ay
        ultimate_ay = latest_ay + reserve_ay
        ultimate.loc[ay] = ultimate_ay

        for c in range(k + 1, len(dev_cols)):
            d = int(dev_cols[c])
            rep_d = float(reported.loc[d])
            full_cum.loc[ay, d] = latest_ay + (rep_d - rep_k) * apriori_ay

    full_inc = cumulative_to_incremental(full_cum)
    reserve = (ultimate - latest).rename("reserve")

    return MethodResult(
        method=method_name,
        reserve_by_ay=reserve,
        ultimate_by_ay=ultimate,
        latest_by_ay=latest,
        full_incremental=full_inc,
    )


def fit_cape_cod(
    incremental_triangle: pd.DataFrame,
    exposure_by_ay: pd.Series,
    mature_reported_threshold: float = 0.70,
) -> MethodResult:
    """Cape-Cod using AY exposure and chain-ladder reported proportions.

    The implementation estimates a single ELR on mature AYs, then sets
    prior ultimate_i = exposure_i * ELR_hat and applies BF mechanics.
    """

    tri = _ensure_numeric(incremental_triangle)
    cum = incremental_to_cumulative(tri)

    f = selected_link_ratios(cum)
    cdf = cdf_from_link_ratios(f, final_dev_label=int(cum.columns[-1]))

    dev_cols = list(cum.columns)
    latest = pd.Series(index=cum.index, dtype=float)
    reported = pd.Series(index=cum.index, dtype=float)

    for ay in cum.index:
        row = cum.loc[ay]
        k = _latest_observed_dev(row)
        k_dev = int(dev_cols[k])
        latest.loc[ay] = float(row.iloc[k])
        reported.loc[ay] = 1.0 / float(cdf.loc[k_dev])

    exposure = pd.to_numeric(exposure_by_ay.reindex(cum.index), errors="coerce")
    if exposure.isna().any():
        missing = list(exposure[exposure.isna()].index)
        raise ValueError(f"Missing exposure_by_ay values for AY: {missing}")
    if (exposure <= 0).any():
        bad = list(exposure[exposure <= 0].index)
        raise ValueError(f"Exposure values must be strictly positive. Invalid AY: {bad}")

    mature_mask = (reported >= mature_reported_threshold) & (exposure > 0)
    if mature_mask.sum() == 0:
        mature_mask = exposure > 0

    denom = float((exposure[mature_mask] * reported[mature_mask]).sum())
    if denom <= 0:
        raise ValueError("Cannot estimate Cape-Cod ELR: denominator <= 0.")

    numer = float(latest[mature_mask].sum())
    elr_hat = numer / denom

    apriori = (exposure * elr_hat).rename("apriori_ultimate")
    return fit_bornhuetter_ferguson(
        incremental_triangle=tri,
        apriori_ultimate_by_ay=apriori,
        method_name="Cape-Cod",
    )


def _fit_glm_odp_components(
    incremental_triangle: pd.DataFrame,
    include_calendar_trend: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, float]:
    tri = _ensure_numeric(incremental_triangle)
    ay_labels = list(tri.index)
    dy_labels = list(tri.columns)

    obs_records = []
    full_records = []
    for ay in ay_labels:
        for dy in dy_labels:
            y = tri.loc[ay, dy]
            cy = int(ay) + int(dy)
            full_records.append({"AY": str(ay), "DY": str(dy), "CY": float(cy), "y": y})
            if not pd.isna(y):
                obs_records.append({"AY": str(ay), "DY": str(dy), "CY": float(cy), "y": float(y)})

    obs_df = pd.DataFrame(obs_records)
    full_df = pd.DataFrame(full_records)

    if (obs_df["y"] < 0).any():
        raise ValueError("GLM ODP requires non-negative incremental payments in observed cells.")

    # Parsimonious GLM design for yearly data to avoid CL-equivalent saturation.
    X_obs = pd.get_dummies(obs_df[["AY"]], drop_first=True, dtype=float)
    X_full = pd.get_dummies(full_df[["AY"]], drop_first=True, dtype=float)
    X_full = X_full.reindex(columns=X_obs.columns, fill_value=0)

    dy_mean = float(obs_df["DY"].astype(float).mean())
    dy_std = float(obs_df["DY"].astype(float).std(ddof=0))
    if dy_std <= 1e-12:
        dy_std = 1.0
    X_obs["DY_lin"] = (obs_df["DY"].astype(float) - dy_mean) / dy_std
    X_full["DY_lin"] = (full_df["DY"].astype(float) - dy_mean) / dy_std

    add_cy = include_calendar_trend and obs_df["CY"].nunique() > 1
    if add_cy:
        cy_mean = float(obs_df["CY"].mean())
        cy_std = float(obs_df["CY"].std(ddof=0))
        if cy_std <= 1e-12:
            cy_std = 1.0
        X_obs["CY_lin"] = (obs_df["CY"] - cy_mean) / cy_std
        X_full["CY_lin"] = (full_df["CY"] - cy_mean) / cy_std

    y_obs = pd.to_numeric(obs_df["y"], errors="coerce").astype(float)
    X_obs_const = sm.add_constant(X_obs, has_constant="add").astype(float)
    X_full_const = sm.add_constant(X_full, has_constant="add").astype(float)

    model = sm.GLM(y_obs, X_obs_const, family=sm.families.Poisson())
    try:
        fit = model.fit(scale="X2")
    except ValueError:
        # Fallback without calendar trend if the richer design is numerically unstable.
        if add_cy and "CY_lin" in X_obs.columns:
            X_obs = X_obs.drop(columns=["CY_lin"])
            X_full = X_full.drop(columns=["CY_lin"])
            X_obs_const = sm.add_constant(X_obs, has_constant="add").astype(float)
            X_full_const = sm.add_constant(X_full, has_constant="add").astype(float)
            model = sm.GLM(y_obs, X_obs_const, family=sm.families.Poisson())
            fit = model.fit(scale="X2")
        else:
            raise
    phi = float(fit.scale) if np.isfinite(float(fit.scale)) and float(fit.scale) > 0 else 1.0

    mu_full = fit.predict(X_full_const)
    full_df["mu"] = np.maximum(mu_full, 0.0)

    full_tri = (
        full_df.pivot(index="AY", columns="DY", values="mu")
        .reindex(index=[str(x) for x in ay_labels], columns=[str(x) for x in dy_labels])
        .astype(float)
    )
    full_tri.index = ay_labels
    full_tri.columns = dy_labels

    latest = tri.apply(lambda r: float(r.dropna().sum()), axis=1).rename("latest_observed_cumulative")
    pred_future = pd.Series(index=ay_labels, dtype=float)

    for ay in ay_labels:
        fut_mask = tri.loc[ay].isna()
        pred_future.loc[ay] = float(full_tri.loc[ay, fut_mask].sum()) if fut_mask.any() else 0.0

    reserve = pred_future.rename("reserve")
    ultimate = (latest + reserve).rename("ultimate")

    return full_tri, latest, reserve, ultimate, tri, phi


def fit_glm_odp(incremental_triangle: pd.DataFrame) -> MethodResult:
    full_tri, latest, reserve, ultimate, _, _ = _fit_glm_odp_components(
        incremental_triangle,
        include_calendar_trend=True,
    )

    return MethodResult(
        method="GLM-ODP",
        reserve_by_ay=reserve,
        ultimate_by_ay=ultimate,
        latest_by_ay=latest,
        full_incremental=full_tri,
    )


def run_benchmark_point_estimates(
    incremental_triangle: pd.DataFrame,
    exposure_by_ay: Optional[pd.Series] = None,
    mature_reported_threshold: float = 0.70,
) -> pd.DataFrame:
    cl = fit_chain_ladder(incremental_triangle)
    glm = fit_glm_odp(incremental_triangle)

    results = [cl]

    if exposure_by_ay is not None:
        cc = fit_cape_cod(
            incremental_triangle=incremental_triangle,
            exposure_by_ay=exposure_by_ay,
            mature_reported_threshold=mature_reported_threshold,
        )
        results.append(cc)

    results.append(glm)

    rows = []
    for res in results:
        df = pd.DataFrame(
            {
                "method": res.method,
                "AY": res.reserve_by_ay.index,
                "latest_observed": res.latest_by_ay.values,
                "ultimate": res.ultimate_by_ay.values,
                "reserve": res.reserve_by_ay.values,
            }
        )
        rows.append(df)

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out.sort_values(["method", "AY"]).reset_index(drop=True)

def _bootstrap_single_method(
    method_name: str,
    fit_fn: Callable[[pd.DataFrame], MethodResult],
    base_triangle: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> BootstrapResult:
    rng = np.random.default_rng(random_state)
    tri = _ensure_numeric(base_triangle)

    base_fit = fit_fn(tri)
    mu = base_fit.full_incremental.reindex(index=tri.index, columns=tri.columns)

    obs_mask = ~tri.isna()
    y_obs = tri.where(obs_mask)
    mu_obs = mu.where(obs_mask).clip(lower=1e-9)
    phi_base = _estimate_odp_dispersion(y_obs, mu_obs, parameter_count=mu_obs.shape[0] + mu_obs.shape[1] - 1)

    resid = ((y_obs - mu_obs) / np.sqrt(mu_obs)).stack().dropna().to_numpy()
    if len(resid) == 0:
        raise ValueError("No observed cells available for bootstrap.")

    mu_stacked = mu_obs.stack().dropna()
    mu_vec = mu_stacked.to_numpy()
    mu_index = mu_stacked.index

    samples = []
    for _ in range(n_bootstrap):
        sampled_resid = rng.choice(resid, size=mu_vec.size, replace=True)

        pseudo = tri.copy()
        y_star = np.maximum(mu_vec + sampled_resid * np.sqrt(mu_vec), 0.0)

        pseudo_vals = pd.Series(y_star, index=mu_index)
        for (ay, dy), val in pseudo_vals.items():
            pseudo.loc[ay, dy] = float(val)

        fit_star = fit_fn(pseudo)
        sim_full = fit_star.full_incremental.copy()
        fut_mask = ~obs_mask.reindex(index=sim_full.index, columns=sim_full.columns)
        phi_star = _estimate_odp_dispersion(
            pseudo.where(obs_mask),
            fit_star.full_incremental.reindex(index=tri.index, columns=tri.columns).where(obs_mask),
            parameter_count=mu_obs.shape[0] + mu_obs.shape[1] - 1,
        )

        reserve_star = []
        for ay in sim_full.index:
            for d in sim_full.columns:
                if bool(fut_mask.loc[ay, d]):
                    sim_full.loc[ay, d] = _gamma_process_draw(rng, sim_full.loc[ay, d], phi_star)
            reserve_star.append(float(sim_full.loc[ay, fut_mask.loc[ay]].sum()))

        samples.append(pd.Series(reserve_star, index=sim_full.index))

    sample_df = pd.DataFrame(samples).reset_index(drop=True)
    sample_df.columns = tri.index

    return BootstrapResult(method=method_name, reserve_samples_by_ay=sample_df)


def _bootstrap_chain_ladder(
    base_triangle: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> BootstrapResult:
    """Chain-Ladder bootstrap:
    1) fit CL to observed upper triangle
    2) compute residuals on observed development transitions
    3) resample residuals and rebuild pseudo upper triangle
    4) refit CL to pseudo upper triangle
    5) simulate future lower triangle process error
    """

    rng = np.random.default_rng(random_state)
    tri = _ensure_numeric(base_triangle)
    obs_mask = ~tri.isna()
    ay_index = tri.index
    dy_cols = list(tri.columns)

    # Base cumulative on observed cells only.
    cum_obs = _observed_cumulative(tri)

    # Selected development factors from observed transitions.
    f = selected_link_ratios(cum_obs)
    phi_base_by_dev = _estimate_chain_ladder_dispersion_by_dev(cum_obs, f)

    # Collect observed transition residuals (development j -> j+1).
    transition_cells = []
    residuals = []
    for ay in ay_index:
        row = tri.loc[ay]
        k = _latest_observed_dev(row)
        for j in range(0, k):
            c_prev = float(cum_obs.loc[ay, dy_cols[j]])
            c_next = float(cum_obs.loc[ay, dy_cols[j + 1]])
            mean_inc = max(c_prev * (float(f.loc[dy_cols[j]]) - 1.0), 0.0)
            act_inc = c_next - c_prev
            sd = np.sqrt(max(float(phi_base_by_dev.loc[dy_cols[j]]) * c_prev, 1e-9))
            residuals.append((act_inc - mean_inc) / sd)
            transition_cells.append((ay, j))

    residuals_arr = np.asarray(residuals, dtype=float)
    if residuals_arr.size == 0:
        raise ValueError("No observed Chain-Ladder transitions available for bootstrap.")
    residuals_arr = residuals_arr - np.nanmean(residuals_arr)

    samples = []
    for _ in range(n_bootstrap):
        sampled_resid = rng.choice(residuals_arr, size=len(transition_cells), replace=True)

        # Build pseudo observed cumulative upper triangle.
        pseudo_cum = cum_obs.copy()
        resid_idx = 0
        for ay in ay_index:
            row = tri.loc[ay]
            k = _latest_observed_dev(row)
            for j in range(0, k):
                d_prev = dy_cols[j]
                d_next = dy_cols[j + 1]
                c_prev = float(pseudo_cum.loc[ay, d_prev])
                mean_inc = max(c_prev * (float(f.loc[d_prev]) - 1.0), 0.0)
                sd = np.sqrt(max(float(phi_base_by_dev.loc[d_prev]) * c_prev, 1e-9))
                inc_star = max(mean_inc + float(sampled_resid[resid_idx]) * sd, 0.0)
                pseudo_cum.loc[ay, d_next] = c_prev + inc_star
                resid_idx += 1

        # Convert pseudo observed cumulative into pseudo observed incremental triangle.
        pseudo_tri = tri.copy()
        for ay in ay_index:
            row = tri.loc[ay]
            k = _latest_observed_dev(row)
            # first development remains as observed anchor
            for j in range(1, k + 1):
                d_cur = dy_cols[j]
                d_prev = dy_cols[j - 1]
                pseudo_tri.loc[ay, d_cur] = float(pseudo_cum.loc[ay, d_cur] - pseudo_cum.loc[ay, d_prev])

        fit_star = fit_chain_ladder(pseudo_tri)
        link_star = selected_link_ratios(_observed_cumulative(pseudo_tri))
        phi_star_by_dev = _estimate_chain_ladder_dispersion_by_dev(_observed_cumulative(pseudo_tri), link_star)

        # Simulate lower triangle process error around projected increments.
        sim_full = fit_star.full_incremental.copy()
        for ay in sim_full.index:
            for d in sim_full.columns:
                if pd.isna(tri.reindex(index=sim_full.index, columns=sim_full.columns).loc[ay, d]):
                    prev_dev = sim_full.columns[max(sim_full.columns.get_loc(d) - 1, 0)]
                    phi_d = float(phi_star_by_dev.loc[prev_dev]) if prev_dev in phi_star_by_dev.index else 1.0
                    sim_full.loc[ay, d] = _gamma_process_draw(rng, sim_full.loc[ay, d], phi_d)

        reserve_star = []
        for ay in sim_full.index:
            future_sum = float(sim_full.loc[ay, ~obs_mask.reindex(index=sim_full.index, columns=sim_full.columns).loc[ay]].sum())
            reserve_star.append(future_sum)

        samples.append(pd.Series(reserve_star, index=sim_full.index))

    sample_df = pd.DataFrame(samples).reset_index(drop=True)
    sample_df.columns = sample_df.columns.astype(ay_index.dtype, copy=False)
    sample_df = sample_df.reindex(columns=ay_index)

    return BootstrapResult(method="Chain-Ladder", reserve_samples_by_ay=sample_df)


def _bootstrap_glm_odp(
    base_triangle: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> BootstrapResult:
    """GLM-ODP bootstrap with aligned logic to Chain-Ladder:
    1) fit GLM-ODP to observed upper triangle
    2) resample Pearson residuals to create pseudo observed upper triangle
    3) refit GLM-ODP to pseudo upper triangle
    4) simulate lower triangle process error around projected increments
    """

    rng = np.random.default_rng(random_state)
    tri = _ensure_numeric(base_triangle)
    obs_mask = ~tri.isna()
    ay_index = tri.index

    base_mu_full, _, _, _, tri_base, phi_base = _fit_glm_odp_components(tri)
    y_obs = tri_base.where(obs_mask)
    mu_obs = base_mu_full.where(obs_mask).clip(lower=1e-9)

    denom = np.sqrt(np.maximum(phi_base * mu_obs, 1e-9))
    resid = ((y_obs - mu_obs) / denom).stack().dropna().to_numpy()
    if resid.size == 0:
        raise ValueError("No observed GLM-ODP cells available for bootstrap.")

    mu_stacked = mu_obs.stack().dropna()
    mu_vec = mu_stacked.to_numpy()
    mu_index = mu_stacked.index
    sd_vec = np.sqrt(np.maximum(phi_base * mu_vec, 1e-9))

    samples = []
    for _ in range(n_bootstrap):
        sampled_resid = rng.choice(resid, size=mu_vec.size, replace=True)

        pseudo = tri.copy()
        y_star = np.maximum(mu_vec + sampled_resid * sd_vec, 0.0)

        pseudo_vals = pd.Series(y_star, index=mu_index)
        for (ay, dy), val in pseudo_vals.items():
            pseudo.loc[ay, dy] = float(val)

        mu_star, _, _, _, _, phi_star = _fit_glm_odp_components(pseudo)

        # Add process simulation on future cells to align uncertainty decomposition with CL bootstrap.
        sim_full = mu_star.copy()
        fut_mask = ~obs_mask.reindex(index=sim_full.index, columns=sim_full.columns)
        for ay in sim_full.index:
            for d in sim_full.columns:
                if bool(fut_mask.loc[ay, d]):
                    sim_full.loc[ay, d] = _gamma_process_draw(rng, sim_full.loc[ay, d], phi_star)

        reserve_star = []
        for ay in sim_full.index:
            reserve_star.append(float(sim_full.loc[ay, fut_mask.loc[ay]].sum()))
        samples.append(pd.Series(reserve_star, index=sim_full.index))

    sample_df = pd.DataFrame(samples).reset_index(drop=True)
    sample_df.columns = sample_df.columns.astype(ay_index.dtype, copy=False)
    sample_df = sample_df.reindex(columns=ay_index)

    return BootstrapResult(method="GLM-ODP", reserve_samples_by_ay=sample_df)


def _generate_pseudo_triangle_via_chain_ladder(
    base_triangle: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series]:
    tri = _ensure_numeric(base_triangle)
    ay_index = tri.index
    dy_cols = list(tri.columns)
    cum_obs = _observed_cumulative(tri)
    f = selected_link_ratios(cum_obs)
    phi_by_dev = _estimate_chain_ladder_dispersion_by_dev(cum_obs, f)

    residuals = []
    for ay in ay_index:
        row = tri.loc[ay]
        k = _latest_observed_dev(row)
        for j in range(0, k):
            c_prev = float(cum_obs.loc[ay, dy_cols[j]])
            c_next = float(cum_obs.loc[ay, dy_cols[j + 1]])
            mean_inc = max(c_prev * (float(f.loc[dy_cols[j]]) - 1.0), 0.0)
            sd = np.sqrt(max(float(phi_by_dev.loc[dy_cols[j]]) * c_prev, 1e-9))
            act_inc = c_next - c_prev
            residuals.append((act_inc - mean_inc) / sd)

    resid = np.asarray(residuals, dtype=float)
    if resid.size == 0:
        return tri.copy(), phi_by_dev
    resid = resid - np.nanmean(resid)

    pseudo_cum = cum_obs.copy()
    for ay in ay_index:
        row = tri.loc[ay]
        k = _latest_observed_dev(row)
        for j in range(0, k):
            d_prev = dy_cols[j]
            d_next = dy_cols[j + 1]
            c_prev = float(pseudo_cum.loc[ay, d_prev])
            mean_inc = max(c_prev * (float(f.loc[d_prev]) - 1.0), 0.0)
            sd = np.sqrt(max(float(phi_by_dev.loc[d_prev]) * c_prev, 1e-9))
            e = float(rng.choice(resid))
            inc_star = max(mean_inc + e * sd, 0.0)
            pseudo_cum.loc[ay, d_next] = c_prev + inc_star

    pseudo_tri = tri.copy()
    for ay in ay_index:
        row = tri.loc[ay]
        k = _latest_observed_dev(row)
        for j in range(1, k + 1):
            d_cur = dy_cols[j]
            d_prev = dy_cols[j - 1]
            pseudo_tri.loc[ay, d_cur] = float(pseudo_cum.loc[ay, d_cur] - pseudo_cum.loc[ay, d_prev])

    return pseudo_tri, phi_by_dev


def _bootstrap_bf_like(
    method_name: str,
    fit_fn: Callable[[pd.DataFrame], MethodResult],
    base_triangle: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> BootstrapResult:
    """Bootstrap for BF/Cape-Cod style methods.

    Generic ODP-residual bootstrap collapses for BF/Cape-Cod because the fitted upper
    triangle equals the observed one by construction. We therefore generate pseudo
    observed upper triangles using Chain-Ladder transition residual resampling, refit
    the BF-like method, and then simulate lower-triangle process error around the
    method's projected future increments.
    """

    rng = np.random.default_rng(random_state)
    tri = _ensure_numeric(base_triangle)
    obs_mask = ~tri.isna()
    samples = []

    for _ in range(n_bootstrap):
        pseudo_tri, phi_by_dev = _generate_pseudo_triangle_via_chain_ladder(tri, rng)
        fit_star = fit_fn(pseudo_tri)
        sim_full = fit_star.full_incremental.copy()
        full_mask = obs_mask.reindex(index=sim_full.index, columns=sim_full.columns)

        for ay in sim_full.index:
            for d in sim_full.columns:
                if not bool(full_mask.loc[ay, d]):
                    pos = sim_full.columns.get_loc(d)
                    prev_dev = sim_full.columns[max(pos - 1, 0)]
                    phi_d = float(phi_by_dev.loc[prev_dev]) if prev_dev in phi_by_dev.index else float(phi_by_dev.mean())
                    sim_full.loc[ay, d] = _gamma_process_draw(rng, sim_full.loc[ay, d], phi_d)

        reserve_star = []
        for ay in sim_full.index:
            reserve_star.append(float(sim_full.loc[ay, ~full_mask.loc[ay]].sum()))
        samples.append(pd.Series(reserve_star, index=sim_full.index))

    sample_df = pd.DataFrame(samples).reset_index(drop=True)
    sample_df.columns = sample_df.columns.astype(tri.index.dtype, copy=False)
    sample_df = sample_df.reindex(columns=tri.index)
    return BootstrapResult(method=method_name, reserve_samples_by_ay=sample_df)


def run_bootstrap_uncertainty(
    incremental_triangle: pd.DataFrame,
    apriori_ultimate_by_ay: Optional[pd.Series] = None,
    exposure_by_ay: Optional[pd.Series] = None,
    mature_reported_threshold: float = 0.70,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Dict[str, BootstrapResult]:
    tri = _ensure_numeric(incremental_triangle)

    methods: Dict[str, Callable[[pd.DataFrame], MethodResult]] = {}

    if exposure_by_ay is not None:
        methods["Cape-Cod"] = lambda t: fit_cape_cod(
            t,
            exposure_by_ay=exposure_by_ay,
            mature_reported_threshold=mature_reported_threshold,
        )
    elif apriori_ultimate_by_ay is not None:
        methods["Bornhuetter-Ferguson"] = lambda t: fit_bornhuetter_ferguson(t, apriori_ultimate_by_ay)

    out: Dict[str, BootstrapResult] = {}
    out["Chain-Ladder"] = _bootstrap_chain_ladder(
        base_triangle=tri,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    out["GLM-ODP"] = _bootstrap_glm_odp(
        base_triangle=tri,
        n_bootstrap=n_bootstrap,
        random_state=random_state + 1,
    )

    for i, (name, fn) in enumerate(methods.items(), start=1):
        if name in {"Cape-Cod", "Bornhuetter-Ferguson"}:
            out[name] = _bootstrap_bf_like(
                method_name=name,
                fit_fn=fn,
                base_triangle=tri,
                n_bootstrap=n_bootstrap,
                random_state=random_state + 100 + i,
            )
        else:
            out[name] = _bootstrap_single_method(
                method_name=name,
                fit_fn=fn,
                base_triangle=tri,
                n_bootstrap=n_bootstrap,
                random_state=random_state + 100 + i,
            )
    return out


def summarize_bootstrap(
    bootstrap_results: Dict[str, BootstrapResult],
    quantiles: Iterable[float] = (0.05, 0.5, 0.95),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_ay_rows = []
    total_rows = []

    for method, b in bootstrap_results.items():
        ay_df = b.reserve_samples_by_ay
        total = b.reserve_samples_total

        for ay in ay_df.columns:
            s = ay_df[ay]
            row = {
                "method": method,
                "AY": ay,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
            }
            for q in quantiles:
                row[f"q{int(round(q * 100)):02d}"] = float(s.quantile(q))
            by_ay_rows.append(row)

        total_row = {
            "method": method,
            "mean": float(total.mean()),
            "std": float(total.std(ddof=1)),
        }
        for q in quantiles:
            total_row[f"q{int(round(q * 100)):02d}"] = float(total.quantile(q))
        total_rows.append(total_row)

    by_ay = pd.DataFrame(by_ay_rows).sort_values(["method", "AY"]).reset_index(drop=True)
    total_tbl = pd.DataFrame(total_rows).sort_values(["method"]).reset_index(drop=True)

    return by_ay, total_tbl
