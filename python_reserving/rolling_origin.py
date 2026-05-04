from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .benchmark_engine import run_benchmark_point_estimates
from .synthetic_loader import (
    build_incremental_triangle_from_output,
    compare_to_true_ultimate,
    error_summary_vs_true,
    true_ultimate_by_ay_from_claims,
    true_ultimate_by_ay_from_output,
)


def build_full_incremental_triangle_from_simulator(
    paid: pd.DataFrame,
    claims: pd.DataFrame,
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
) -> pd.DataFrame:
    required_paid = {"Id", "AY", "DY", "Paid"}
    required_claims = {"Id", "AY", "Type"}
    missing_paid = sorted(required_paid - set(paid.columns))
    missing_claims = sorted(required_claims - set(claims.columns))

    if missing_paid:
        raise KeyError(f"Missing required columns in paid: {missing_paid}")
    if missing_claims:
        raise KeyError(f"Missing required columns in claims: {missing_claims}")

    df = paid.merge(claims[["Id", "AY", "Type"]], on=["Id", "AY"], how="inner")
    if claim_type is not None:
        df = df.loc[df["Type"] == claim_type].copy()

    df["AY"] = pd.to_numeric(df["AY"], errors="coerce").astype("Int64")
    df["DY"] = pd.to_numeric(df["DY"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["AY", "DY", "Paid"]).copy()
    df = df.loc[df["DY"] >= 0].copy()

    tri = (
        df.groupby(["AY", "DY"], as_index=False)["Paid"]
        .sum()
        .pivot(index="AY", columns="DY", values="Paid")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    tri = tri / scale_divisor
    tri.index.name = "AY"
    tri.columns.name = "DY"
    return tri


def apply_valuation_cutoff(
    full_incremental_triangle: pd.DataFrame,
    valuation_pay_year: int,
) -> pd.DataFrame:
    tri = full_incremental_triangle.copy()

    for ay in tri.index:
        for dy in tri.columns:
            # Observed at valuation if payment calendar year <= valuation year
            # PayYear = AY + DY in this simulator setup.
            if int(ay) + int(dy) > int(valuation_pay_year):
                tri.loc[ay, dy] = np.nan

    tri = tri.loc[~tri.isna().all(axis=1), :]
    tri = tri.loc[:, ~tri.isna().all(axis=0)]
    return tri


def choose_valuation_years(
    full_incremental_triangle: pd.DataFrame,
    strategy: str = "even_spacing",
    n_points: int = 8,
    spacing: int = 3,
    min_observed_diagonals: int = 6,
    max_valuation_year: Optional[int] = None,
) -> List[int]:
    """Pick valuation years using reproducible rules rather than arbitrary lists.

    Strategies:
    - "all": every feasible valuation year from first usable year to max pay year.
    - "recent": last n_points valuation years.
    - "even_spacing": n_points evenly spaced valuation years.
    - "fixed_spacing": every `spacing` years.
    """

    if full_incremental_triangle.empty:
        return []

    min_ay = int(full_incremental_triangle.index.min())
    max_ay = int(full_incremental_triangle.index.max())

    # In Simulation.Machine setup, evaluation valuations should lie in observable
    # calendar years (typically up to latest AY), not to fully developed horizon.
    max_eval_year = int(max_ay if max_valuation_year is None else max_valuation_year)

    start_year = min_ay + max(1, int(min_observed_diagonals))
    if start_year > max_eval_year:
        return [max_eval_year]

    all_years = list(range(start_year, max_eval_year + 1))
    if not all_years:
        return []

    if strategy == "all":
        return all_years

    if strategy == "recent":
        return all_years[-max(1, int(n_points)) :]

    if strategy == "fixed_spacing":
        step = max(1, int(spacing))
        yrs = list(range(start_year, max_eval_year + 1, step))
        if yrs[-1] != max_eval_year:
            yrs.append(max_eval_year)
        return yrs

    # default: even_spacing
    k = max(1, int(n_points))
    if k == 1:
        return [all_years[-1]]
    idx = np.linspace(0, len(all_years) - 1, num=min(k, len(all_years)), dtype=int)
    return sorted({all_years[i] for i in idx})


def claim_count_exposure_by_ay(
    claims: pd.DataFrame,
    ay_index: Sequence,
    claim_type: Optional[int] = None,
    valuation_pay_year: Optional[int] = None,
) -> pd.Series:
    data = claims.copy()
    if claim_type is not None:
        data = data.loc[data["Type"] == claim_type].copy()

    # Optional valuation-time filter to avoid using future-reported claims in rolling-origin.
    if valuation_pay_year is not None and "RepAY" in data.columns:
        data["RepAY"] = pd.to_numeric(data["RepAY"], errors="coerce").astype("Int64")
        data = data.loc[data["RepAY"].notna() & (data["RepAY"] <= int(valuation_pay_year))].copy()

    data["AY"] = pd.to_numeric(data["AY"], errors="coerce").astype("Int64")
    exposure = data.groupby("AY").size().astype(float).rename("exposure_proxy_claim_count")
    exposure = exposure.reindex(ay_index).fillna(0.0)

    if (exposure <= 0).any():
        positive_min = exposure[exposure > 0].min() if (exposure > 0).any() else 1.0
        exposure = exposure.mask(exposure <= 0, positive_min)

    return exposure


def run_static_benchmark(
    paid: pd.DataFrame,
    claims: pd.DataFrame,
    full_claims: Optional[pd.DataFrame] = None,
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
    mature_reported_threshold: float = 0.70,
    include_cape_cod: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tri_obs = build_full_incremental_triangle_from_simulator(
        paid=paid,
        claims=claims,
        claim_type=claim_type,
        scale_divisor=scale_divisor,
    )
    tri_obs = tri_obs.clip(lower=0.0)

    exposure = claim_count_exposure_by_ay(claims=claims, ay_index=tri_obs.index, claim_type=claim_type)

    point_tbl = run_benchmark_point_estimates(
        incremental_triangle=tri_obs,
        exposure_by_ay=exposure if include_cape_cod else None,
        mature_reported_threshold=mature_reported_threshold,
    )

    truth_claims = full_claims if full_claims is not None else claims
    true_ult = true_ultimate_by_ay_from_claims(claims=truth_claims, claim_type=claim_type, scale_divisor=scale_divisor)
    compare_tbl = compare_to_true_ultimate(point_tbl, true_ult)
    err_tbl = error_summary_vs_true(compare_tbl)
    return compare_tbl, err_tbl


def run_rolling_origin_benchmark(
    full_paid: pd.DataFrame,
    full_claims: pd.DataFrame,
    claim_type: Optional[int],
    valuation_pay_years: Iterable[int],
    scale_divisor: float = 1000.0,
    mature_reported_threshold: float = 0.70,
    include_cape_cod: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_tri = build_full_incremental_triangle_from_simulator(
        paid=full_paid,
        claims=full_claims,
        claim_type=claim_type,
        scale_divisor=scale_divisor,
    )
    true_ult = true_ultimate_by_ay_from_claims(claims=full_claims, claim_type=claim_type, scale_divisor=scale_divisor)

    compare_all: List[pd.DataFrame] = []
    summary_all: List[pd.DataFrame] = []

    for v in valuation_pay_years:
        tri_obs = apply_valuation_cutoff(full_tri, valuation_pay_year=int(v)).clip(lower=0.0)
        if tri_obs.empty:
            continue

        exposure = claim_count_exposure_by_ay(
            claims=full_claims,
            ay_index=tri_obs.index,
            claim_type=claim_type,
            valuation_pay_year=int(v),
        )

        point_tbl = run_benchmark_point_estimates(
            incremental_triangle=tri_obs,
            exposure_by_ay=exposure if include_cape_cod else None,
            mature_reported_threshold=mature_reported_threshold,
        )

        compare_tbl = compare_to_true_ultimate(point_tbl, true_ult)
        compare_tbl["valuation_pay_year"] = int(v)
        compare_tbl["lob"] = "All" if claim_type is None else f"Type_{claim_type}"
        compare_all.append(compare_tbl)

        err_tbl = error_summary_vs_true(compare_tbl)
        err_tbl["valuation_pay_year"] = int(v)
        err_tbl["lob"] = "All" if claim_type is None else f"Type_{claim_type}"
        summary_all.append(err_tbl)

    compare_long = pd.concat(compare_all, ignore_index=True) if compare_all else pd.DataFrame()
    summary_long = pd.concat(summary_all, ignore_index=True) if summary_all else pd.DataFrame()

    return compare_long, summary_long


def claim_count_exposure_from_output(
    output_df: pd.DataFrame,
    ay_index: Sequence,
    claim_type: Optional[int] = None,
    valuation_pay_year: Optional[int] = None,
) -> pd.Series:
    """Cape-Cod exposure proxy from raw output without rolling-origin leakage.

    When a valuation year is supplied, the claim-count proxy is restricted to claims
    that would have been reported by that valuation. The simulator stores reporting
    delay but not an explicit RepAY column in `output`, so we reconstruct it as
    floor(AY + RepDelayYears).
    """

    data = output_df.copy()
    data["AY"] = pd.to_numeric(data["AY"], errors="coerce").astype("Int64")
    data["Type"] = pd.to_numeric(data["Type"], errors="coerce").astype("Int64")

    if claim_type is not None:
        data = data.loc[data["Type"] == int(claim_type)].copy()

    if valuation_pay_year is not None:
        if "RepAY" in data.columns:
            data["RepAY"] = pd.to_numeric(data["RepAY"], errors="coerce").astype("Int64")
        elif "RepDelayYears" in data.columns:
            rep_delay = pd.to_numeric(data["RepDelayYears"], errors="coerce").fillna(0.0)
            data["RepAY"] = np.floor(pd.to_numeric(data["AY"], errors="coerce") + rep_delay).astype("Int64")
        else:
            data["RepAY"] = data["AY"]
        data = data.loc[data["RepAY"].notna() & (data["RepAY"] <= int(valuation_pay_year))].copy()

    exposure = data.groupby("AY").size().astype(float).rename("exposure_proxy_claim_count")
    exposure = exposure.reindex(ay_index).fillna(0.0)
    if (exposure <= 0).any():
        positive_min = exposure[exposure > 0].min() if (exposure > 0).any() else 1.0
        exposure = exposure.mask(exposure <= 0, positive_min)
    return exposure


def run_static_benchmark_from_output(
    output_df: pd.DataFrame,
    valuation_year: int,
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
    mature_reported_threshold: float = 0.70,
    include_cape_cod: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tri_obs = build_incremental_triangle_from_output(
        output_df=output_df,
        claim_type=claim_type,
        valuation_year=int(valuation_year),
        scale_divisor=scale_divisor,
    ).clip(lower=0.0)

    exposure = claim_count_exposure_from_output(output_df, ay_index=tri_obs.index, claim_type=claim_type)

    point_tbl = run_benchmark_point_estimates(
        incremental_triangle=tri_obs,
        exposure_by_ay=exposure if include_cape_cod else None,
        mature_reported_threshold=mature_reported_threshold,
    )

    true_ult = true_ultimate_by_ay_from_output(output_df, claim_type=claim_type, scale_divisor=scale_divisor)
    compare_tbl = compare_to_true_ultimate(point_tbl, true_ult)
    err_tbl = error_summary_vs_true(compare_tbl)
    return compare_tbl, err_tbl


def run_rolling_origin_benchmark_from_output(
    output_df: pd.DataFrame,
    claim_type: Optional[int],
    valuation_pay_years: Iterable[int],
    scale_divisor: float = 1000.0,
    mature_reported_threshold: float = 0.70,
    include_cape_cod: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    true_ult = true_ultimate_by_ay_from_output(output_df, claim_type=claim_type, scale_divisor=scale_divisor)

    compare_all: List[pd.DataFrame] = []
    summary_all: List[pd.DataFrame] = []

    for v in valuation_pay_years:
        tri_obs = build_incremental_triangle_from_output(
            output_df=output_df,
            claim_type=claim_type,
            valuation_year=int(v),
            scale_divisor=scale_divisor,
        ).clip(lower=0.0)
        if tri_obs.empty:
            continue

        exposure = claim_count_exposure_from_output(
            output_df,
            ay_index=tri_obs.index,
            claim_type=claim_type,
            valuation_pay_year=int(v),
        )

        point_tbl = run_benchmark_point_estimates(
            incremental_triangle=tri_obs,
            exposure_by_ay=exposure if include_cape_cod else None,
            mature_reported_threshold=mature_reported_threshold,
        )

        compare_tbl = compare_to_true_ultimate(point_tbl, true_ult)
        compare_tbl["valuation_pay_year"] = int(v)
        compare_tbl["lob"] = "All" if claim_type is None else f"Type_{claim_type}"
        compare_all.append(compare_tbl)

        err_tbl = error_summary_vs_true(compare_tbl)
        err_tbl["valuation_pay_year"] = int(v)
        err_tbl["lob"] = "All" if claim_type is None else f"Type_{claim_type}"
        summary_all.append(err_tbl)

    compare_long = pd.concat(compare_all, ignore_index=True) if compare_all else pd.DataFrame()
    summary_long = pd.concat(summary_all, ignore_index=True) if summary_all else pd.DataFrame()
    return compare_long, summary_long
