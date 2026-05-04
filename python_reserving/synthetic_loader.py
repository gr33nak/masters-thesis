from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _require_pyreadr():
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyreadr is required for loading .rda files. Install with: pip install pyreadr"
        ) from exc
    return pyreadr


def load_rda_single_object(path: Path) -> pd.DataFrame:
    pyreadr = _require_pyreadr()
    result = pyreadr.read_r(str(path))
    if len(result.keys()) == 0:
        raise ValueError(f"No objects found in R data file: {path}")

    first_key = list(result.keys())[0]
    obj = result[first_key]
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Object '{first_key}' in {path.name} is not a data frame.")
    return obj


def load_synthetic_simulator_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    expected = {
        "claims": data_dir / "claims.rda",
        "paid": data_dir / "paid.rda",
        "full_claims": data_dir / "full_claims.rda",
        "full_paid": data_dir / "full_paid.rda",
        "reopen": data_dir / "reopen.rda",
    }

    out: Dict[str, pd.DataFrame] = {}
    for name, path in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"Required synthetic file missing: {path}")
        out[name] = load_rda_single_object(path)
    return out


def _dev_from_col(col_name: str, prefix: str) -> Optional[int]:
    m = re.fullmatch(rf"{prefix}(\d{{2}})", str(col_name))
    if m is None:
        return None
    return int(m.group(1))


def pay_columns_from_output(output_df: pd.DataFrame) -> list[str]:
    cols = sorted(
        [c for c in output_df.columns if _dev_from_col(c, "Pay") is not None],
        key=lambda c: int(_dev_from_col(c, "Pay")),
    )
    if not cols:
        raise KeyError("No Pay00..PayNN columns found in Simulation.Machine output.")
    return cols


def build_incremental_triangle_from_output(
    output_df: pd.DataFrame,
    claim_type: Optional[int] = None,
    valuation_year: Optional[int] = None,
    scale_divisor: float = 1000.0,
) -> pd.DataFrame:
    """Build AYxDY incremental paid triangle directly from raw yearly output table."""

    data = output_df.copy()
    required = {"AY", "Type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns in output_df: {missing}")

    data["AY"] = pd.to_numeric(data["AY"], errors="coerce").astype("Int64")
    data["Type"] = pd.to_numeric(data["Type"], errors="coerce").astype("Int64")

    if claim_type is not None:
        data = data.loc[data["Type"] == int(claim_type)].copy()

    pay_cols = pay_columns_from_output(data)
    for c in pay_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

    ay_paid = data.groupby("AY", as_index=True)[pay_cols].sum().sort_index()
    tri = ay_paid.rename(columns=lambda c: int(_dev_from_col(c, "Pay")))
    tri.columns.name = "DY"
    tri.index.name = "AY"
    tri = tri / float(scale_divisor)

    if valuation_year is not None:
        v = int(valuation_year)
        for ay in tri.index:
            for dy in tri.columns:
                if int(ay) + int(dy) > v:
                    tri.loc[ay, dy] = np.nan

    tri = tri.loc[~tri.isna().all(axis=1), :]
    tri = tri.loc[:, ~tri.isna().all(axis=0)]
    return tri


def true_ultimate_by_ay_from_output(
    output_df: pd.DataFrame,
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
) -> pd.Series:
    """True AY ultimates directly from raw yearly output by summing Pay00..PayNN."""

    data = output_df.copy()
    data["AY"] = pd.to_numeric(data["AY"], errors="coerce").astype("Int64")
    data["Type"] = pd.to_numeric(data["Type"], errors="coerce").astype("Int64")

    if claim_type is not None:
        data = data.loc[data["Type"] == int(claim_type)].copy()

    pay_cols = pay_columns_from_output(data)
    for c in pay_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

    out = (
        data.assign(_ultimate=data[pay_cols].sum(axis=1))
        .groupby("AY", as_index=True)["_ultimate"]
        .sum()
        .div(float(scale_divisor))
        .sort_index()
        .rename("true_ultimate")
    )
    return out


def valuation_view_from_full_simulation_machine(
    full_claims: pd.DataFrame,
    full_paid: pd.DataFrame,
    valuation_pay_year: int,
    include_unreported: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create observed claims/paid snapshot from full Simulation.Machine data.

    This enforces valuation-time availability:
    - observed claims are those with AY <= valuation year and (optionally) reported by valuation
    - observed paid are payment rows with PayYear <= valuation year for observed claims
    """

    claims_obs = full_claims.copy()
    claims_obs["AY"] = pd.to_numeric(claims_obs["AY"], errors="coerce").astype("Int64")
    claims_obs = claims_obs.loc[claims_obs["AY"].notna() & (claims_obs["AY"] <= int(valuation_pay_year))].copy()

    if (not include_unreported) and ("RepAY" in claims_obs.columns):
        claims_obs["RepAY"] = pd.to_numeric(claims_obs["RepAY"], errors="coerce").astype("Int64")
        claims_obs = claims_obs.loc[claims_obs["RepAY"].notna() & (claims_obs["RepAY"] <= int(valuation_pay_year))].copy()

    observed_ids = set(pd.to_numeric(claims_obs["Id"], errors="coerce").dropna().astype("Int64").tolist())

    paid_obs = full_paid.copy()
    paid_obs["PayYear"] = pd.to_numeric(paid_obs["PayYear"], errors="coerce").astype("Int64")
    paid_obs = paid_obs.loc[paid_obs["PayYear"].notna() & (paid_obs["PayYear"] <= int(valuation_pay_year))].copy()
    paid_obs["Id"] = pd.to_numeric(paid_obs["Id"], errors="coerce").astype("Int64")
    paid_obs = paid_obs.loc[paid_obs["Id"].isin(observed_ids)].copy()

    return claims_obs, paid_obs


def load_simulation_machine_data(
    rdata_path: Path,
    valuation_pay_year: Optional[int] = None,
    include_unreported: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load Simulation.Machine.V1/simulated_data.RData and adapt to existing schema.

    The source file contains one wide claim-level table (`output`) with columns:
    claim id/features + Pay00..Pay11 and Open00..Open11.
    This adapter emits the same dictionary keys used by the existing notebook:
    claims, paid, full_claims, full_paid, reopen.
    """

    pyreadr = _require_pyreadr()
    result = pyreadr.read_r(str(rdata_path))
    if "output" not in result:
        available = list(result.keys())
        raise KeyError(f"Expected object 'output' in {rdata_path.name}; found: {available}")

    raw = result["output"]
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Object 'output' in simulated_data.RData is not a data frame.")

    data = raw.copy()
    data = data.rename(columns={"ClNr": "Id", "LoB": "Type", "age": "Age", "RepDel": "RepDelayYears"})

    required = {"Id", "AY", "Type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns in Simulation.Machine output: {missing}")

    data["Id"] = pd.to_numeric(data["Id"], errors="coerce").astype("Int64")
    data["AY"] = pd.to_numeric(data["AY"], errors="coerce")
    data["Type"] = pd.to_numeric(data["Type"], errors="coerce").astype("Int64")
    data["RepDelayYears"] = pd.to_numeric(data.get("RepDelayYears", 0.0), errors="coerce").fillna(0.0)
    data["AQ"] = pd.to_numeric(data.get("AQ", np.nan), errors="coerce")
    data["cc"] = pd.to_numeric(data.get("cc", np.nan), errors="coerce")
    data["inj_part"] = pd.to_numeric(data.get("inj_part", np.nan), errors="coerce")
    data["Age"] = pd.to_numeric(data.get("Age", np.nan), errors="coerce")

    pay_cols = sorted(
        [c for c in data.columns if _dev_from_col(c, "Pay") is not None],
        key=lambda c: int(_dev_from_col(c, "Pay")),
    )
    open_cols = sorted(
        [c for c in data.columns if _dev_from_col(c, "Open") is not None],
        key=lambda c: int(_dev_from_col(c, "Open")),
    )
    if not pay_cols:
        raise KeyError("No Pay00..Pay11 columns found in Simulation.Machine output.")

    for c in pay_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)
    for c in open_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

    # Build full claim-level table in yearly format.
    full_claims = pd.DataFrame(
        {
            "Id": data["Id"],
            "Type": data["Type"],
            "AY": pd.to_numeric(data["AY"], errors="coerce").astype("Int64"),
            "RepAY": np.floor(pd.to_numeric(data["AY"], errors="coerce") + data["RepDelayYears"]).astype("Int64"),
            "RepDelayYears": data["RepDelayYears"],
            "Age": data["Age"],
            "AQ": data["AQ"],
            "cc": data["cc"],
            "inj_part": data["inj_part"],
            "Ultimate": data[pay_cols].sum(axis=1),
        }
    )

    # Build full long paid table (including zero rows to preserve observed zero increments).
    paid_frames = []
    for c in pay_cols:
        dy = int(_dev_from_col(c, "Pay"))
        open_col = f"Open{dy:02d}"
        frame = pd.DataFrame(
            {
                "Id": data["Id"],
                "AY": pd.to_numeric(data["AY"], errors="coerce").astype("Int64"),
                "DY": int(dy),
                "PayYear": (pd.to_numeric(data["AY"], errors="coerce") + int(dy)).astype("Int64"),
                "Paid": data[c],
                "OpenInd": data[open_col] if open_col in data.columns else 0.0,
                "PayInd": (data[c] != 0).astype(float),
            }
        )
        paid_frames.append(frame)

    full_paid = pd.concat(paid_frames, ignore_index=True)

    ay_max = int(pd.to_numeric(data["AY"], errors="coerce").max())
    valuation_year = int(ay_max if valuation_pay_year is None else valuation_pay_year)
    claims, paid = valuation_view_from_full_simulation_machine(
        full_claims=full_claims,
        full_paid=full_paid,
        valuation_pay_year=valuation_year,
        include_unreported=include_unreported,
    )

    reopen = pd.DataFrame(columns=["Id", "ReopenYear", "ReopenInd"])

    return {
        "output": data,
        "claims": claims,
        "paid": paid,
        "full_claims": full_claims,
        "full_paid": full_paid,
        "reopen": reopen,
    }

def build_paid_triangle_from_simulator(
    paid: pd.DataFrame,
    claims: pd.DataFrame,
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
    cumulative: bool = False,
    as_observed_triangle: bool = True,
) -> pd.DataFrame:
    """Build incremental (or cumulative) paid triangle from simulator data.

    Uses direct yearly fields from Simulation.Machine output adapter:
    - claims AY + Type
    - paid AY + DY + Paid
    """

    required_paid = {"Id", "AY", "DY", "Paid"}
    required_claims = {"Id", "AY", "Type"}
    missing_paid = sorted(required_paid - set(paid.columns))
    missing_claims = sorted(required_claims - set(claims.columns))
    if missing_paid:
        raise KeyError(f"Missing required columns in paid: {missing_paid}")
    if missing_claims:
        raise KeyError(f"Missing required columns in claims: {missing_claims}")

    payments = paid.merge(claims[["Id", "AY", "Type"]], on=["Id", "AY"], how="inner")

    if claim_type is not None:
        payments = payments.loc[payments["Type"] == claim_type].copy()

    payments["AccYear"] = pd.to_numeric(payments["AY"], errors="coerce").astype("Int64")
    payments["PayDelay"] = pd.to_numeric(payments["DY"], errors="coerce").astype("Int64")

    payments = payments.dropna(subset=["AccYear", "PayDelay", "Paid"]).copy()
    payments = payments.loc[payments["PayDelay"] >= 0].copy()

    tri = (
        payments.groupby(["AccYear", "PayDelay"], as_index=False)["Paid"]
        .sum()
        .pivot(index="AccYear", columns="PayDelay", values="Paid")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    tri = tri / scale_divisor
    tri.index.name = "AY"
    tri.columns.name = "DY"

    if as_observed_triangle:
        tri = apply_upper_triangle_mask(tri)

    # Keep only informative AY/DY slices to avoid all-NaN rows/columns.
    tri = tri.loc[~tri.isna().all(axis=1), :]
    tri = tri.loc[:, ~tri.isna().all(axis=0)]

    if cumulative:
        tri = tri.cumsum(axis=1)

    return tri


def apply_upper_triangle_mask(incremental_triangle: pd.DataFrame) -> pd.DataFrame:
    tri = incremental_triangle.copy()
    n_rows = tri.shape[0]
    n_cols = tri.shape[1]

    # Mimics R loop where lower-right cells are set to NA by development age.
    for r in range(n_rows):
        for c in range(n_cols):
            if r + c > n_rows - 1:
                tri.iat[r, c] = np.nan
    return tri


def true_ultimate_by_ay_from_claims(
    claims: pd.DataFrame,
    claim_type: Optional[int] = None,
    scale_divisor: float = 1000.0,
) -> pd.Series:
    required = {"AY", "Ultimate", "Type"}
    missing = sorted(required - set(claims.columns))
    if missing:
        raise KeyError(f"Missing required columns in claims: {missing}")

    data = claims.copy()
    if claim_type is not None:
        data = data.loc[data["Type"] == claim_type].copy()

    data["AY"] = pd.to_numeric(data["AY"], errors="coerce").astype("Int64")

    out = (
        data.groupby("AY", as_index=True)["Ultimate"]
        .sum()
        .div(scale_divisor)
        .sort_index()
        .rename("true_ultimate")
    )
    return out


def default_bf_apriori_from_observed(
    incremental_triangle: pd.DataFrame,
    loading_factor: float = 1.15,
) -> pd.Series:
    observed_cum = incremental_triangle.fillna(0.0).cumsum(axis=1).max(axis=1)
    apriori = observed_cum * loading_factor
    apriori.name = "apriori_ultimate"
    return apriori


def compare_to_true_ultimate(point_by_ay: pd.DataFrame, true_ultimate_by_ay: pd.Series) -> pd.DataFrame:
    true_df = true_ultimate_by_ay.rename("true_ultimate").reset_index().rename(columns={"index": "AY"})

    out = point_by_ay.merge(true_df, on="AY", how="left")
    out["error"] = out["ultimate"] - out["true_ultimate"]
    out["abs_error"] = out["error"].abs()
    out["pct_error"] = np.where(
        out["true_ultimate"].abs() > 1e-9,
        100.0 * out["error"] / out["true_ultimate"],
        np.nan,
    )
    return out


def error_summary_vs_true(compare_df: pd.DataFrame) -> pd.DataFrame:
    tbl = (
        compare_df.groupby("method", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("error", "mean"),
            mape=("pct_error", lambda s: float(np.nanmean(np.abs(s)))),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return tbl
