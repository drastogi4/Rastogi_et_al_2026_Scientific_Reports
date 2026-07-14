"""
variance_decomposition.py

Perform variance decomposition (ANOVA) to partition variance in
future–historical deltas across ESMs, Method, and Reference.

Input:
    processed/03_future_minus_historical_deltas.parquet

Output:
    processed/variance_decomposition.csv
    processed/variance_decomposition.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ================= CONFIG =================
DATA_DIR = Path("./mvi-analysis/")   # <-- change this
PROCESSED_DIR = DATA_DIR / "processed-cw"
# ==========================================

def variance_decomp_region_metric(df_sub: pd.DataFrame) -> pd.Series:
    """
    Perform ANOVA for a single (region, metric_clean) subset:
        value ~ C(esm) + C(method) + C(reference)

    Returns fractions of SS for each factor and residual.
    """
    df_sub = df_sub.dropna(subset=["value"]).copy()
    n = len(df_sub)
    if df_sub["value"].nunique() < 2 or n < 4:
        # Not enough variation or samples to fit
        return pd.Series({
            "SS_esm_frac": np.nan,
            "SS_method_frac": np.nan,
            "SS_reference_frac": np.nan,
            "SS_resid_frac": np.nan,
            "N": n
        })

    # Cast to category for safety
    for col in ["esm", "method", "reference"]:
        df_sub[col] = df_sub[col].astype("category")

    try:
        # Fit linear model
        model = ols("value ~ C(esm) + C(method) + C(reference)", data=df_sub).fit()

        # ANOVA table (Type II)
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Total SS (including residual)
        SS_total = anova_table["sum_sq"].sum()

        def get_frac(term):
            return (
                anova_table.loc[term, "sum_sq"] / SS_total
                if term in anova_table.index and SS_total > 0
                else np.nan
            )

        SS_esm_frac       = get_frac("C(esm)")
        SS_method_frac    = get_frac("C(method)")
        SS_reference_frac = get_frac("C(reference)")
        SS_resid_frac     = get_frac("Residual")

        return pd.Series({
            "SS_esm_frac": SS_esm_frac,
            "SS_method_frac": SS_method_frac,
            "SS_reference_frac": SS_reference_frac,
            "SS_resid_frac": SS_resid_frac,
            "N": n
        })
    except Exception as e:
        # In case of singular designs / weird small-n issues
        return pd.Series({
            "SS_esm_frac": np.nan,
            "SS_method_frac": np.nan,
            "SS_reference_frac": np.nan,
            "SS_resid_frac": np.nan,
            "N": n
        })

def main():
    # ----------------- Load delta data -----------------
    df_delta = pd.read_parquet(PROCESSED_DIR / "03_future_minus_historical_deltas.parquet")
    print(f"[INFO] Loaded df_delta with shape: {df_delta.shape}")

    # Identify delta columns
    delta_cols = [c for c in df_delta.columns if c.startswith("Δ_")]
    if not delta_cols:
        raise ValueError("No Δ_ columns found in delta file. Check 03_future_minus_historical_deltas.parquet.")

    # Melt to long format
    df_delta_long = df_delta.melt(
        id_vars=["esm", "method", "reference", "region"],
        value_vars=delta_cols,
        var_name="metric",
        value_name="value"
    )
    df_delta_long["metric_clean"] = df_delta_long["metric"].str.replace("^Δ_", "", regex=True)

    print(f"[INFO] Long-format delta data shape: {df_delta_long.shape}")

    # ----------------- Apply variance decomposition per region/metric -----------------
    var_decomp = (
        df_delta_long
        .groupby(["region", "metric_clean"], dropna=False)
        .apply(variance_decomp_region_metric)
        .reset_index()
    )

    print(f"[INFO] Variance decomposition result shape: {var_decomp.shape}")

    # ----------------- Save outputs -----------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    var_decomp.to_csv(PROCESSED_DIR / "variance_decomposition.csv", index=False)
    var_decomp.to_parquet(PROCESSED_DIR / "variance_decomposition.parquet", index=False)

    print(f"[DONE] Saved variance decomposition outputs to {PROCESSED_DIR.resolve()}")

if __name__ == "__main__":
    main()

