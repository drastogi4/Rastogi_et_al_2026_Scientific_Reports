"""
mvi_analysis_downscaled.py

Compute Method Variability Index (MVI) from future–historical deltas,
restricted to downscaled / bias-corrected / AI methods

Input:
    processed/03_future_minus_historical_deltas.parquet

Output:
    processed/mvi_all_downscaled.csv
    processed/mvi_all_downscaled.parquet
    processed/mvi_by_driver_downscaled.csv
    processed/mvi_by_driver_downscaled.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ================= CONFIG =================
DATA_DIR = Path("./mvi-analysis")
PROCESSED_DIR = DATA_DIR / "processed"
# ==========================================


def main():
    # ----------------- Load delta data -----------------
    df_delta = pd.read_parquet(PROCESSED_DIR / "03_future_minus_historical_deltas.parquet")
    print(f"[INFO] Loaded df_delta with shape: {df_delta.shape}")

    # Identify delta columns (e.g., Δ_parea_mean, Δ_tmax_mean, Δ_duration_mean, ...)
    delta_cols = [c for c in df_delta.columns if c.startswith("Δ_")]
    if not delta_cols:
        raise ValueError("No Δ_ columns found in delta file. Check 03_future_minus_historical_deltas.parquet.")

    print("[INFO] Unique methods in df_delta:", sorted(df_delta["method"].unique()))
    print("[INFO] Unique references in df_delta:", sorted(df_delta["reference"].unique()))

    # ----------------- Restrict to downscaled / BC / AI methods -----------------
    # Criteria:
    #   - method != 'ESM'  (exclude raw CMIP6)
    #   - reference != 'None' (exclude uncorrected RegCM)
    df_delta_sub = df_delta[(df_delta["method"] != "ESM") & (df_delta["reference"] != "None")].copy()

    print(f"[INFO] Kept {len(df_delta_sub)} rows after filtering to downscaled/BC/AI methods")
    print("[INFO] Methods in filtered set:", sorted(df_delta_sub["method"].unique()))
    print("[INFO] References in filtered set:", sorted(df_delta_sub["reference"].unique()))

    if df_delta_sub.empty:
        raise ValueError("Filtered delta DataFrame is empty. Check method/reference values and filter logic.")

    # Melt to long format
    df_delta_long = df_delta_sub.melt(
        id_vars=["esm", "method", "reference", "region"],
        value_vars=delta_cols,
        var_name="metric",
        value_name="value"
    )
    # Clean metric names: Δ_parea_mean -> parea_mean
    df_delta_long["metric_clean"] = df_delta_long["metric"].str.replace("^Δ_", "", regex=True)

    print(f"[INFO] Long-format delta data shape (downscaled only): {df_delta_long.shape}")

    # ----------------- Define MVI function -----------------
    def compute_mvi(group: pd.DataFrame) -> pd.Series:
        """Compute CV-based and RMAD-based MVI for a group."""
        vals = group["value"].dropna().values
        if len(vals) < 2:
            return pd.Series({"MVI_CV": np.nan, "MVI_RMAD": np.nan, "N": len(vals)})

        mean_val = np.nanmean(vals)
        std_val = np.nanstd(vals)
        median_val = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - median_val))

        # avoid divide by zero
        denom_mean = np.abs(mean_val) if np.abs(mean_val) > 1e-6 else np.nan
        denom_median = np.abs(median_val) if np.abs(median_val) > 1e-6 else np.nan

        MVI_CV = std_val / denom_mean if not np.isnan(denom_mean) else np.nan
        MVI_RMAD = (1.4826 * mad) / denom_median if not np.isnan(denom_median) else np.nan

        return pd.Series({"MVI_CV": MVI_CV, "MVI_RMAD": MVI_RMAD, "N": len(vals)})

    # ----------------- MVI: pooled across drivers -----------------
    mvi_all = (
        df_delta_long
        .groupby(["region", "metric_clean"], dropna=False)
        .apply(compute_mvi)
        .reset_index()
    )
    print(f"[INFO] MVI (pooled, downscaled only) shape: {mvi_all.shape}")

    # ----------------- MVI: conditional on ESM+reference -----------------
    mvi_by_driver = (
        df_delta_long
        .groupby(["esm", "reference", "region", "metric_clean"], dropna=False)
        .apply(compute_mvi)
        .reset_index()
    )
    print(f"[INFO] MVI (by driver, downscaled only) shape: {mvi_by_driver.shape}")

    # ----------------- Save outputs -----------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mvi_all.to_csv(PROCESSED_DIR / "mvi_all_downscaled.csv", index=False)
    mvi_by_driver.to_csv(PROCESSED_DIR / "mvi_by_driver_downscaled.csv", index=False)

    mvi_all.to_parquet(PROCESSED_DIR / "mvi_all_downscaled.parquet", index=False)
    mvi_by_driver.to_parquet(PROCESSED_DIR / "mvi_by_driver_downscaled.parquet", index=False)

    print(f"[DONE] Saved MVI (downscaled only) outputs to {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()

