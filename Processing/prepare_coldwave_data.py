# =============================================================================
# Build analysis-ready DataFrames from *_cw.csv event files
# - Combines all files
# - Parses method/reference/period from filename
# - Reshapes to long format: one row per (year, region, method, esm, reference, period)
# - Produces:
#   1) df_events_long  : per-event records (parea + tmin) in long format
#   2) df_annual       : annual aggregates per (year, esm, method, reference, period, region)
#   3) df_delta        : future - historical deltas per (esm, method, reference, region)
# - Saves parquet files alongside CSV (optional)
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------------------- CONFIG ----------------------
DATA_DIR = Path("../csvfiles/")   # <-- CHANGE THIS
SAVE_PARQUET = True                     # set False if you don't want parquet outputs
# ----------------------------------------------------

def parse_filename(fp: Path):
    """Parse method, reference, period from filename."""
    name = fp.name

    # period
    name_low = name.lower()
    if "historical" in name_low:
        period = "historical"
    elif "future" in name_low:
        period = "future"
    else:
        period = "unknown"

    # method / reference
    method = reference = None
    # order matters (most specific to least)
    if "DBCCA_Daymet" in name:
        method, reference = "DBCCA", "Daymet"
    elif "DBCCA_Livneh" in name:
        method, reference = "DBCCA", "Livneh"
    elif "RegCM_Daymet" in name:
        method, reference = "RegCM", "Daymet"
    elif "RegCM_Livneh" in name:
        method, reference = "RegCM", "Livneh"
    elif re.search(r"(^|_)RegCM(_|\.|$)", name) and ("Daymet" not in name and "Livneh" not in name):
        method, reference = "RegCM", "None"
    elif "LOCA_Livneh" in name:
        method, reference = "LOCA", "Livneh"
    elif "SRCNN_Daymet" in name:
        method, reference = "SRCNN", "Daymet"
    elif "SRGAN_Daymet" in name:
        method, reference = "SRGAN", "Daymet"
    elif "DaymetV4" in name:
        method, reference = "Reference", "Daymet"
    elif re.search(r"(^|_)Livneh(_|\.|$)", name):
        method, reference = "Reference", "Livneh"
    elif re.search(r"(^|_)CMIP6(_|\.|$)", name):
        method, reference = "ESM", "None"
    else:
        method, reference = "Unknown", "Unknown"

    return method, reference, period

def load_one_file(fp: Path):
    """Load a single CSV and add labels."""
    method, reference, period = parse_filename(fp)
    # read
    df = pd.read_csv(fp)

    # expected columns (subset)
    keep_cols = [
        "year","esm","sim","duration",
        "pareaCONUS","pareaCAR","pareaCENT","pareaFL","pareaMIDA","pareaMIDW",
        "pareaNE","pareaNY","pareaSE","pareaTEN","pareaSW","pareaCAL","pareaTEX","pareaNW",
        "tminCONUS","tminCAR","tminCENT","tminFL","tminMIDA","tminMIDW",
        "tminNE","tminNY","tminSE","tminTEN","tminSW","tminCAL","tminTEX","tminNW"
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].copy()

    # ensure presence of key id columns
    if "year" in df.columns:
        # some files have float year (e.g., 2020.0)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        df["year"] = pd.NA

    if "esm" not in df.columns:
        df["esm"] = "NA"

    if "sim" not in df.columns:
        df["sim"] = f"{method}_{reference}"

    df["method"] = method
    df["reference"] = reference
    df["period"] = period

    return df

def build_combined_df(data_dir: Path):
    files = sorted(data_dir.glob("*_cw.csv"))
    if not files:
        raise FileNotFoundError(f"No *_cw.csv files found in {data_dir.resolve()}")

    dfs = [load_one_file(fp) for fp in files]
    df_all = pd.concat(dfs, ignore_index=True)

    # make sure numeric cols are numeric
    numeric_like = [c for c in df_all.columns if c.startswith(("parea","tmin","duration"))]
    for c in numeric_like:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # harmonize ESM names (strip spaces)
    df_all["esm"] = df_all["esm"].astype(str).str.strip()

    # normalize method/reference strings
    for col in ["method","reference","period","sim"]:
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(str)

    return df_all

def reshape_long(df_all: pd.DataFrame):
    """Melt parea* and tmin* to long format and merge; also attach duration per event group."""
    id_vars = ["year","esm","sim","method","reference","period"]

    parea_cols = [c for c in df_all.columns if c.startswith("parea")]
    tmin_cols  = [c for c in df_all.columns if c.startswith("tmin")]

    df_parea = df_all.melt(
        id_vars=id_vars,
        value_vars=parea_cols,
        var_name="region_var",
        value_name="parea"
    )
    df_parea["region"] = df_parea["region_var"].str.replace("^parea", "", regex=True)

    df_tmin = df_all.melt(
        id_vars=id_vars,
        value_vars=tmin_cols,
        var_name="region_var",
        value_name="tmin"
    )
    df_tmin["region"] = df_tmin["region_var"].str.replace("^tmin", "", regex=True)

    # merge parea & tmin
    df_long = pd.merge(
        df_parea.drop(columns=["region_var"]),
        df_tmin.drop(columns=["region_var"]),
        on=id_vars + ["region"],
        how="outer"
    )

    # attach per-event duration summary (mean duration per group of id_vars)
    # Note: duration is defined at event level; we use mean over events per (id_vars)
    if "duration" in df_all.columns:
        df_dur = (
            df_all.groupby(id_vars, dropna=False)["duration"]
                  .mean()
                  .reset_index()
                  .rename(columns={"duration":"duration_mean_events"})
        )
        df_long = df_long.merge(df_dur, on=id_vars, how="left")

    # ensure ordering
    df_long = df_long.sort_values(id_vars + ["region"]).reset_index(drop=True)
    return df_long

def annual_aggregates(df_long: pd.DataFrame):
    """
    Build annual aggregates per (year, esm, method, reference, period, region):
      - parea_mean, parea_max, parea_sum
      - tmin_mean, tmin_max
      - duration_mean (mean of per-group duration_mean_events)
      - n_events (count of contributing rows)
    """
    group_cols = ["year","esm","method","reference","period","region"]

    # if duration_mean_events is not present, set NaN
    if "duration_mean_events" not in df_long.columns:
        df_long["duration_mean_events"] = np.nan

    df_annual = (
        df_long.groupby(group_cols, dropna=False)
               .agg(
                   parea_mean = ("parea","mean"),
                   parea_max  = ("parea","max"),
                   parea_sum  = ("parea","sum"),
                   tmin_mean  = ("tmin","mean"),
                   tmin_max   = ("tmin","max"),
                   duration_mean = ("duration_mean_events","mean"),
                   n_events   = ("parea","count")
               )
               .reset_index()
    )

    # helpful cleanups
    # cast year back to int where possible
    df_annual["year"] = pd.to_numeric(df_annual["year"], errors="coerce").astype("Int64")
    return df_annual

def compute_future_minus_historical(df_annual: pd.DataFrame):
    """Compute future - historical deltas per (esm, method, reference, region)."""
    hist = df_annual[df_annual["period"]=="historical"].copy()
    fut  = df_annual[df_annual["period"]=="future"].copy()

    # average across years within each period
    agg_cols = ["esm","method","reference","region"]
    hist_mean = (hist.groupby(agg_cols, dropna=False)
                      .mean(numeric_only=True)
                      .reset_index())
    fut_mean  = (fut.groupby(agg_cols, dropna=False)
                     .mean(numeric_only=True)
                     .reset_index())

    # merge and compute deltas
    df_delta = pd.merge(
        fut_mean, hist_mean, on=agg_cols, suffixes=("_fut","_hist"), how="inner"
    )

    for v in ["parea_mean","parea_max","parea_sum","tmin_mean","tmin_max","duration_mean","n_events"]:
        if f"{v}_fut" in df_delta.columns and f"{v}_hist" in df_delta.columns:
            df_delta[f"Δ_{v}"] = df_delta[f"{v}_fut"] - df_delta[f"{v}_hist"]

    # keep only useful columns
    keep = agg_cols + [c for c in df_delta.columns if c.startswith("Δ_")]
    df_delta = df_delta[keep].copy()
    return df_delta

def main():
    # 1) read & combine
    df_all = build_combined_df(DATA_DIR)
    print(f"[INFO] Combined rows: {len(df_all):,}")

    # 2) long format (per-event records melted to region)
    df_events_long = reshape_long(df_all)
    print(f"[INFO] Events-long rows: {len(df_events_long):,}")

    # 3) annual aggregates per (year, esm, method, reference, period, region)
    df_annual = annual_aggregates(df_events_long)
    print(f"[INFO] Annual aggregates rows: {len(df_annual):,}")

    # 4) future - historical deltas per (esm, method, reference, region)
    df_delta = compute_future_minus_historical(df_annual)
    print(f"[INFO] Delta rows: {len(df_delta):,}")

    # 5) save outputs
    out_dir = DATA_DIR / "processed-cw"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(out_dir / "00_combined_raw.csv", index=False)
    df_events_long.to_csv(out_dir / "01_events_long.csv", index=False)
    df_annual.to_csv(out_dir / "02_annual_aggregates.csv", index=False)
    df_delta.to_csv(out_dir / "03_future_minus_historical_deltas.csv", index=False)

    if SAVE_PARQUET:
        df_all.to_parquet(out_dir / "00_combined_raw.parquet", index=False)
        df_events_long.to_parquet(out_dir / "01_events_long.parquet", index=False)
        df_annual.to_parquet(out_dir / "02_annual_aggregates.parquet", index=False)
        df_delta.to_parquet(out_dir / "03_future_minus_historical_deltas.parquet", index=False)

    print(f"[DONE] Wrote outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

