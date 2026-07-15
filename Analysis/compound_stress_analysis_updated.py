"""
compound_stress_analysis.py

Workflow:
1. Train linear demand models per EIA region using Daymet + EIA (historical).
2. Save models and scalers per region.
3. Load CMIP6 + downscaled event data (RegCM / DBCCA / LOCA / SRCNN / SRGAN).
4. Apply demand models to CMIP6 events to get Demand_predicted.
5. Compute:
    - CESI (climate-side Compound Stress Index)
    - CLSI (demand-side Cumulative Load Stress Index)
   per event (ID, year, region).
6. Aggregate historical vs future changes and write out CSVs for plotting.

You can split this into multiple files if desired, but here it's kept together
for clarity.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# CONFIG
# =========================

BASE_DIR = Path(".")  # adjust if needed
DAYMET_NETCDF_DIR = BASE_DIR / "../netcdf/DaymetV4"
DAYMET_TXT_DIR = BASE_DIR / "../txtfiles/DaymetV4"
EIA_DIR = BASE_DIR / "./EIA"
MODELS_DIR = BASE_DIR / "models_daymet"
CMIP6_NETCDF_DIR = BASE_DIR / "../netcdf"
CMIP6_TXT_DIR = BASE_DIR / "../txtfiles"
PLOTS_DIR = BASE_DIR / "plots_cmip6"
OUTPUT_CSV = BASE_DIR / "compound_stress_events_updated.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# heatwave threshold & mask configuration for TE files
THRESHOLD = "t95"       # or "t95" depending on what you used
THRESHOLDp = "p95"       # or "t95" depending on what you used
MINIMUM_SIZE = "12500"

# CMIP6 run IDs (special case for CNRM)
RUN_IDS = {
    "CNRM-ESM2-1": "r1i1p1f2",
    "default": "r1i1p1f1",
}

# EIA / region list
REGIONS = ["NE", "MIDA", "MIDW", "SE", "CENT", "FL", "CAR",
           "TEN", "SW", "CAL", "TEX", "NW"]  # you can add NY if needed

CMIP6_MODELS = [
    "ACCESS-CM2", "BCC-CSM2-MR", "CNRM-ESM2-1",
    "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM"
]

DOWNSCALERS = ["DBCCA", "RegCM", "SRCNN", "SRGAN"]

YEARS_HIST = range(1980, 2020)
YEARS_FUT = range(2020, 2060)
SCENARIO = "ssp585"


# =========================
# HELPERS: READING DAYMET + EIA
# =========================

def read_daymet_ext(year: int) -> pd.DataFrame:
    """Read TE event file for Daymet (geometry, ID structure)."""
    fname = DAYMET_NETCDF_DIR / f"DaymetV4_VIC4_tmax_{year}_{THRESHOLD}_numdays_track_3days_{MINIMUM_SIZE}.txt"
    df = pd.read_csv(
        fname, sep="\t", header=None, skiprows=[0],
        names=["doy", "minlat", "maxlat", "minlon", "maxlon",
               "meanlon", "meanlat", "clon", "clat", "area"]
    )
    return df


def read_daymet_area(year: int) -> pd.DataFrame:
    """Read % area per region per day for Daymet events."""
    fname = DAYMET_TXT_DIR / f"DaymetV4_VIC4_tmax_{year}_{THRESHOLD}_numdays_track_3days_{MINIMUM_SIZE}_CONUS_percentagearea.txt"
    df = pd.read_csv(
        fname, sep=" ", header=None, skiprows=[0],
        names=["doy", "CONUS", "CAR", "CENT", "FL", "MIDA", "MIDW", "NE",
               "NY", "SE", "TEN", "SW", "CAL", "TEX", "NW"]
    )
    df["doy"] += 120  # aligning with your TE convention
    df["year"] = year
    return df


def read_daymet_intensity(year: int) -> pd.DataFrame:
    """Read avg Tmax per region per day for Daymet events."""
    fname = DAYMET_TXT_DIR / f"DaymetV4_VIC4_tmax_{year}_{THRESHOLD}_numdays_track_3days_{MINIMUM_SIZE}_CONUS_avgtmax.txt"
    df = pd.read_csv(
        fname, sep=" ", header=None, skiprows=[0],
        names=["doy", "CONUS", "CAR", "CENT", "FL", "MIDA", "MIDW", "NE",
               "NY", "SE", "TEN", "SW", "CAL", "TEX", "NW"]
    )
    df["doy"] += 120
    df["year"] = year

    # rename Tmax columns to tmaxREGION
    rename_dict = {
        "CONUS": "tmaxCONUS", "CAR": "tmaxCAR", "CENT": "tmaxCENT",
        "FL": "tmaxFL", "MIDA": "tmaxMIDA", "MIDW": "tmaxMIDW",
        "NE": "tmaxNE", "NY": "tmaxNY", "SE": "tmaxSE",
        "TEN": "tmaxTEN", "SW": "tmaxSW", "CAL": "tmaxCAL",
        "TEX": "tmaxTEX", "NW": "tmaxNW"
    }
    return df.rename(columns=rename_dict)


def process_ext_events(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Process the raw TE event file to add event IDs and durations.
    ID increments at blank separator rows (where clat is NaN).
    """
    df = df.copy()
    ids = []
    years = []
    current_id = 0
    for i in range(len(df)):
        if pd.isna(df.loc[i, "clat"]):
            current_id += 1
        ids.append(current_id)
        years.append(year)
    df["ID"] = ids
    df["year"] = years

    dfproc = df[["ID", "doy", "clat", "clon", "area", "year"]].dropna()
    dfproc["ID"] = dfproc["ID"].astype(int)
    dfproc["doy"] = dfproc["doy"].astype(int) + 120
    dfproc = dfproc.reset_index(drop=True)

    duration_counts = dfproc["ID"].value_counts().to_dict()
    dfproc["duration"] = dfproc["ID"].map(duration_counts)
    return dfproc


def combine_on_doy_year(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.copy()
    df2 = df2.copy()
    df1["year"] = df1["year"].astype(int)
    df2["year"] = df2["year"].astype(int)
    return pd.merge(df1, df2, how="outer", on=["doy", "year"])


def load_eia_regions() -> pd.DataFrame:
    """
    Read EIA regional daily demand and assemble into a single DataFrame
    with columns Local date, year, doy, and D_REGION for each region.
    """
    regions = ("NE", "NY", "MIDA", "MIDW", "SE", "CENT", "FLA",
               "CAR", "TEN", "SW", "CAL", "TEX", "NW")
    all_regions = []

    for region in regions:
        fpath = EIA_DIR / f"Region_{region}.xlsx"
        if not fpath.exists():
            print(f"[WARN] Missing EIA file for {region}: {fpath}")
            continue
        df = pd.read_excel(fpath)
        df_grouped = df.groupby("Local date")[["D"]].mean().reset_index()
        df_grouped.rename(columns={"D": f"D_{region}"}, inplace=True)
        all_regions.append(df_grouped[["Local date", f"D_{region}"]])

    dffinal = all_regions[0]
    for df in all_regions[1:]:
        dffinal = pd.merge(dffinal, df, on="Local date", how="outer")

    dffinal["Date"] = pd.to_datetime(dffinal["Local date"])
    dffinal["year"] = dffinal["Date"].dt.year
    dffinal["doy"] = dffinal["Date"].dt.dayofyear - 1

    # limit to warm season if desired
    dffinal = dffinal[(dffinal["doy"] > 119) & (dffinal["doy"] < 273)]
    return dffinal


# =========================
# STEP 1: BUILD HISTORICAL TRAINING DATA (DAYMET + EIA)
# =========================

def build_daymet_eia_join() -> pd.DataFrame:
    """Load all years of Daymet TE events and merge with EIA regional demand."""
    years = range(1980, 2023)
    ext_list, area_list, intensity_list = [], [], []

    for year in years:
        try:
            df_ext = read_daymet_ext(year)
            df_proc = process_ext_events(df_ext, year)
            df_area = read_daymet_area(year)
            df_int = read_daymet_intensity(year)

            ext_list.append(df_proc)
            area_list.append(df_area)
            intensity_list.append(df_int)
        except FileNotFoundError as e:
            print(f"[WARN] Missing Daymet file for year {year}: {e}")
        except Exception as e:
            print(f"[WARN] Error processing year {year}: {e}")

    dfext = pd.concat(ext_list, ignore_index=True)
    dfarea = pd.concat(area_list, ignore_index=True)
    dfintensity = pd.concat(intensity_list, ignore_index=True)

    # Load EIA regional daily demand
    dfeia = load_eia_regions()

    # Merge TE + EIA + area + intensity
    dfjoin1 = combine_on_doy_year(dfext, dfeia)
    dfjoin2 = combine_on_doy_year(dfjoin1, dfarea)
    dfjoin = combine_on_doy_year(dfjoin2, dfintensity)

    # Time features
    dfjoin["dayofweek"] = pd.to_datetime(dfjoin["Local date"]).dt.dayofweek
    dfjoin["Month"] = pd.to_datetime(dfjoin["Local date"]).dt.month

    return dfjoin


# =========================
# STEP 2: TRAIN LINEAR MODELS PER REGION
# =========================

def train_linear_demand_model(df: pd.DataFrame, region: str,
                              save_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """
    Train a linear regression demand model for a single region:
    Demand_scaled ~ CDD_scaled + Area_scaled + calendar dummies.
    Saves:
        - model_<region>.pkl
        - scaler_demand_<region>.pkl
        - scaler_cdd_<region>.pkl
        - scaler_area_<region>.pkl
    Returns df_reg with Demand_predicted.
    """

    # FL naming special-case (FL vs FLA in EIA files)
    demand_col = f"D_{region}A" if region == "FL" else f"D_{region}"
    if demand_col not in df.columns:
        print(f"[WARN] Missing demand column {demand_col} for {region}, skipping.")
        return pd.DataFrame()

    tmax_col = f"tmax{region}"
    area_col = region
    required_cols = ["ID", "doy", "duration", "year", area_col, tmax_col,
                     demand_col, "dayofweek", "Month"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns for {region}: {missing}, skipping.")
        return pd.DataFrame()

    df_reg = df[required_cols].copy()
    df_reg["Region"] = region
    df_reg.rename(columns={
        area_col: "percentarea",
        tmax_col: "intensity",
        demand_col: "Demand"
    }, inplace=True)

    # Drop rows with missing key values
    df_reg.dropna(subset=["percentarea", "intensity", "Demand"], inplace=True)
    # Remove anomalously low SE demand values before scaling and training
    if region == "SE":
        n_before = len(df_reg)

        df_reg = df_reg[
            df_reg["Demand"] >= 5000
        ].copy()

        print(
            f"[QC] Removed {n_before - len(df_reg)} "
            "SE observations with Demand < 5000."
    )

    # CDD
    df_reg["cdd"] = (df_reg["intensity"] - 18).clip(lower=0)

    # Scalers
    scaler_cdd = MinMaxScaler()
    scaler_area = MinMaxScaler()
    scaler_demand = MinMaxScaler()

    df_reg["cdd_scaled"] = scaler_cdd.fit_transform(df_reg[["cdd"]])
    df_reg["percentarea_scaled"] = scaler_area.fit_transform(df_reg[["percentarea"]])
    df_reg["Demand_scaled"] = scaler_demand.fit_transform(df_reg[["Demand"]])

    # Build model dataset
    drop_cols = ["ID", "doy", "duration", "cdd", "Demand",
                 "percentarea", "intensity", "Region", "year"]
    df_model = df_reg.drop(columns=drop_cols)
    df_model = pd.get_dummies(df_model, columns=["dayofweek", "Month"], drop_first=True)

    X = df_model.drop("Demand_scaled", axis=1)
    y = df_model["Demand_scaled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[INFO] Region: {region} | Linear model MSE: {mse:.3f} | R²: {r2:.3f}")

    # Predict on full
    X_full = df_model.drop("Demand_scaled", axis=1)
    scaled_pred = model.predict(X_full)
    df_reg["Demand_predicted"] = scaler_demand.inverse_transform(
        scaled_pred.reshape(-1, 1)
    )

    # Save model + scalers
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / f"model_{region}.pkl")
    joblib.dump(scaler_demand, save_dir / f"scaler_demand_{region}.pkl")
    joblib.dump(scaler_cdd, save_dir / f"scaler_cdd_{region}.pkl")
    joblib.dump(scaler_area, save_dir / f"scaler_area_{region}.pkl")

    return df_reg


# =========================
# STEP 3: CMIP6 INPUT READING (SIMILAR TO DAYMET)
# =========================

def read_cmip6_ext(model: str, scenario: str, downscaler: str,
                   year: int) -> pd.DataFrame:
    run_id = RUN_IDS.get(model, RUN_IDS["default"])
    fname = (CMIP6_NETCDF_DIR /
             f"{downscaler}_Daymet/{model}_{scenario}_{run_id}_{downscaler}_Daymet"
             f"_VIC4_tmax_{year}_{THRESHOLDp}_numdays_track_3days_{MINIMUM_SIZE}.txt")
    print(fname)
    df = pd.read_csv(
        fname, sep="\t", header=None, skiprows=[0],
        names=["doy", "minlat", "maxlat", "minlon", "maxlon",
               "meanlon", "meanlat", "clon", "clat", "area"]
    )
    return df


def read_cmip6_area(model: str, scenario: str, downscaler: str,
                    year: int) -> pd.DataFrame:
    run_id = RUN_IDS.get(model, RUN_IDS["default"])
    fname = (CMIP6_TXT_DIR /
             f"{downscaler}_Daymet/{model}_{scenario}_{run_id}_{downscaler}_Daymet"
             f"_VIC4_tmax_{year}_{THRESHOLDp}_numdays_track_3days_{MINIMUM_SIZE}_CONUS_percentagearea.txt")
    df = pd.read_csv(
        fname, sep=" ", header=None, skiprows=[0],
        names=["doy", "CONUS", "CAR", "CENT", "FL", "MIDA", "MIDW", "NE",
               "NY", "SE", "TEN", "SW", "CAL", "TEX", "NW"]
    )
    df["doy"] += 120
    df["year"] = year
    return df


def read_cmip6_intensity(model: str, scenario: str, downscaler: str,
                         year: int) -> pd.DataFrame:
    run_id = RUN_IDS.get(model, RUN_IDS["default"])
    fname = (CMIP6_TXT_DIR /
             f"{downscaler}_Daymet/{model}_{scenario}_{run_id}_{downscaler}_Daymet"
             f"_VIC4_tmax_{year}_{THRESHOLDp}_numdays_track_3days_{MINIMUM_SIZE}_CONUS_avgtmax.txt")
    df = pd.read_csv(
        fname, sep=" ", header=None, skiprows=[0],
        names=["doy", "CONUS", "CAR", "CENT", "FL", "MIDA", "MIDW", "NE",
               "NY", "SE", "TEN", "SW", "CAL", "TEX", "NW"]
    )
    df["doy"] += 120
    df["year"] = year

    rename_dict = {
        "CONUS": "tmaxCONUS", "CAR": "tmaxCAR", "CENT": "tmaxCENT",
        "FL": "tmaxFL", "MIDA": "tmaxMIDA", "MIDW": "tmaxMIDW",
        "NE": "tmaxNE", "NY": "tmaxNY", "SE": "tmaxSE",
        "TEN": "tmaxTEN", "SW": "tmaxSW", "CAL": "tmaxCAL",
        "TEX": "tmaxTEX", "NW": "tmaxNW"
    }
    return df.rename(columns=rename_dict)


def load_cmip6_inputs(model: str, scenario: str, downscaler: str,
                      years: range) -> pd.DataFrame:
    """Load TE outputs (event geometry + area + tmax) for a given CMIP6 model + downscaler."""
    ext_list, area_list, intensity_list = [], [], []

    for year in years:
        try:
            df_ext = read_cmip6_ext(model, scenario, downscaler, year)
            df_proc = process_ext_events(df_ext, year)
            df_area = read_cmip6_area(model, scenario, downscaler, year)
            df_int = read_cmip6_intensity(model, scenario, downscaler, year)

            ext_list.append(df_proc)
            area_list.append(df_area)
            intensity_list.append(df_int)
        except FileNotFoundError:
            print(f"[WARN] Missing CMIP6 TE file for {model} {downscaler} {year}")
            continue

    dfext = pd.concat(ext_list, ignore_index=True)
    dfarea = pd.concat(area_list, ignore_index=True)
    dfintensity = pd.concat(intensity_list, ignore_index=True)

    dfjoin1 = combine_on_doy_year(dfext, dfarea)
    dfjoin2 = combine_on_doy_year(dfjoin1, dfintensity)

    # For CMIP6 we don't have Local date; build a synthetic date from doy if needed
    # This is mainly used for dayofweek / Month (if you want them)
    dfjoin2["Date"] = pd.to_datetime(dfjoin2["doy"], format="%j", errors="coerce")
    dfjoin2["dayofweek"] = dfjoin2["Date"].dt.dayofweek
    dfjoin2["Month"] = dfjoin2["Date"].dt.month

    return dfjoin2


# =========================
# STEP 4: APPLY MODELS TO CMIP6 + COMPUTE CESI & CLSI
# =========================

def apply_linear_model_to_cmip6(df_cmip6: pd.DataFrame, region: str,
                                models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """
    Apply the previously trained linear model for a region to CMIP6 TE events.
    Returns df_region with Demand_predicted, CDD, etc.
    """
    model_path = models_dir / f"model_{region}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model for {region} not found at {model_path}")

    model = joblib.load(model_path)
    scaler_demand = joblib.load(models_dir / f"scaler_demand_{region}.pkl")
    scaler_cdd = joblib.load(models_dir / f"scaler_cdd_{region}.pkl")
    scaler_area = joblib.load(models_dir / f"scaler_area_{region}.pkl")

    tmax_col = f"tmax{region}"
    area_col = region
    required_cols = ["ID", "doy", "duration", "year", area_col, tmax_col,
                     "dayofweek", "Month"]
    missing = [c for c in required_cols if c not in df_cmip6.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} for CMIP6 application in {region}")

    df_region = df_cmip6[required_cols].copy()
    df_region.rename(columns={
        area_col: "percentarea",
        tmax_col: "intensity"
    }, inplace=True)

    df_region.dropna(subset=["percentarea", "intensity"], inplace=True)

    # CDD
    df_region["cdd"] = (df_region["intensity"] - 18).clip(lower=0)
    df_region["cdd_scaled"] = scaler_cdd.transform(df_region[["cdd"]])
    df_region["percentarea_scaled"] = scaler_area.transform(df_region[["percentarea"]])

    # Build feature matrix to feed the model
    df_model = df_region[["cdd_scaled", "percentarea_scaled", "dayofweek", "Month"]].copy()
    df_model = pd.get_dummies(df_model, columns=["dayofweek", "Month"], drop_first=True)

    # Align columns with training
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    else:
        # For older sklearn versions, we may not have this. Then we rely on saved X columns.
        raise AttributeError("Model does not have feature_names_in_. Save X columns if needed.")

    X = df_model.reindex(columns=model_features, fill_value=0)

    scaled_pred = model.predict(X)
    df_region["Demand_predicted"] = scaler_demand.inverse_transform(
        scaled_pred.reshape(-1, 1)
    )

    df_region["Region"] = region
    return df_region


def compute_event_cesi_clsi(df_region: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Compute both CESI (climate-side compound index) and CLSI (demand-side cumulative
    anomaly) per event (ID, year) for a given region, using Demand_predicted as the
    model-based demand time series.

    Assumes df_region has:
        - ID, year, duration, doy, dayofweek
        - percentarea, cdd
        - Demand_predicted
    """
    df = df_region.copy()

    required_cols = ["ID", "year", "duration", "doy", "dayofweek",
                     "percentarea", "cdd", "Demand_predicted"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in df_region for {region}")

    # Build a historical model-based climatology (1980–2019)
    df_hist = df[(df["year"] >= 1980) & (df["year"] <= 2019)].copy()
    if df_hist.empty:
        raise ValueError(f"No historical (1980–2019) data in df_region for {region}")

    clim = (
        df_hist.groupby(["doy", "dayofweek"], as_index=False)["Demand_predicted"]
        .mean()
        .rename(columns={"Demand_predicted": "Demand_clim"})
    )

    # Merge climatology onto full df
    df = df.merge(clim, on=["doy", "dayofweek"], how="left")

    # Demand anomaly
    df["Demand_anom"] = df["Demand_predicted"] - df["Demand_clim"]

    # Aggregate to event level
    events = (
        df.groupby(["ID", "year"], as_index=False)
          .agg(
              duration=("duration", "first"),
              cdd_mean=("cdd", "mean"),
              area_mean=("percentarea", "mean"),
              CLSI=("Demand_anom", "mean")
          )
    )

    # CESI = mean CDD * duration * mean area
    events["CESI"] = events["cdd_mean"] * events["area_mean"]
    events["Region"] = region
    return events

# =========================
# MAIN DRIVER
# =========================

def main():
    # ---- 1) Build Daymet+EIA training dataset ----
    print("[STEP 1] Building Daymet + EIA training dataset...")
    dfjoin_daymet = build_daymet_eia_join()
    print(f"[INFO] dfjoin_daymet shape: {dfjoin_daymet.shape}")

    # ---- 2) Train linear models per region ----
    print("[STEP 2] Training linear demand models per region...")
    all_hist_pred = []
    for region in REGIONS:
        df_reg = train_linear_demand_model(dfjoin_daymet, region)
        if not df_reg.empty:
            all_hist_pred.append(df_reg)
    if all_hist_pred:
        dffinal_hist = pd.concat(all_hist_pred, ignore_index=True)
        print(f"[INFO] Combined historical prediction DataFrame shape: {dffinal_hist.shape}")
    else:
        print("[WARN] No historical region models produced output.")

    # ---- 3) Apply models to CMIP6 + compute CESI/CLSI ----
    print("[STEP 3] Applying models to CMIP6 and computing CESI/CLSI...")

    records = []
    for downscaler in DOWNSCALERS:
        for model_name in CMIP6_MODELS:
            print(f"[INFO] Processing CMIP6 model={model_name}, downscaler={downscaler}...")
            try:
                df_cmip6 = load_cmip6_inputs(
                    model=model_name,
                    scenario=SCENARIO,
                    downscaler=downscaler,
                    years=range(1980, 2060)
                )
            except Exception as e:
                print(f"[WARN] Failed to load CMIP6 inputs for {model_name}/{downscaler}: {e}")
                continue

            for region in REGIONS:
                if region not in df_cmip6.columns or f"tmax{region}" not in df_cmip6.columns:
                    print(f"[WARN] CMIP6 inputs missing region {region} for {model_name}/{downscaler}")
                    continue

                try:
                    df_region = apply_linear_model_to_cmip6(df_cmip6, region)
                    df_events = compute_event_cesi_clsi(df_region, region=region)
                except Exception as e:
                    print(f"[WARN] Failed for region={region}, {model_name}/{downscaler}: {e}")
                    continue

                # Compute mean historical/future CLSI & CESI
                hist_mask = (df_events["year"] >= 1980) & (df_events["year"] <= 2019)
                fut_mask = (df_events["year"] >= 2020) & (df_events["year"] <= 2059)

                hist_clsi = df_events.loc[hist_mask, "CLSI"].mean()
                fut_clsi = df_events.loc[fut_mask, "CLSI"].mean()
                hist_cesi = df_events.loc[hist_mask, "CESI"].mean()
                fut_cesi = df_events.loc[fut_mask, "CESI"].mean()

                delta_clsi = fut_clsi - hist_clsi
                delta_cesi = fut_cesi - hist_cesi
                rel_delta_cesi  = (fut_cesi  - hist_cesi)  / hist_cesi
                rel_delta_clsi = (fut_clsi - hist_clsi) / hist_clsi

                records.append({
                    "ESM": model_name,
                    "Downscaler": downscaler,
                    "Region": region,
                    "CLSI_hist": hist_clsi,
                    "CLSI_fut": fut_clsi,
                    "CESI_hist": hist_cesi,
                    "CESI_fut": fut_cesi,
                    "Delta_CLSI": delta_clsi,
                    "Delta_CESI": delta_cesi,
                    "REL_Delta_CESI":rel_delta_cesi,
                    "REL_Delta_CLSI":rel_delta_clsi
                })

    if records:
        df_events_summary = pd.DataFrame(records)
        df_events_summary.to_csv(OUTPUT_CSV, index=False)
        print(f"[DONE] Saved compound stress summary to {OUTPUT_CSV}")
    else:
        print("[WARN] No event-level records generated.")


if __name__ == "__main__":
    main()
