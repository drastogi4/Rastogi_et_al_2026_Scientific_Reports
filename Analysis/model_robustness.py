"""
# ==========================================================
# Model Robustness Assessment
# ==========================================================
# This script evaluates the predictive performance and temporal
# robustness of the regional electricity-demand models developed
# for the compound climate-load stress analysis.
#
# Validation consists of:
#   (1) Random 80/20 train-test evaluation
#   (2) Leave-one-year-out cross-validation (LOYO-CV) over
#       the 2015–2022 observational period.
#
# The script reports R², RMSE, MAE, and bias for each region
# and summarizes cross-validation statistics used in the
# manuscript and supplementary material.
# ==========================================================
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
OUTPUT_CSV = BASE_DIR / "compound_stress_events.csv"
CV_FOLD_CSV = BASE_DIR / "electricity_demand_LOYO_fold_metrics.csv"
CV_SUMMARY_CSV = BASE_DIR / "electricity_demand_LOYO_regional_summary.csv"
CV_YEARS = range(2015, 2023)

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
    df_reg.dropna(
    subset=["percentarea", "intensity", "Demand"],
    inplace=True
    )

    # SE-specific quality control
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
    coef_df = pd.DataFrame({
    "Variable": X_train.columns,
    "Coefficient": model.coef_
    })

    coef_df["AbsCoeff"] = coef_df["Coefficient"].abs()

    if region == "SE":
        print("\n==============================")
        print("Final model coefficients for SE")
        print("==============================")
        print(coef_df.sort_values("AbsCoeff", ascending=False))

    # Evaluate
    # Predict the random 20% test set in scaled units
    y_pred_scaled = model.predict(X_test)

    # Convert observations and predictions back to original load units
    y_test_original = scaler_demand.inverse_transform(
        y_test.to_numpy().reshape(-1, 1)
    ).ravel()

    y_pred_original = scaler_demand.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    # R² is unchanged by linear scaling, but calculate it in original units
    test_r2 = r2_score(y_test_original, y_pred_original)

    # RMSE is now in the original electricity-load units
    test_rmse = np.sqrt(
        mean_squared_error(y_test_original, y_pred_original)
    )

    print(
    f"[INFO] Region: {region} | "
    f"Random test R²={test_r2:.3f} | "
    f"Random test RMSE={test_rmse:.1f}"
    )

    # Predict on full
    # Predict on full dataset
    X_full = df_model.drop("Demand_scaled", axis=1)
    scaled_pred = model.predict(X_full)

    # Convert back to original demand units
    pred = scaler_demand.inverse_transform(
    scaled_pred.reshape(-1,1)).flatten()

    obs = df_reg["Demand"].values

    # Performance on full dataset
    r2_full = r2_score(obs, pred)
    rmse_full = np.sqrt(mean_squared_error(obs, pred))

    print(f"{region:5s} Final Model: R2={r2_full:.3f}  RMSE={rmse_full:.1f}")

    df_reg["Demand_predicted"] = pred

    # Save model + scalers
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / f"model_{region}.pkl")
    joblib.dump(scaler_demand, save_dir / f"scaler_demand_{region}.pkl")
    joblib.dump(scaler_cdd, save_dir / f"scaler_cdd_{region}.pkl")
    joblib.dump(scaler_area, save_dir / f"scaler_area_{region}.pkl")

    return df_reg, test_r2, test_rmse



# =========================
# STEP 2B: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
# =========================

def prepare_region_demand_data(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Prepare the historical Daymet–EIA records used by a regional demand model."""
    demand_col = f"D_{region}A" if region == "FL" else f"D_{region}"
    tmax_col = f"tmax{region}"
    area_col = region

    required_cols = [
        "ID", "doy", "duration", "year", area_col, tmax_col,
        demand_col, "dayofweek", "Month"
    ]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        print(f"[WARN] Missing columns for {region} cross-validation: {missing}")
        return pd.DataFrame()

    df_reg = df[required_cols].copy()
    df_reg["Region"] = region
    df_reg.rename(
        columns={
            area_col: "percentarea",
            tmax_col: "intensity",
            demand_col: "Demand",
        },
        inplace=True,
    )

    df_reg.dropna(
        subset=["year", "percentarea", "intensity", "Demand", "dayofweek", "Month"],
        inplace=True,
    )
    df_reg["year"] = df_reg["year"].astype(int)
    df_reg["cdd"] = (df_reg["intensity"] - 18.0).clip(lower=0)
    return df_reg


def leave_one_year_out_cv_region(
    df: pd.DataFrame,
    region: str,
    validation_years=CV_YEARS,
) -> pd.DataFrame:
    """
    Perform leave-one-year-out cross-validation for one EIA region.

    For each year from 2015 through 2022, the year is withheld, all model
    scalers and coefficients are fitted using the other available years, and
    demand is predicted for the withheld year. Metrics are computed in the
    original demand units.
    """
    df_reg = prepare_region_demand_data(df, region)

    df_reg.dropna(
    subset=[
        "year",
        "percentarea",
        "intensity",
        "Demand",
        "dayofweek",
        "Month",
    ],
    inplace=True,
    )

    if region == "SE":
        n_before = len(df_reg)

        df_reg = df_reg[
            df_reg["Demand"] >= 5000
        ].copy()

        print(
            f"[QC-CV] Removed {n_before - len(df_reg)} "
         "SE observations with Demand < 5000."
        )
    if df_reg.empty:
        return pd.DataFrame()

    valid_years = sorted(set(validation_years).intersection(df_reg["year"].unique()))
    if len(valid_years) < 2:
        print(
            f"[WARN] Region {region} has fewer than two years in the "
            f"2015–2022 validation period: {valid_years}"
        )
        return pd.DataFrame()

    cv_data = df_reg[df_reg["year"].isin(valid_years)].copy()
    fold_records = []

    for held_out_year in valid_years:
        train = cv_data[cv_data["year"] != held_out_year].copy()
        test = cv_data[cv_data["year"] == held_out_year].copy()

        if train.empty or len(test) < 2:
            print(
                f"[WARN] Skipping {region} year {held_out_year}: "
                f"train rows={len(train)}, test rows={len(test)}"
            )
            continue

        # Fit every scaler using training data only to prevent leakage.
        scaler_cdd = MinMaxScaler()
        scaler_area = MinMaxScaler()
        scaler_demand = MinMaxScaler()

        train["cdd_scaled"] = scaler_cdd.fit_transform(train[["cdd"]])
        train["percentarea_scaled"] = scaler_area.fit_transform(train[["percentarea"]])
        train["Demand_scaled"] = scaler_demand.fit_transform(train[["Demand"]])

        test["cdd_scaled"] = scaler_cdd.transform(test[["cdd"]])
        test["percentarea_scaled"] = scaler_area.transform(test[["percentarea"]])

        feature_cols = ["cdd_scaled", "percentarea_scaled", "dayofweek", "Month"]
        X_train = pd.get_dummies(
            train[feature_cols], columns=["dayofweek", "Month"], drop_first=True
        )
        X_test = pd.get_dummies(
            test[feature_cols], columns=["dayofweek", "Month"], drop_first=True
        )
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        y_train = train["Demand_scaled"]
        model = LinearRegression()
        model.fit(X_train, y_train)

        predicted_scaled = model.predict(X_test)
        y_pred = scaler_demand.inverse_transform(
            predicted_scaled.reshape(-1, 1)
        ).ravel()
        y_true = test["Demand"].to_numpy()

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        bias = np.mean(y_pred - y_true)

        fold_records.append(
            {
                "Region": region,
                "Held_out_year": int(held_out_year),
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Bias": bias,
                "N_test": len(test),
                "N_train": len(train),
            }
        )

        print(
            f"[CV] Region={region} | held-out year={held_out_year} | "
            f"R²={r2:.3f} | RMSE={rmse:.3f} | N={len(test)}"
        )

    return pd.DataFrame(fold_records)


def run_leave_one_year_out_cv(
    df: pd.DataFrame,
    regions=REGIONS,
    fold_csv: Path = CV_FOLD_CSV,
    summary_csv: Path = CV_SUMMARY_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run regional LOYO validation and save fold-level and summary tables."""
    regional_results = []

    for region in regions:
        fold_df = leave_one_year_out_cv_region(df, region)
        if not fold_df.empty:
            regional_results.append(fold_df)

    if not regional_results:
        print("[WARN] No valid leave-one-year-out cross-validation results generated.")
        return pd.DataFrame(), pd.DataFrame()

    folds = pd.concat(regional_results, ignore_index=True)
    summary = (
        folds.groupby("Region", as_index=False)
        .agg(
            Mean_R2=("R2", "mean"),
            Std_R2=("R2", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Std_RMSE=("RMSE", "std"),
            Mean_MAE=("MAE", "mean"),
            Mean_Bias=("Bias", "mean"),
            Number_of_folds=("Held_out_year", "count"),
        )
    )

    # Keep the manuscript's requested columns first.
    summary = summary[
        [
            "Region", "Mean_R2", "Std_R2", "Mean_RMSE",
            "Std_RMSE", "Mean_MAE", "Mean_Bias", "Number_of_folds",
        ]
    ]

    folds.to_csv(fold_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] LOYO fold metrics: {fold_csv}")
    print(f"[SAVED] LOYO regional summary: {summary_csv}")
    print("[INFO] LOYO regional summary:\n", summary.to_string(index=False))
    return folds, summary



# =========================
# MAIN DRIVER
# =========================

def main():

    # ==========================================================
    # STEP 1: BUILD DAYMET + EIA HISTORICAL TRAINING DATA
    # ==========================================================

    print("[STEP 1] Building Daymet + EIA training dataset...")

    dfjoin_daymet = build_daymet_eia_join()

    print(
        f"[INFO] dfjoin_daymet shape: "
        f"{dfjoin_daymet.shape}"
    )

    print(
        f"[INFO] Available years: "
        f"{sorted(dfjoin_daymet['year'].dropna().unique())}"
    )

    # ==========================================================
    # STEP 1B: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
    # ==========================================================

    print(
        "[STEP 1B] Running leave-one-year-out "
        "cross-validation for 2015–2022..."
    )

    run_leave_one_year_out_cv(
        dfjoin_daymet
    )

    # ==========================================================
    # STEP 2: TRAIN FINAL LINEAR DEMAND MODELS
    # ==========================================================

    print(
        "[STEP 2] Training final linear demand models "
        "per region..."
    )

    all_hist_pred = []
    training_results = []

    for region in REGIONS:

        print(f"\n[INFO] Training final model for region: {region}")

        result = train_linear_demand_model(
            dfjoin_daymet,
            region
        )

        # The updated training function should return:
        # df_reg, r2_full, rmse_full
        if not isinstance(result, tuple) or len(result) != 3:
            print(
                f"[WARN] Training function did not return "
                f"metrics for {region}. Skipping."
            )
            continue

        df_reg, r2, rmse = result

        if df_reg.empty:
            print(
                f"[WARN] No final-model output produced "
                f"for region {region}."
            )
            continue

        all_hist_pred.append(df_reg)

        training_results.append({
            "Region": region,
            "Training_R2": r2,
            "Training_RMSE": rmse
        })

    # Combine regional historical predictions
    if all_hist_pred:

        dffinal_hist = pd.concat(
            all_hist_pred,
            ignore_index=True
        )

        print(
            "[INFO] Combined historical prediction "
            f"DataFrame shape: {dffinal_hist.shape}"
        )

        historical_prediction_file = (
            BASE_DIR / "historical_demand_predictions.csv"
        )

        dffinal_hist.to_csv(
            historical_prediction_file,
            index=False
        )

        print(
            "[SAVED] Historical demand predictions: "
            f"{historical_prediction_file}"
        )

    else:

        print(
            "[WARN] No historical region models "
            "produced output."
        )

    # Save final-model performance table
    if training_results:

        training_performance_df = pd.DataFrame(
            training_results
        )

        # Use the same region ordering as REGIONS
        training_performance_df["Region"] = pd.Categorical(
            training_performance_df["Region"],
            categories=REGIONS,
            ordered=True
        )

        training_performance_df = (
            training_performance_df
            .sort_values("Region")
            .reset_index(drop=True)
        )

        training_performance_file = (
            BASE_DIR / "training_model_performance.csv"
        )

        training_performance_df.to_csv(
            training_performance_file,
            index=False
        )

        print("\n==============================================")
        print("FINAL MODEL PERFORMANCE")
        print("==============================================")

        print(
            training_performance_df.to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}"
            )
        )

        print(
            "\n[SAVED] Final-model performance table: "
            f"{training_performance_file}"
        )

    else:

        print(
            "[WARN] No final-model performance "
            "metrics were generated."
        )


if __name__ == "__main__":
    main()


