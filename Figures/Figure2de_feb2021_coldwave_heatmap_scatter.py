#!/usr/bin/env python3
"""
Create the February 2021 cold-wave heatmap and scatter plot for EIA regions.

Outputs
-------
Feb_2021_CW_percentarea.pdf
scatter2_Feb2021.pdf

Expected directory structure
----------------------------
../data/DaymetV4/
../eia/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

THRESHOLD = "t05"
AREA_THRESHOLD = "200k"
YEARS = range(2015, 2023)

DAYMET_DIR = Path("../data/DaymetV4")
EIA_DIR = Path("../eia")

EVENT_ID = 7
EVENT_WINTER_YEAR = 2020

CLIMATOLOGY_START_DOY = 97
CLIMATOLOGY_END_DOY = 112

REGIONS = (
    "NE", "NW", "MIDA", "MIDW", "SE", "CENT",
    "FL", "CAR", "TEN", "SW", "CAL", "TEX",
)

EIA_FILE_REGIONS = (
    "NE", "NY", "MIDA", "MIDW", "SE", "CENT",
    "FLA", "CAR", "TEN", "SW", "CAL", "TEX", "NW",
)

REGION_ORDER = (
    "NW", "CAL", "SW", "TEX", "CENT", "MIDA",
    "MIDW", "TEN", "SE", "CAR", "FL", "NE",
)


# ---------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------

def read_ext(year: int) -> pd.DataFrame:
    """Read tracked cold-extreme objects for one winter year."""
    path = (
        DAYMET_DIR
        / (
            f"DaymetV4_VIC4_tmin_{year}_{THRESHOLD}"
            f"_numdays_track_3days_{AREA_THRESHOLD}.txt"
        )
    )

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=1,
        names=[
            "doy", "minlat", "maxlat", "minlon", "maxlon",
            "meanlon", "meanlat", "clon", "clat", "area",
        ],
    )


def read_area(year: int) -> pd.DataFrame:
    """Read percentage of each EIA region covered by the cold extreme."""
    path = (
        DAYMET_DIR
        / (
            f"DaymetV4_VIC4_tmin_{year}_{THRESHOLD}"
            f"_numdays_track_3days_{AREA_THRESHOLD}"
            "_CONUS_percentagearea.txt"
        )
    )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        names=[
            "doy", "CONUS", "CAR", "CENT", "FL",
            "MIDA", "MIDW", "NE", "NY", "SE",
            "TEN", "SW", "CAL", "TEX", "NW",
        ],
    )

    df["year"] = year
    return df


def read_intensity(year: int) -> pd.DataFrame:
    """Read mean minimum temperature for each EIA region."""
    path = (
        DAYMET_DIR
        / (
            f"DaymetV4_VIC4_tmin_{year}_{THRESHOLD}"
            f"_numdays_track_3days_{AREA_THRESHOLD}"
            "_CONUS_avgtmin.txt"
        )
    )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        names=[
            "doy", "CONUS", "CAR", "CENT", "FL",
            "MIDA", "MIDW", "NE", "NY", "SE",
            "TEN", "SW", "CAL", "TEX", "NW",
        ],
    )

    df["year"] = year
    return df


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def process_ext(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Assign an event ID to each tracked object and calculate event duration.

    Blank separator rows in the tracked-object file mark the start of a new
    event. The numbering matches the original notebook logic.
    """
    out = df.copy()

    separator = out["clat"].isna()
    out["ID"] = separator.cumsum()
    out["year"] = year

    out = (
        out[["ID", "doy", "clat", "clon", "area", "year"]]
        .dropna()
        .copy()
    )

    out["ID"] = out["ID"].astype(int)
    out["doy"] = out["doy"].astype(int)
    out["year"] = out["year"].astype(int)
    out["duration"] = out.groupby("ID")["ID"].transform("size")

    return out.reset_index(drop=True)


def read_eia_regions_cold_season() -> pd.DataFrame:
    """
    Read daily EIA demand and natural-gas data for the extended cold season.

    The cold-season year runs from November through March:
      - November and December retain their calendar year.
      - January through March are assigned to the previous year.

    The custom cold-season day coordinate follows the original notebook:
      - Nov 1 is approximately day 0.
      - Jan 1 is day 61.
    """
    regional_frames = []

    for region in EIA_FILE_REGIONS:
        path = EIA_DIR / f"Region_{region}.xlsx"

        df = pd.read_excel(path)
        df["Local date"] = pd.to_datetime(df["Local date"])

        # Keep the same aggregation used in the notebook.
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = (
            df.groupby("Local date", as_index=False)[numeric_cols]
            .mean()
        )

        required = {"D", "NG"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        df = df.rename(
            columns={
                "D": f"D_{region}",
                "NG": f"NG_{region}",
            }
        )

        regional_frames.append(
            df[["Local date", f"D_{region}", f"NG_{region}"]]
        )

    eia = regional_frames[0]

    for frame in regional_frames[1:]:
        eia = eia.merge(frame, on="Local date", how="outer")

    eia["Date"] = pd.to_datetime(eia["Local date"])
    eia["year"] = eia["Date"].dt.year
    eia["doy1"] = eia["Date"].dt.dayofyear

    jan_to_mar = eia["doy1"].between(1, 90)
    nov_to_dec = eia["doy1"] >= 305

    eia.loc[jan_to_mar, "year"] = eia.loc[jan_to_mar, "year"] - 1

    cold = eia.loc[jan_to_mar | nov_to_dec].copy()

    cold.loc[jan_to_mar, "doy"] = cold.loc[jan_to_mar, "doy1"] + 60
    cold.loc[nov_to_dec, "doy"] = cold.loc[nov_to_dec, "doy1"] - 305

    cold["year"] = cold["year"].astype(int)
    cold["doy"] = cold["doy"].astype(int)

    return cold


def combine(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Merge two dataframes by cold-season day and winter year."""
    left = left.copy()
    right = right.copy()

    left["year"] = left["year"].astype(int)
    right["year"] = right["year"].astype(int)

    return left.merge(right, how="outer", on=["doy", "year"])


def regional(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Convert one EIA region to the common long-format schema."""
    if region == "FL":
        demand_col = "D_FLA"
        gas_col = "NG_FLA"
    else:
        demand_col = f"D_{region}"
        gas_col = f"NG_{region}"

    columns = [
        "ID", "doy", "duration", "year",
        region, f"tmin{region}",
        demand_col, gas_col,
        "dayofweek", "Month",
    ]

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns for region {region}: {missing}"
        )

    out = df[columns].copy()
    out["Region"] = region

    out = out.rename(
        columns={
            region: "percentarea",
            f"tmin{region}": "intensity",
            demand_col: "Demand",
            gas_col: "NG",
        }
    )

    scaler = MinMaxScaler()
    valid = out["Demand"].notna()

    out["Demands"] = np.nan
    if valid.any():
        out.loc[valid, "Demands"] = scaler.fit_transform(
            out.loc[valid, ["Demand"]]
        ).ravel()

    return out


def load_all_data() -> pd.DataFrame:
    """Load and merge Daymet and EIA data for all requested years."""
    ext_frames = []
    area_frames = []
    intensity_frames = []

    for year in YEARS:
        print(f"Reading {year}...")

        ext_frames.append(process_ext(read_ext(year), year))
        area_frames.append(read_area(year))
        intensity_frames.append(read_intensity(year))

    dfext = pd.concat(ext_frames, ignore_index=True)
    dfarea = pd.concat(area_frames, ignore_index=True)
    dfintensity = pd.concat(intensity_frames, ignore_index=True)

    temperature_regions = (
        "CONUS", "CAR", "CENT", "FL", "MIDA", "MIDW",
        "NE", "NY", "SE", "TEN", "SW", "CAL", "TEX", "NW",
    )

    dfintensity = dfintensity.rename(
        columns={
            region: f"tmin{region}"
            for region in temperature_regions
        }
    )

    dfeia = read_eia_regions_cold_season()

    joined = combine(dfext, dfeia)
    joined = combine(joined, dfarea)
    joined = combine(joined, dfintensity)

    joined["Local date"] = pd.to_datetime(joined["Local date"])
    joined["dayofweek"] = joined["Local date"].dt.dayofweek
    joined["Month"] = joined["Local date"].dt.month

    region_frames = [regional(joined, region) for region in REGIONS]

    return (
        pd.concat(region_frames, ignore_index=True)
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def prepare_event_and_climatology(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the February 2021 event and its comparison climatology."""
    event = (
        data.loc[
            (data["ID"] == EVENT_ID)
            & (data["year"] == EVENT_WINTER_YEAR)
        ]
        .drop_duplicates()
        .copy()
    )

    if event.empty:
        raise ValueError(
            "No event data were found for "
            f"ID={EVENT_ID}, winter year={EVENT_WINTER_YEAR}."
        )

    event = event.rename(
        columns={
            "percentarea": "percentareaHW",
            "intensity": "intensityHW",
            "Demand": "DemandHW",
            "NG": "NGHW",
        }
    )

    climatology = data.loc[data["year"] != EVENT_WINTER_YEAR].copy()

    climatology = (
        climatology
        .groupby(["doy", "Region"], as_index=False)
        .mean(numeric_only=True)
    )

    climatology = climatology.loc[
        climatology["doy"].between(
            CLIMATOLOGY_START_DOY,
            CLIMATOLOGY_END_DOY,
        )
    ].copy()

    return event, climatology


def calculate_anomalies(
    event: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate demand, gas, temperature, and affected-area anomalies."""
    merged = event.merge(
        climatology,
        how="outer",
        on=["doy", "Region"],
    )

    merged["Demanddiff"] = (
        merged["DemandHW"] - merged["Demand"]
    )

    merged["Demandperdiff"] = (
        100.0
        * (merged["DemandHW"] - merged["Demand"])
        / merged["Demand"]
    )

    merged["NGperdiff"] = (
        100.0
        * (merged["NGHW"] - merged["NG"])
        / merged["NG"]
    )

    merged["intensitydiff"] = (
        merged["intensityHW"] - merged["intensity"]
    )

    merged["percentareadiff"] = (
        merged["percentareaHW"] - merged["percentarea"]
    )

    return merged


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_heatmap(
    anomaly_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the regional percentage-area heatmap."""
    plot_data = anomaly_df.copy()

    plot_data["Region"] = pd.Categorical(
        plot_data["Region"],
        categories=REGION_ORDER,
        ordered=True,
    )

    plot_data = plot_data.sort_values(["Region", "doy"])

    pivot = plot_data.pivot_table(
        index="Region",
        columns="doy",
        values="percentareaHW",
        aggfunc="mean",
        observed=False,
    )

    pivot = pivot.reindex(REGION_ORDER)

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="gist_heat_r",
        vmin=0,
        vmax=100,
        cbar_kws={
            "ticks": [0, 10, 20, 40, 60, 80, 100],
            "label": "Region affected (%)",
        },
    )

    ax.set_xlabel("Cold-season day")
    ax.set_ylabel("EIA region")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    anomaly_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the temperature-demand-area anomaly scatter plot."""
    plot_data = anomaly_df.loc[
        anomaly_df["percentareadiff"] > 0
    ].copy()

    plot_data = plot_data.dropna(
        subset=[
            "intensitydiff",
            "Demandperdiff",
            "percentareadiff",
            "Region",
        ]
    )

    if plot_data.empty:
        raise ValueError(
            "No valid rows remain for the scatter plot after filtering "
            "percentareadiff > 0."
        )

    grid = sns.relplot(
        data=plot_data,
        x="intensitydiff",
        y="Demandperdiff",
        hue="Region",
        hue_order=list(REGION_ORDER),
        size="percentareadiff",
        sizes=(40, 400),
        alpha=1,
        palette="muted",
        height=6,
    )

    grid.set_axis_labels(
        "Minimum-temperature anomaly",
        "Electricity-demand anomaly (%)",
    )

    grid.fig.tight_layout()
    grid.fig.savefig(output_path, bbox_inches="tight")
    plt.close(grid.fig)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete February 2021 cold-wave plotting workflow."""
    output_heatmap = Path("Feb_2021_CW_percentarea.pdf")
    output_scatter = Path("scatter2_Feb2021.pdf")

    data = load_all_data()
    event, climatology = prepare_event_and_climatology(data)
    anomalies = calculate_anomalies(event, climatology)

    plot_heatmap(anomalies, output_heatmap)
    plot_scatter(anomalies, output_scatter)

    print(f"Saved: {output_heatmap.resolve()}")
    print(f"Saved: {output_scatter.resolve()}")


if __name__ == "__main__":
    main()
