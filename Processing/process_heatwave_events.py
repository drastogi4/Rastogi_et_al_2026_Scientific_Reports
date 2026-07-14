#!/usr/bin/env python3

"""
process_heatwave_events.py

Combine TempestExtremes event statistics, regional affected area,
and regional mean Tmax into analysis-ready CSV files.

The script processes:

1. ESM-driven and downscaled datasets
   - Historical: 1980-2019
   - Future:     2020-2059

2. Observational datasets
   - Daymet v4: 1980-2019
   - Livneh v2: 1980-2018

Expected TempestExtremes statistics files
-----------------------------------------
For ESM-driven datasets:

../netcdf/<simulation>/
<ESM>_ssp585_<ensemble>_<simulation>_VIC4_tmax_<year>_p95_
daily_mask_track_3days_12500.txt

For observations:

../netcdf/<observation>/
<observation>_tmax_<year>_p95_
daily_mask_track_3days_12500.txt

Expected regional files
-----------------------
Affected area:

../txtfiles/<dataset>/<prefix>_CONUS_percentagearea.txt

Mean Tmax:

../txtfiles/<dataset>/<prefix>_CONUS_avgtmax.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

NETCDF_ROOT = Path("../netcdf")
TEXT_ROOT = Path("../txtfiles")
OUTPUT_ROOT = Path("./csvfiles")

SCENARIO = "ssp585"
GRID = "VIC4"
VARIABLE = "tmax"
THRESHOLD = "p95"

MINIMUM_DURATION = 3
MINIMUM_SIZE = 12500

# The daily-mask files contain May-September only.
# Index 0 therefore corresponds to original day-of-year index 120.
DOY_OFFSET = 120


SIMULATIONS = (
    "CMIP6",
    "RegCM",
    "RegCM_Daymet",
    "RegCM_Livneh",
    "DBCCA_Daymet",
    "DBCCA_Livneh",
    "SRCNN_Daymet",
    "SRGAN_Daymet",
    "LOCA_Livneh",
)


ESMS = (
    "ACCESS-CM2",
    "BCC-CSM2-MR",
    "CNRM-ESM2-1",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "NorESM2-MM",
)


PERIODS = {
    "historical": range(1980, 2020),
    "future": range(2020, 2060),
}


# Observations do not have ESM, ensemble, or scenario identifiers.
OBSERVATION_PERIODS = {
    "DaymetV4": range(1980, 2020),
    "Livneh": range(1980, 2019),
}


REGIONS = (
    "CONUS",
    "CAR",
    "CENT",
    "FL",
    "MIDA",
    "MIDW",
    "NE",
    "NY",
    "SE",
    "TEN",
    "SW",
    "CAL",
    "TEX",
    "NW",
)


EVENT_COLUMNS = (
    "doy",
    "minlat",
    "maxlat",
    "minlon",
    "maxlon",
    "meanlon",
    "meanlat",
    "clon",
    "clat",
    "area",
)


REGIONAL_COLUMNS = (
    "ID",
    "doy",
    *REGIONS,
)


# ============================================================
# Filename helpers
# ============================================================

def get_ensemble(esm: str) -> str:
    """Return the ensemble member used for an ESM."""

    if esm == "CNRM-ESM2-1":
        return "r1i1p1f2"

    return "r1i1p1f1"


def build_esm_prefix(
    year: int,
    esm: str,
    simulation: str,
) -> str:
    """
    Construct the shared filename prefix for an ESM-driven dataset.
    """

    ensemble = get_ensemble(esm)

    return (
        f"{esm}_{SCENARIO}_{ensemble}_{simulation}_"
        f"{GRID}_{VARIABLE}_{year}_{THRESHOLD}_"
        f"daily_mask_track_{MINIMUM_DURATION}days_{MINIMUM_SIZE}"
    )


def build_observation_prefix(
    year: int,
    observation: str,
) -> str:
    """
    Construct the shared filename prefix for Daymet or Livneh.
    """

    return (
        f"{observation}_{VARIABLE}_{year}_{THRESHOLD}_"
        f"daily_mask_track_{MINIMUM_DURATION}days_{MINIMUM_SIZE}"
    )


# ============================================================
# File readers
# ============================================================

def read_event_statistics(
    filename: Path,
) -> pd.DataFrame:
    """
    Read the BlobStats output from TempestExtremes.
    """

    if not filename.is_file():
        raise FileNotFoundError(
            f"Missing TempestExtremes statistics file:\n{filename}"
        )

    return pd.read_csv(
        filename,
        sep=r"\s+|\t+",
        engine="python",
        header=None,
        skiprows=1,
        names=EVENT_COLUMNS,
    )


def read_regional_table(
    filename: Path,
    year: int,
    esm: str,
    simulation: str,
) -> pd.DataFrame:
    """
    Read a regional affected-area or regional-temperature table.
    """

    if not filename.is_file():
        raise FileNotFoundError(
            f"Missing regional statistics file:\n{filename}"
        )

    dataframe = pd.read_csv(
        filename,
        sep=r"\s+",
        engine="python",
        header=None,
        skiprows=1,
        names=REGIONAL_COLUMNS,
    )

    dataframe["ID"] = pd.to_numeric(
        dataframe["ID"],
        errors="coerce",
    )

    dataframe["doy"] = pd.to_numeric(
        dataframe["doy"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["ID", "doy"]
    ).copy()

    dataframe["ID"] = dataframe["ID"].astype(int)

    dataframe["doy"] = (
        dataframe["doy"].astype(int)
        + DOY_OFFSET
    )

    dataframe["year"] = year
    dataframe["esm"] = esm
    dataframe["sim"] = simulation

    return dataframe


# ============================================================
# Event processing
# ============================================================

def process_event_statistics(
    dataframe: pd.DataFrame,
    year: int,
    esm: str,
    simulation: str,
) -> pd.DataFrame:
    """
    Add event IDs and duration to the BlobStats table.

    TempestExtremes separates events using rows where the centroid
    latitude is missing. The cumulative number of these separator
    rows is used as the event ID, reproducing the original workflow.
    """

    dataframe = dataframe.copy()

    # Each missing clat row indicates the start of a new event block.
    dataframe["ID"] = (
        dataframe["clat"]
        .isna()
        .cumsum()
    )

    dataframe["year"] = year
    dataframe["esm"] = esm
    dataframe["sim"] = simulation

    dataframe = dataframe[
        [
            "ID",
            "doy",
            "clat",
            "clon",
            "area",
            "year",
            "esm",
            "sim",
        ]
    ].copy()

    dataframe = dataframe.dropna(
        subset=[
            "doy",
            "clat",
            "clon",
            "area",
        ]
    ).copy()

    dataframe["ID"] = dataframe["ID"].astype(int)

    dataframe["doy"] = (
        pd.to_numeric(
            dataframe["doy"],
            errors="raise",
        ).astype(int)
        + DOY_OFFSET
    )

    # Duration is the number of daily records belonging to the event.
    dataframe["duration"] = (
        dataframe.groupby(
            [
                "year",
                "esm",
                "sim",
                "ID",
            ]
        )["ID"]
        .transform("size")
        .astype(int)
    )

    return dataframe.reset_index(drop=True)


def rename_area_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add the parea prefix to regional area columns."""

    rename_map = {
        region: f"parea{region}"
        for region in REGIONS
    }

    return dataframe.rename(
        columns=rename_map
    )


def rename_temperature_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add the tmax prefix to regional temperature columns."""

    rename_map = {
        region: f"tmax{region}"
        for region in REGIONS
    }

    return dataframe.rename(
        columns=rename_map
    )


def combine_tables(
    event_data: pd.DataFrame,
    area_data: pd.DataFrame,
    intensity_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge event, area, and intensity information.
    """

    merge_columns = [
        "doy",
        "year",
        "ID",
        "esm",
        "sim",
    ]

    combined = pd.merge(
        event_data,
        area_data,
        how="outer",
        on=merge_columns,
        validate="many_to_one",
    )

    combined = pd.merge(
        combined,
        intensity_data,
        how="outer",
        on=merge_columns,
        validate="many_to_one",
    )

    return combined.sort_values(
        [
            "esm",
            "sim",
            "year",
            "ID",
            "doy",
        ]
    ).reset_index(drop=True)


# ============================================================
# Process one ESM-driven dataset
# ============================================================

def process_esm_dataset(
    simulation: str,
    period_name: str,
    years: range,
) -> pd.DataFrame:
    """
    Process one simulation for all ESMs and years.
    """

    event_frames = []
    area_frames = []
    intensity_frames = []

    for esm in ESMS:

        print(
            f"\nProcessing {simulation}, "
            f"{esm}, {period_name}"
        )

        for year in years:

            print(f"  Year {year}")

            prefix = build_esm_prefix(
                year=year,
                esm=esm,
                simulation=simulation,
            )

            event_file = (
                NETCDF_ROOT
                / simulation
                / f"{prefix}.txt"
            )

            area_file = (
                TEXT_ROOT
                / simulation
                / f"{prefix}_CONUS_percentagearea.txt"
            )

            intensity_file = (
                TEXT_ROOT
                / simulation
                / f"{prefix}_CONUS_avgtmax.txt"
            )

            event_raw = read_event_statistics(
                event_file
            )

            event_processed = process_event_statistics(
                dataframe=event_raw,
                year=year,
                esm=esm,
                simulation=simulation,
            )

            area = read_regional_table(
                filename=area_file,
                year=year,
                esm=esm,
                simulation=simulation,
            )

            intensity = read_regional_table(
                filename=intensity_file,
                year=year,
                esm=esm,
                simulation=simulation,
            )

            event_frames.append(event_processed)
            area_frames.append(area)
            intensity_frames.append(intensity)

    events = pd.concat(
        event_frames,
        ignore_index=True,
    )

    areas = pd.concat(
        area_frames,
        ignore_index=True,
    )

    intensities = pd.concat(
        intensity_frames,
        ignore_index=True,
    )

    areas = rename_area_columns(areas)
    intensities = rename_temperature_columns(intensities)

    return combine_tables(
        event_data=events,
        area_data=areas,
        intensity_data=intensities,
    )


# ============================================================
# Process Daymet or Livneh
# ============================================================

def process_observation(
    observation: str,
    years: range,
) -> pd.DataFrame:
    """
    Process one observational dataset.

    The observation name is used in the esm column because there is
    no driving ESM. This keeps the output structure consistent with
    the ESM-driven datasets.
    """

    event_frames = []
    area_frames = []
    intensity_frames = []

    print(f"\nProcessing observation: {observation}")

    for year in years:

        print(f"  Year {year}")

        prefix = build_observation_prefix(
            year=year,
            observation=observation,
        )

        event_file = (
            NETCDF_ROOT
            / observation
            / f"{prefix}.txt"
        )

        area_file = (
            TEXT_ROOT
            / observation
            / f"{prefix}_CONUS_percentagearea.txt"
        )

        intensity_file = (
            TEXT_ROOT
            / observation
            / f"{prefix}_CONUS_avgtmax.txt"
        )

        event_raw = read_event_statistics(
            event_file
        )

        event_processed = process_event_statistics(
            dataframe=event_raw,
            year=year,
            esm=observation,
            simulation=observation,
        )

        area = read_regional_table(
            filename=area_file,
            year=year,
            esm=observation,
            simulation=observation,
        )

        intensity = read_regional_table(
            filename=intensity_file,
            year=year,
            esm=observation,
            simulation=observation,
        )

        event_frames.append(event_processed)
        area_frames.append(area)
        intensity_frames.append(intensity)

    events = pd.concat(
        event_frames,
        ignore_index=True,
    )

    areas = rename_area_columns(
        pd.concat(
            area_frames,
            ignore_index=True,
        )
    )

    intensities = rename_temperature_columns(
        pd.concat(
            intensity_frames,
            ignore_index=True,
        )
    )

    return combine_tables(
        event_data=events,
        area_data=areas,
        intensity_data=intensities,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Historical and future ESM-driven datasets
    # --------------------------------------------------------

    for simulation in SIMULATIONS:

        for period_name, years in PERIODS.items():

            combined = process_esm_dataset(
                simulation=simulation,
                period_name=period_name,
                years=years,
            )

            first_year = years.start
            last_year = years.stop - 1

            output_file = (
                OUTPUT_ROOT
                / (
                    f"{simulation}_{period_name}_all_"
                    f"{first_year}-{last_year}_hw.csv"
                )
            )

            combined.to_csv(
                output_file,
                index=False,
            )

            print(f"\nSaved: {output_file}")

    # --------------------------------------------------------
    # Historical observations
    # --------------------------------------------------------

    for observation, years in OBSERVATION_PERIODS.items():

        combined = process_observation(
            observation=observation,
            years=years,
        )

        first_year = years.start
        last_year = years.stop - 1

        output_file = (
            OUTPUT_ROOT
            / (
                f"{observation}_{first_year}-"
                f"{last_year}_hw.csv"
            )
        )

        combined.to_csv(
            output_file,
            index=False,
        )

        print(f"\nSaved: {output_file}")

    print("\nAll heat-wave processing completed.")


if __name__ == "__main__":
    main()
