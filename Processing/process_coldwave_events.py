#!/usr/bin/env python3

"""
process_coldwave_events.py

Combine TempestExtremes cold-event statistics, regional affected area,
and regional mean Tmin into analysis-ready CSV files.

Periods
-------
ESM-driven and downscaled datasets:
    Historical cold seasons: 1980-2018
    Future cold seasons:     2020-2058

Observations:
    Daymet v4: 1980-2018
    Livneh v2: 1980-2017

A cold season beginning in YEAR contains:
    November-December of YEAR
    January-March of YEAR + 1

Requirements
------------
Python 3
pandas
numpy
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
VARIABLE = "tmin"
THRESHOLD = "p05"

MINIMUM_DURATION = 3
MINIMUM_SIZE = 12500


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


# These are cold-season starting years.
PERIODS = {
    "historical": range(1980, 2019),  # 1980-2018
    "future": range(2020, 2059),      # 2020-2058
}


# Observations do not have future projections.
# Adjust the final year if your observational files extend farther.
OBSERVATION_PERIODS = {
    "DaymetV4": range(1980, 2019),  # 1980-2018
    "Livneh": range(1980, 2018),    # 1980-2017
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
    "season_doy",
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
    "season_doy",
    *REGIONS,
)


# ============================================================
# Filename helpers
# ============================================================

def get_ensemble(esm: str) -> str:
    """Return the ensemble member used for each ESM."""

    if esm == "CNRM-ESM2-1":
        return "r1i1p1f2"

    return "r1i1p1f1"


def build_esm_prefix(
    year: int,
    esm: str,
    simulation: str,
) -> str:
    """
    Construct the TempestExtremes filename prefix for an
    ESM-driven dataset.
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
    Construct the TempestExtremes filename prefix for an
    observational dataset.
    """

    return (
        f"{observation}_{VARIABLE}_{year}_{THRESHOLD}_"
        f"daily_mask_track_{MINIMUM_DURATION}days_{MINIMUM_SIZE}"
    )


# ============================================================
# Cold-season time conversion
# ============================================================

def convert_cold_season_time(
    dataframe: pd.DataFrame,
    season_start_year: int,
    source_column: str = "season_doy",
) -> pd.DataFrame:
    """
    Convert the zero-based November-March seasonal index into
    calendar year and zero-based calendar day of year.

    Seasonal index:
        0-60   -> November 1-December 31 of season_start_year
        61-150 -> January 1-March 31 of season_start_year + 1

    This follows the fixed-index seasonal extraction:
        tmin(304:364) from YEAR
        tmin(0:89) from YEAR + 1

    Therefore:
        November 1  -> calendar doy 304
        December 31 -> calendar doy 364
        January 1   -> calendar doy 0
        March 31    -> calendar doy 89
    """

    result = dataframe.copy()

    season_doy = pd.to_numeric(
        result[source_column],
        errors="coerce",
    )

    result = result.loc[season_doy.notna()].copy()
    season_doy = season_doy.loc[result.index].astype(int)

    november_december = season_doy <= 60

    result["year"] = np.where(
        november_december,
        season_start_year,
        season_start_year + 1,
    )

    result["doy"] = np.where(
        november_december,
        season_doy + 304,
        season_doy - 61,
    )

    result["season_year"] = season_start_year
    result["season_doy"] = season_doy

    result["year"] = result["year"].astype(int)
    result["doy"] = result["doy"].astype(int)
    result["season_year"] = result["season_year"].astype(int)
    result["season_doy"] = result["season_doy"].astype(int)

    return result


# ============================================================
# File readers
# ============================================================

def read_event_statistics(
    filename: Path,
) -> pd.DataFrame:
    """Read TempestExtremes BlobStats output."""

    if not filename.is_file():
        raise FileNotFoundError(
            f"Missing TempestExtremes statistics file:\n{filename}"
        )

    return pd.read_csv(
        filename,
        sep=r"\s+",
        engine="python",
        header=None,
        skiprows=1,
        names=EVENT_COLUMNS,
    )


def read_regional_table(
    filename: Path,
    season_start_year: int,
    esm: str,
    simulation: str,
) -> pd.DataFrame:
    """
    Read a regional affected-area or regional-temperature file.
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

    dataframe = dataframe.dropna(
        subset=["ID", "season_doy"]
    ).copy()

    dataframe["ID"] = dataframe["ID"].astype(int)

    dataframe = convert_cold_season_time(
        dataframe=dataframe,
        season_start_year=season_start_year,
        source_column="season_doy",
    )

    dataframe["esm"] = esm
    dataframe["sim"] = simulation

    return dataframe


# ============================================================
# Event processing
# ============================================================

def process_event_statistics(
    dataframe: pd.DataFrame,
    season_start_year: int,
    esm: str,
    simulation: str,
) -> pd.DataFrame:
    """
    Assign event IDs and calculate event duration.

    Missing centroid-latitude rows in the BlobStats output are
    interpreted as separators between event blocks, reproducing
    the original workflow.
    """

    dataframe = dataframe.copy()

    dataframe["ID"] = (
        dataframe["clat"]
        .isna()
        .cumsum()
    )

    dataframe = dataframe[
        [
            "ID",
            "season_doy",
            "clat",
            "clon",
            "area",
        ]
    ].copy()

    dataframe = dataframe.dropna(
        subset=[
            "season_doy",
            "clat",
            "clon",
            "area",
        ]
    ).copy()

    dataframe["ID"] = dataframe["ID"].astype(int)

    dataframe = convert_cold_season_time(
        dataframe=dataframe,
        season_start_year=season_start_year,
        source_column="season_doy",
    )

    dataframe["esm"] = esm
    dataframe["sim"] = simulation

    # Number of daily records associated with each event.
    dataframe["duration"] = (
        dataframe.groupby(
            [
                "season_year",
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
    """Rename regional area columns using the parea prefix."""

    return dataframe.rename(
        columns={
            region: f"parea{region}"
            for region in REGIONS
        }
    )


def rename_temperature_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Rename regional temperature columns using the tmin prefix."""

    return dataframe.rename(
        columns={
            region: f"tmin{region}"
            for region in REGIONS
        }
    )


def combine_tables(
    event_data: pd.DataFrame,
    area_data: pd.DataFrame,
    intensity_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge event, area, and intensity information."""

    merge_columns = [
        "season_doy",
        "season_year",
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
            "season_year",
            "ID",
            "season_doy",
        ]
    ).reset_index(drop=True)


# ============================================================
# Process an ESM-driven dataset
# ============================================================

def process_esm_dataset(
    simulation: str,
    period_name: str,
    years: range,
) -> pd.DataFrame:
    """Process one simulation across all ESMs and years."""

    event_frames = []
    area_frames = []
    intensity_frames = []

    for esm in ESMS:

        print(
            f"\nProcessing {simulation}, "
            f"{esm}, {period_name}"
        )

        for year in years:

            print(f"  Cold season {year}-{year + 1}")

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
                / f"{prefix}_CONUS_avgtmin.txt"
            )

            event_raw = read_event_statistics(
                event_file
            )

            event_processed = process_event_statistics(
                dataframe=event_raw,
                season_start_year=year,
                esm=esm,
                simulation=simulation,
            )

            area = read_regional_table(
                filename=area_file,
                season_start_year=year,
                esm=esm,
                simulation=simulation,
            )

            intensity = read_regional_table(
                filename=intensity_file,
                season_start_year=year,
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
# Process Daymet or Livneh
# ============================================================

def process_observation(
    observation: str,
    years: range,
) -> pd.DataFrame:
    """
    Process one observational dataset.

    The observation name is stored in both the esm and sim columns
    so that its output has the same structure as the model output.
    """

    event_frames = []
    area_frames = []
    intensity_frames = []

    print(f"\nProcessing observation: {observation}")

    for year in years:

        print(f"  Cold season {year}-{year + 1}")

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
            / f"{prefix}_CONUS_avgtmin.txt"
        )

        event_raw = read_event_statistics(
            event_file
        )

        event_processed = process_event_statistics(
            dataframe=event_raw,
            season_start_year=year,
            esm=observation,
            simulation=observation,
        )

        area = read_regional_table(
            filename=area_file,
            season_start_year=year,
            esm=observation,
            simulation=observation,
        )

        intensity = read_regional_table(
            filename=intensity_file,
            season_start_year=year,
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
    """Process all cold-wave datasets."""

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
                    f"{first_year}-{last_year}_cw.csv"
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
                f"{last_year}_cw.csv"
            )
        )

        combined.to_csv(
            output_file,
            index=False,
        )

        print(f"\nSaved: {output_file}")

    print("\nAll cold-wave processing completed.")


if __name__ == "__main__":
    main()
