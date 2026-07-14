#!/usr/bin/env python3
"""
Create the observed cold-wave scatter plot used in Figure 1.

The script reads event-level cold-wave data for the DaymetV4 observational
dataset, converts regional columns from wide to long format, averages daily
values within each cold-wave event and EIA region, and plots:

    x-axis: cold-wave intensity (regional mean minimum temperature)
    y-axis: electricity demand
    marker size: percentage of the region affected by the cold wave
    marker color: EIA region

Default input
-------------
./csvfiles/DaymetV4_1980-2022_cw.csv

Default output
--------------
Demand_intensity_percentarea_DaymetV4_cw.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OBSERVATION = "DaymetV4"
INPUT_FILE = Path(f"./csvfiles/{OBSERVATION}_1980-2022_cw.csv")
OUTPUT_FILE = Path(
    f"Demand_intensity_percentarea_{OBSERVATION}_cw.pdf"
)

REGION_ORDER = [
    "CAR", "CENT", "FL", "MIDA", "MIDW", "NE", "NY",
    "SE", "TEN", "SW", "CAL", "TEX", "NW",
]

# Source-column mappings for each EIA region.
# Florida uses D_FLA in the input file.
REGION_COLUMNS = {
    "CAR": ("pareaCAR", "tminCAR", "D_CAR"),
    "CENT": ("pareaCENT", "tminCENT", "D_CENT"),
    "FL": ("pareaFL", "tminFL", "D_FLA"),
    "MIDA": ("pareaMIDA", "tminMIDA", "D_MIDA"),
    "MIDW": ("pareaMIDW", "tminMIDW", "D_MIDW"),
    "NE": ("pareaNE", "tminNE", "D_NE"),
    "NY": ("pareaNY", "tminNY", "D_NY"),
    "SE": ("pareaSE", "tminSE", "D_SE"),
    "TEN": ("pareaTEN", "tminTEN", "D_TEN"),
    "SW": ("pareaSW", "tminSW", "D_SW"),
    "CAL": ("pareaCAL", "tminCAL", "D_CAL"),
    "TEX": ("pareaTEX", "tminTEX", "D_TEX"),
    "NW": ("pareaNW", "tminNW", "D_NW"),
}

CUSTOM_COLORS = [
    "#F69999", "#5F98C6", "#AFCBE3", "#AD71B5", "#D6B8DA",
    "#F57E20", "#EC008C", "#F799D1", "#00AEEF", "#34A048",
    "#B35B28", "#000000", "#777777",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Check that the input file contains every required column."""
    required = {"ID", "year"}

    for percent_area, intensity, demand in REGION_COLUMNS.values():
        required.update([percent_area, intensity, demand])

    missing = sorted(required.difference(df.columns))

    if missing:
        raise KeyError(
            "The input CSV is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def reshape_regional_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert regional cold-wave variables from wide to long format.

    The output contains one row per cold-wave event, year, and EIA region.
    Daily values belonging to the same event are averaged, matching the
    behavior of the original analysis script.
    """
    validate_columns(df)

    regional_frames = []

    for region, columns in REGION_COLUMNS.items():
        percent_area_col, intensity_col, demand_col = columns

        region_df = df[
            [
                "ID",
                "year",
                percent_area_col,
                intensity_col,
                demand_col,
            ]
        ].copy()

        region_df = region_df.rename(
            columns={
                percent_area_col: "percentarea",
                intensity_col: "intensity",
                demand_col: "Demand",
            }
        )

        region_df["Region"] = region
        regional_frames.append(region_df)

    long_df = pd.concat(regional_frames, ignore_index=True)

    # The source data use -999 as a missing-value flag.
    long_df = long_df.replace(-999.0, np.nan)

    long_df = long_df.dropna(
        subset=[
            "ID",
            "year",
            "Region",
            "percentarea",
            "intensity",
            "Demand",
        ]
    )

    event_means = (
        long_df
        .groupby(["ID", "year", "Region"], as_index=False)
        .mean(numeric_only=True)
    )

    return event_means


def create_scatter_plot(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Create the Figure 1 cold-wave demand-intensity-area scatter plot.

    Colder events appear farther left because intensity is represented by
    mean minimum temperature rather than heating degree days.
    """
    sns.set_theme(style="whitegrid")
    sns.set_palette(CUSTOM_COLORS)

    grid = sns.relplot(
        data=df,
        x="intensity",
        y="Demand",
        hue="Region",
        hue_order=REGION_ORDER,
        size="percentarea",
        sizes=(40, 400),
        alpha=1.0,
        height=6,
    )

    grid.set_axis_labels(
        "Cold-wave intensity",
        "Electricity demand",
    )

    # Preserve the y-axis range used in the original Figure 1 script.
    grid.set(ylim=(0, 120000))

    grid.fig.tight_layout()
    grid.fig.savefig(output_file, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    """Read the cold-wave data and generate the Figure 1 scatter plot."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    observations = pd.read_csv(INPUT_FILE)
    event_data = reshape_regional_data(observations)
    create_scatter_plot(event_data, OUTPUT_FILE)

    print(f"Saved figure: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
