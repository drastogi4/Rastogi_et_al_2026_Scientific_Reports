#!/usr/bin/env python3
"""
Create the observed heatwave scatter plot used in Figure 1.

The script reads event-level heatwave data for an observational dataset,
converts regional columns from wide to long format, averages daily values
within each heatwave event and EIA region, and plots:

    x-axis: heatwave intensity (regional mean maximum temperature)
    y-axis: electricity demand
    marker size: percentage of the region affected by the heatwave
    marker color: EIA region

Default input
-------------
./csvfiles/DaymetV4_1980-2022_hw.csv

Default output
--------------
Demand_intensity_percentarea_DaymetV4.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OBSERVATION = "DaymetV4"
INPUT_FILE = Path(f"./csvfiles/{OBSERVATION}_1980-2022_hw.csv")
OUTPUT_FILE = Path(f"Demand_intensity_percentarea_{OBSERVATION}.pdf")

REGION_ORDER = [
    "CAR", "CENT", "FL", "MIDA", "MIDW", "NE", "NY",
    "SE", "TEN", "SW", "CAL", "TEX", "NW",
]

# Regional source-column mappings. Florida uses D_FLA in the input file.
REGION_COLUMNS = {
    "CAR": ("pareaCAR", "tmaxCAR", "D_CAR"),
    "CENT": ("pareaCENT", "tmaxCENT", "D_CENT"),
    "FL": ("pareaFL", "tmaxFL", "D_FLA"),
    "MIDA": ("pareaMIDA", "tmaxMIDA", "D_MIDA"),
    "MIDW": ("pareaMIDW", "tmaxMIDW", "D_MIDW"),
    "NE": ("pareaNE", "tmaxNE", "D_NE"),
    "NY": ("pareaNY", "tmaxNY", "D_NY"),
    "SE": ("pareaSE", "tmaxSE", "D_SE"),
    "TEN": ("pareaTEN", "tmaxTEN", "D_TEN"),
    "SW": ("pareaSW", "tmaxSW", "D_SW"),
    "CAL": ("pareaCAL", "tmaxCAL", "D_CAL"),
    "TEX": ("pareaTEX", "tmaxTEX", "D_TEX"),
    "NW": ("pareaNW", "tmaxNW", "D_NW"),
}

CUSTOM_COLORS = [
    "#F69999", "#5F98C6", "#AFCBE3", "#AD71B5", "#D6B8DA",
    "#F57E20", "#EC008C", "#F799D1", "#00AEEF", "#34A048",
    "#B35B28", "#000000", "#777777",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when required input columns are missing."""
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
    Convert regional heatwave and demand columns from wide to long format.

    Each output row represents one heatwave event, year, and EIA region.
    Daily rows belonging to the same event are averaged, matching the
    behavior of the original notebook.
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

    # The source files use -999 as a missing-value flag.
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
    """Create and save the Figure 1 demand-intensity-area scatter plot."""
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
        "Heatwave intensity",
        "Electricity demand",
    )

    grid.fig.tight_layout()
    grid.fig.savefig(output_file, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    """Read the observational data and generate the Figure 1 scatter plot."""
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
