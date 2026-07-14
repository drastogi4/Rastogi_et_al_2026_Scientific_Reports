#!/usr/bin/env python3
"""
plot_figure3_heatwave_historical.py

Generate Figure 3 of the manuscript: historical heat-wave characteristics
across observations, CMIP6, and high-resolution downscaled datasets.

The figure compares four metrics across U.S. EIA regions:
    (a) affected area
    (b) heat-wave intensity
    (c) event duration
    (d) annual event frequency

Inputs
------
CSV files produced by the heat-wave post-processing workflow and stored in
CSV_DIRECTORY.

Expected files:
    <simulation>_historical_all_1980-2019_hw.csv
    DaymetV4_1980-2022_hw.csv
    Livneh_1980-2018_hw.csv

Outputs
-------
1. Publication-quality PDF and PNG figures.
2. CSV files containing the plotted values for each panel.

Requirements
------------
Python 3.9+
numpy
pandas
matplotlib
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

CSV_DIRECTORY = Path("./csvfiles")
OUTPUT_FIGURE = Path("figure3_historical_heatwaves.pdf")
VERBOSE = True


REGION_CONFIG = {
    "CONUS": ("pareaCONUS", "tmaxCONUS"),
    "NE": ("pareaNE", "tmaxNE"),
    "NY": ("pareaNY", "tmaxNY"),
    "MIDA": ("pareaMIDA", "tmaxMIDA"),
    "MIDW": ("pareaMIDW", "tmaxMIDW"),
    "SE": ("pareaSE", "tmaxSE"),
    "CENT": ("pareaCENT", "tmaxCENT"),
    "FL": ("pareaFL", "tmaxFL"),
    "CAR": ("pareaCAR", "tmaxCAR"),
    "TEN": ("pareaTEN", "tmaxTEN"),
    "SW": ("pareaSW", "tmaxSW"),
    "CAL": ("pareaCAL", "tmaxCAL"),
    "TEX": ("pareaTEX", "tmaxTEX"),
    "NW": ("pareaNW", "tmaxNW"),
}


REGION_ORDER = [
    "NE",
    "NY",
    "MIDA",
    "MIDW",
    "SE",
    "CENT",
    "FL",
    "CAR",
    "TEN",
    "SW",
    "CAL",
    "TEX",
    "NW",
]


EXPERIMENT_ORDER = [
    "CMIP6",
    "RegCM",
    "RegCM_Daymet",
    "DBCCA_Daymet",
    "SRCNN_Daymet",
    "SRGAN_Daymet",
    "RegCM_Livneh",
    "DBCCA_Livneh",
    "LOCA_Livneh",
    "Livneh",
    "DaymetV4",
]


DISPLAY_NAMES = {
    "CMIP6": "CMIP6",
    "RegCM": "RegCM",
    "RegCM_Daymet": "RegCM Daymet",
    "DBCCA_Daymet": "DBCCA Daymet",
    "SRCNN_Daymet": "SRCNN Daymet",
    "SRGAN_Daymet": "SRGAN Daymet",
    "RegCM_Livneh": "RegCM Livneh v2",
    "DBCCA_Livneh": "DBCCA Livneh v2",
    "LOCA_Livneh": "LOCA2 Livneh",
    "Livneh": "Livneh v2",
    "DaymetV4": "Daymet v4",
}


COLORS = {
    "CMIP6": "#000000",
    "RegCM": "#1f77b4",
    "RegCM_Daymet": "#17becf",
    "DBCCA_Daymet": "#d62728",
    "SRCNN_Daymet": "#9467bd",
    "SRGAN_Daymet": "#ff7f0e",
    "RegCM_Livneh": "#2ca02c",
    "DBCCA_Livneh": "#8c564b",
    "LOCA_Livneh": "#e377c2",
    "Livneh": "#7f7f7f",
    "DaymetV4": "#bcbd22",
}


MARKERS = {
    "CMIP6": "o",
    "RegCM": "s",
    "RegCM_Daymet": "^",
    "DBCCA_Daymet": "D",
    "SRCNN_Daymet": "P",
    "SRGAN_Daymet": "X",
    "RegCM_Livneh": "v",
    "DBCCA_Livneh": "<",
    "LOCA_Livneh": ">",
    "Livneh": "*",
    "DaymetV4": "h",
}


# ============================================================
# Data processing
# ============================================================

def summarize_heatwave_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide regional event table into long format and calculate
    mean historical heat-wave characteristics for each ESM-experiment pair.
    """

    required_base_columns = [
        "esm",
        "sim",
        "ID",
        "year",
        "doy",
        "area",
        "duration",
        "clat",
        "clon",
    ]

    missing_base = [
        column for column in required_base_columns if column not in df.columns
    ]

    if missing_base:
        raise ValueError(
            "Input dataframe is missing required columns: "
            + ", ".join(missing_base)
        )

    regional_frames: list[pd.DataFrame] = []

    for region, (area_column, intensity_column) in REGION_CONFIG.items():
        required_columns = required_base_columns + [
            area_column,
            intensity_column,
        ]

        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns for region {region}: "
                + ", ".join(missing_columns)
            )

        region_df = df[required_columns].copy()
        region_df["Region"] = region

        region_df = region_df.rename(
            columns={
                area_column: "percentarea",
                intensity_column: "intensity",
            }
        )

        region_df = region_df.replace(-999.0, np.nan)

        region_df = region_df.dropna(
            subset=[
                "esm",
                "sim",
                "ID",
                "year",
                "duration",
                "percentarea",
                "intensity",
            ]
        )

        regional_frames.append(region_df)

    df_long = pd.concat(regional_frames, ignore_index=True)

    df_long = df_long.rename(
        columns={
            "esm": "model",
            "sim": "experiment",
        }
    )

    # Retain only events that affect the region.
    df_events = df_long.loc[df_long["percentarea"] > 0].copy()

    # Use a separator to avoid ambiguous concatenated names.
    df_events["modelexperiment"] = (
        df_events["model"].astype(str)
        + "_"
        + df_events["experiment"].astype(str)
    )

    grouping_columns = [
        "modelexperiment",
        "Region",
        "experiment",
    ]

    # Mean affected area, event intensity, and supplied duration field.
    summary = (
        df_events.groupby(grouping_columns, observed=True)[
            ["duration", "percentarea", "intensity"]
        ]
        .mean()
        .reset_index()
    )

    # Mean event duration based on the number of daily records per event.
    event_duration = (
        df_events.groupby(
            [
                "modelexperiment",
                "Region",
                "year",
                "ID",
                "experiment",
            ],
            observed=True,
        )["ID"]
        .count()
        .reset_index(name="length")
    )

    mean_duration = (
        event_duration.groupby(grouping_columns, observed=True)[["length"]]
        .mean()
        .reset_index()
    )

    # Mean annual event frequency.
    annual_frequency = (
        df_events.groupby(
            [
                "modelexperiment",
                "Region",
                "year",
                "experiment",
            ],
            observed=True,
        )[["ID"]]
        .nunique()
        .reset_index()
    )

    mean_frequency = (
        annual_frequency.groupby(grouping_columns, observed=True)[["ID"]]
        .mean()
        .reset_index()
    )

    summary = summary.merge(
        mean_frequency,
        on=grouping_columns,
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        mean_duration,
        on=grouping_columns,
        how="left",
        validate="one_to_one",
    )

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_historical_stripplots(
    dfavg: pd.DataFrame,
    output_file: Path = OUTPUT_FIGURE,
) -> None:
    """
    Plot historical heat-wave characteristics using distinct colors,
    marker shapes, and vertical offsets for each dataset.
    """

    required_columns = {
        "Region",
        "experiment",
        "modelexperiment",
        "percentarea",
        "intensity",
        "length",
        "ID",
    }

    missing_columns = required_columns.difference(dfavg.columns)

    if missing_columns:
        raise ValueError(
            "dfavg is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    metrics = [
        ("percentarea", "Affected area (%)", "(a) Affected area"),
        ("intensity", "Heat-wave intensity (°C)", "(b) Intensity"),
        ("length", "Duration (days)", "(c) Duration"),
        ("ID", r"Frequency (events year$^{-1}$)", "(d) Frequency"),
    ]

    dfplot = dfavg.loc[
        dfavg["Region"].isin(REGION_ORDER)
        & dfavg["experiment"].isin(EXPERIMENT_ORDER)
    ].copy()

    if dfplot.empty:
        raise ValueError(
            "No rows remain after filtering by region and experiment."
        )

    missing_experiments = [
        experiment
        for experiment in EXPERIMENT_ORDER
        if experiment not in dfplot["experiment"].unique()
    ]

    if missing_experiments:
        print(
            "Warning: no plotted data were found for: "
            + ", ".join(missing_experiments)
        )

    for variable in ["percentarea", "intensity", "length", "ID"]:
        dfplot[variable] = pd.to_numeric(
            dfplot[variable],
            errors="coerce",
        )

    region_positions = {
        region: index for index, region in enumerate(REGION_ORDER)
    }

    offsets = np.linspace(
        -0.42,
        0.42,
        len(EXPERIMENT_ORDER),
    )

    experiment_offsets = dict(zip(EXPERIMENT_ORDER, offsets))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 12),
        sharey=True,
    )

    axes = axes.ravel()

    for ax, (variable, xlabel, panel_title) in zip(axes, metrics):
        for experiment in EXPERIMENT_ORDER:
            subset = dfplot.loc[
                dfplot["experiment"] == experiment
            ].dropna(subset=[variable])

            if subset.empty:
                continue

            y_values = np.array(
                [
                    region_positions[region]
                    for region in subset["Region"]
                ],
                dtype=float,
            )

            y_values += experiment_offsets[experiment]

            if experiment in {"Livneh", "DaymetV4"}:
                marker_size = 120
                zorder = 6
            else:
                marker_size = 95
                zorder = 4

            ax.scatter(
                subset[variable],
                y_values,
                s=marker_size,
                marker=MARKERS[experiment],
                facecolor=COLORS[experiment],
                edgecolor="none",
                linewidth=0,
                alpha=1.0,
                zorder=zorder,
            )

        ax.set_title(
            panel_title,
            fontsize=14,
            fontweight="bold",
            loc="left",
            pad=8,
        )

        ax.set_xlabel(xlabel, fontsize=12)

        ax.set_yticks(np.arange(len(REGION_ORDER)))
        ax.set_yticklabels(REGION_ORDER, fontsize=11)

        # Add room above NE and below NW for the largest markers.
        ax.set_ylim(
            len(REGION_ORDER) - 0.15,
            -0.85,
        )

        for y_position in range(len(REGION_ORDER)):
            ax.axhline(
                y=y_position,
                color="0.88",
                linewidth=0.7,
                zorder=0,
            )

        ax.grid(
            axis="x",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
            zorder=0,
        )

        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Region", fontsize=12)
    axes[2].set_ylabel("Region", fontsize=12)
    axes[1].set_ylabel("")
    axes[3].set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[experiment],
            linestyle="None",
            markerfacecolor=COLORS[experiment],
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=12,
            label=DISPLAY_NAMES[experiment],
        )
        for experiment in EXPERIMENT_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        title="Dataset",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=12,
        title_fontsize=13,
        columnspacing=1.8,
        handletextpad=0.7,
        handlelength=1.2,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.21,
        top=0.96,
        wspace=0.16,
        hspace=0.22,
    )

    output_file = Path(output_file)
    output_root = output_file.with_suffix("")

    # Save the exact plotted values for reproducibility.
    dfplot.pivot_table(
        index="modelexperiment",
        columns="Region",
        values="percentarea",
        aggfunc="mean",
    ).to_csv(f"{output_root}_panel_a_affected_area.csv")

    dfplot.pivot_table(
        index="modelexperiment",
        columns="Region",
        values="intensity",
        aggfunc="mean",
    ).to_csv(f"{output_root}_panel_b_intensity.csv")

    dfplot.pivot_table(
        index="modelexperiment",
        columns="Region",
        values="length",
        aggfunc="mean",
    ).to_csv(f"{output_root}_panel_c_duration.csv")

    dfplot.pivot_table(
        index="modelexperiment",
        columns="Region",
        values="ID",
        aggfunc="mean",
    ).to_csv(f"{output_root}_panel_d_frequency.csv")

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    png_file = output_file.with_suffix(".png")

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved PDF: {output_file}")
    print(f"Saved PNG: {png_file}")


# ============================================================
# Input handling
# ============================================================

def read_simulation_data(
    simulation: str,
    csv_directory: Path = CSV_DIRECTORY,
) -> pd.DataFrame:
    """
    Read one historical heat-wave dataset and standardize metadata columns.
    """

    if simulation == "DaymetV4":
        filename = csv_directory / f"{simulation}_1980-2022_hw.csv"
        dataframe = pd.read_csv(filename)
        dataframe = dataframe.loc[dataframe["year"] < 2020].copy()

    elif simulation == "Livneh":
        filename = csv_directory / f"{simulation}_1980-2018_hw.csv"
        dataframe = pd.read_csv(filename).copy()

    else:
        filename = (
            csv_directory
            / f"{simulation}_historical_all_1980-2019_hw.csv"
        )
        dataframe = pd.read_csv(filename).copy()


    if "esm" not in dataframe.columns:
        raise ValueError(
            f"The file {filename} does not contain an 'esm' column."
        )

    # Standardize experiment labels using the requested dataset identifier.
    dataframe.loc[:, "sim"] = simulation

    # Observations do not have a driving ESM.
    if simulation in {"DaymetV4", "Livneh"}:
        dataframe.loc[:, "esm"] = simulation

    if VERBOSE:
        print(
            f"Read {simulation}: rows={len(dataframe)}, "
            f"ESMs={dataframe['esm'].nunique()}"
        )

    return dataframe


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Read all datasets, summarize characteristics, and generate Figure 3."""

    simulations = [
        "CMIP6",
        "RegCM",
        "RegCM_Daymet",
        "DBCCA_Daymet",
        "SRCNN_Daymet",
        "SRGAN_Daymet",
        "RegCM_Livneh",
        "DBCCA_Livneh",
        "LOCA_Livneh",
        "Livneh",
        "DaymetV4",
    ]

    dataframes = [
        read_simulation_data(
            simulation=simulation,
            csv_directory=CSV_DIRECTORY,
        )
        for simulation in simulations
    ]

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    summary = summarize_heatwave_characteristics(combined)

    plot_historical_stripplots(
        summary,
        output_file=OUTPUT_FIGURE,
    )


if __name__ == "__main__":
    main()
