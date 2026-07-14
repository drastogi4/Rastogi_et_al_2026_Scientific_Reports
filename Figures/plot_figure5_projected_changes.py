#!/usr/bin/env python3
"""
plot_figure5_projected_changes.py

Generate Figure 5 of the manuscript: multimodel-mean projected changes in
heat-wave and cold-wave characteristics.

The figure contains eight heatmaps arranged as two rows by four columns:

Top row: heat-wave changes
    (a) affected area
    (b) intensity
    (c) duration
    (d) frequency

Bottom row: cold-wave changes
    (e) affected area
    (f) intensity
    (g) duration
    (h) frequency

Projected changes are calculated as:

    future mean - historical mean

for each ESM, downscaling experiment, and region. The resulting changes are
then averaged across ESMs to produce a multimodel mean for each experiment.

Inputs
------
CSV files produced by:
    process_heatwave_events.py
    process_coldwave_events.py

Expected heat-wave files:
    <simulation>_historical_all_1980-2019_hw.csv
    <simulation>_future_all_2020-2059_hw.csv

Expected cold-wave files:
    <simulation>_historical_all_1980-2018_cw.csv
    <simulation>_future_all_2020-2058_cw.csv

Outputs
-------
1. figure5_projected_changes.pdf
2. figure5_projected_changes.png
3. One CSV file for each heatmap panel.

Requirements
------------
Python 3.9+
numpy
pandas
matplotlib
seaborn
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# Configuration
# ============================================================

CSV_DIRECTORY = Path("./csvfiles")
OUTPUT_FIGURE = Path("figure5_projected_changes.pdf")

SIMULATIONS = [
    "CMIP6",
    "RegCM",
    "RegCM_Daymet",
    "DBCCA_Daymet",
    "SRCNN_Daymet",
    "SRGAN_Daymet",
    "RegCM_Livneh",
    "DBCCA_Livneh",
    "LOCA_Livneh",
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

REGION_CONFIG = {
    "CONUS": "pareaCONUS",
    "NE": "pareaNE",
    "NY": "pareaNY",
    "MIDA": "pareaMIDA",
    "MIDW": "pareaMIDW",
    "SE": "pareaSE",
    "CENT": "pareaCENT",
    "FL": "pareaFL",
    "CAR": "pareaCAR",
    "TEN": "pareaTEN",
    "SW": "pareaSW",
    "CAL": "pareaCAL",
    "TEX": "pareaTEX",
    "NW": "pareaNW",
}

# Preserve the plotting ranges used in the original scripts.
HEAT_LIMITS = {
    "percentarea_diff": (-20, 20),
    "intensity_diff": (-2, 2),
    "length_diff": (-16, 16),
    "count_diff": (-2, 2),
}

COLD_LIMITS = {
    "percentarea_diff": (-15, 15),
    "intensity_diff": (-2, 2),
    "length_diff": (-2, 2),
    "count_diff": (-4, 4),
}


# ============================================================
# Input handling
# ============================================================

def standardize_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize metadata names used by older and newer processing scripts.
    """

    dataframe = dataframe.copy()



    required = {
        "esm",
        "sim",
        "ID",
        "year",
        "doy",
        "area",
        "duration",
        "clat",
        "clon",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            "Input dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return dataframe


def read_period_files(
    event_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read and combine historical and future CSV files for all simulations.

    Parameters
    ----------
    event_type
        Either "heatwave" or "coldwave".
    """

    if event_type == "heatwave":
        historical_suffix = "historical_all_1980-2019_hw.csv"
        future_suffix = "future_all_2020-2059_hw.csv"
    elif event_type == "coldwave":
        historical_suffix = "historical_all_1980-2018_cw.csv"
        future_suffix = "future_all_2020-2058_cw.csv"
    else:
        raise ValueError(
            "event_type must be either 'heatwave' or 'coldwave'."
        )

    historical_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []

    for simulation in SIMULATIONS:
        historical_file = (
            CSV_DIRECTORY / f"{simulation}_{historical_suffix}"
        )
        future_file = (
            CSV_DIRECTORY / f"{simulation}_{future_suffix}"
        )

        if not historical_file.is_file():
            raise FileNotFoundError(
                f"Missing historical file:\n{historical_file}"
            )

        if not future_file.is_file():
            raise FileNotFoundError(
                f"Missing future file:\n{future_file}"
            )

        historical = standardize_metadata(
            pd.read_csv(historical_file)
        )
        future = standardize_metadata(
            pd.read_csv(future_file)
        )

        # Standardize experiment labels using the requested identifier.
        historical.loc[:, "sim"] = simulation
        future.loc[:, "sim"] = simulation

        historical_frames.append(historical)
        future_frames.append(future)

    return (
        pd.concat(historical_frames, ignore_index=True),
        pd.concat(future_frames, ignore_index=True),
    )


# ============================================================
# Regional summaries
# ============================================================

def summarize_events(
    dataframe: pd.DataFrame,
    intensity_prefix: str,
) -> pd.DataFrame:
    """
    Convert wide regional event data into long format and calculate mean
    event characteristics for each ESM, experiment, and region.

    Parameters
    ----------
    dataframe
        Combined historical or future event dataframe.

    intensity_prefix
        "tmax" for heat waves or "tmin" for cold waves.
    """

    regional_frames: list[pd.DataFrame] = []

    base_columns = [
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

    for region, area_column in REGION_CONFIG.items():
        intensity_column = f"{intensity_prefix}{region}"

        required_columns = (
            base_columns + [area_column, intensity_column]
        )

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns for region {region}: "
                + ", ".join(missing)
            )

        region_df = dataframe[required_columns].copy()
        region_df["Region"] = region

        region_df = region_df.rename(
            columns={
                area_column: "percentarea",
                intensity_column: "intensity",
                "esm": "model",
                "sim": "experiment",
            }
        )

        region_df = region_df.replace(-999.0, np.nan)

        region_df = region_df.dropna(
            subset=[
                "model",
                "experiment",
                "ID",
                "year",
                "percentarea",
                "intensity",
            ]
        )

        regional_frames.append(region_df)

    long_df = pd.concat(regional_frames, ignore_index=True)

    # Exclude rows where the event does not affect the region.
    long_df = long_df.loc[
        long_df["percentarea"] > 0
    ].copy()

    grouping_columns = [
        "model",
        "experiment",
        "Region",
    ]

    summary = (
        long_df.groupby(grouping_columns, observed=True)[
            ["percentarea", "intensity"]
        ]
        .mean()
        .reset_index()
    )

    # Mean event duration.
    event_duration = (
        long_df.groupby(
            [
                "model",
                "experiment",
                "Region",
                "year",
                "ID",
            ],
            observed=True,
        )["ID"]
        .count()
        .reset_index(name="length")
    )

    mean_duration = (
        event_duration.groupby(
            grouping_columns,
            observed=True,
        )[["length"]]
        .mean()
        .reset_index()
    )

    # Mean annual event frequency.
    annual_frequency = (
        long_df.groupby(
            [
                "model",
                "experiment",
                "Region",
                "year",
            ],
            observed=True,
        )[["ID"]]
        .nunique()
        .reset_index()
    )

    mean_frequency = (
        annual_frequency.groupby(
            grouping_columns,
            observed=True,
        )[["ID"]]
        .mean()
        .reset_index()
    )

    summary = summary.merge(
        mean_duration,
        on=grouping_columns,
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        mean_frequency,
        on=grouping_columns,
        how="left",
        validate="one_to_one",
    )

    return summary


def calculate_multimodel_changes(
    historical: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate future-minus-historical changes and average them across ESMs.
    """

    historical = historical.rename(
        columns={
            "percentarea": "percentarea_historical",
            "intensity": "intensity_historical",
            "length": "length_historical",
            "ID": "count_historical",
        }
    )

    future = future.rename(
        columns={
            "percentarea": "percentarea_future",
            "intensity": "intensity_future",
            "length": "length_future",
            "ID": "count_future",
        }
    )

    merge_columns = [
        "model",
        "experiment",
        "Region",
    ]

    combined = historical.merge(
        future,
        on=merge_columns,
        how="inner",
        validate="one_to_one",
    )

    combined["percentarea_diff"] = (
        combined["percentarea_future"]
        - combined["percentarea_historical"]
    )

    combined["intensity_diff"] = (
        combined["intensity_future"]
        - combined["intensity_historical"]
    )

    combined["length_diff"] = (
        combined["length_future"]
        - combined["length_historical"]
    )

    combined["count_diff"] = (
        combined["count_future"]
        - combined["count_historical"]
    )

    multimodel_mean = (
        combined.groupby(
            ["experiment", "Region"],
            observed=True,
        )[
            [
                "percentarea_diff",
                "intensity_diff",
                "length_diff",
                "count_diff",
            ]
        ]
        .mean()
        .reset_index()
    )

    multimodel_mean["experiment"] = pd.Categorical(
        multimodel_mean["experiment"],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )

    multimodel_mean["Region"] = pd.Categorical(
        multimodel_mean["Region"],
        categories=REGION_ORDER,
        ordered=True,
    )

    multimodel_mean = multimodel_mean.sort_values(
        ["experiment", "Region"]
    ).reset_index(drop=True)

    return multimodel_mean


# ============================================================
# Heatmap preparation
# ============================================================

def make_panel_matrix(
    dataframe: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Create an experiment-by-region matrix for one figure panel."""

    matrix = dataframe.pivot(
        index="experiment",
        columns="Region",
        values=value_column,
    )

    matrix = matrix.reindex(
        index=EXPERIMENT_ORDER,
        columns=REGION_ORDER,
    )

    matrix.index = [
        DISPLAY_NAMES.get(value, value)
        for value in matrix.index
    ]

    return matrix


def plot_heatmap_panel(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    limits: tuple[float, float],
    show_ylabels: bool,
    cbar_label: str,
) -> None:
    """Plot one projected-change heatmap panel."""

    vmin, vmax = limits

    heatmap = sns.heatmap(
        matrix,
        ax=axis,
        cmap="vlag",
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={
            "shrink": 0.65,
            "label": cbar_label,
        },
    )

    heatmap.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    heatmap.set_xlabel("")
    heatmap.set_ylabel("")

    heatmap.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=9,
    )

    if show_ylabels:
        heatmap.tick_params(
            axis="y",
            labelrotation=0,
            labelsize=9,
        )
    else:
        heatmap.set_yticklabels([])


# ============================================================
# Figure generation
# ============================================================

def plot_figure(
    heat_changes: pd.DataFrame,
    cold_changes: pd.DataFrame,
    output_file: Path = OUTPUT_FIGURE,
) -> None:
    """
    Create the combined 2 x 4 projected-change figure.
    """

    heat_matrices = {
        "percentarea_diff": make_panel_matrix(
            heat_changes,
            "percentarea_diff",
        ),
        "intensity_diff": make_panel_matrix(
            heat_changes,
            "intensity_diff",
        ),
        "length_diff": make_panel_matrix(
            heat_changes,
            "length_diff",
        ),
        "count_diff": make_panel_matrix(
            heat_changes,
            "count_diff",
        ),
    }

    cold_matrices = {
        "percentarea_diff": make_panel_matrix(
            cold_changes,
            "percentarea_diff",
        ),
        "intensity_diff": make_panel_matrix(
            cold_changes,
            "intensity_diff",
        ),
        "length_diff": make_panel_matrix(
            cold_changes,
            "length_diff",
        ),
        "count_diff": make_panel_matrix(
            cold_changes,
            "count_diff",
        ),
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(24, 11),
    )

    top_titles = [
        "(a) Heat-wave affected area",
        "(b) Heat-wave intensity",
        "(c) Heat-wave duration",
        "(d) Heat-wave frequency",
    ]

    bottom_titles = [
        "(e) Cold-wave affected area",
        "(f) Cold-wave intensity",
        "(g) Cold-wave duration",
        "(h) Cold-wave frequency",
    ]

    value_columns = [
        "percentarea_diff",
        "intensity_diff",
        "length_diff",
        "count_diff",
    ]

    colorbar_labels = [
        "Change in affected area (%)",
        "Change in temperature (°C)",
        "Change in duration (days)",
        r"Change in frequency (events year$^{-1}$)",
    ]

    for column_index, value_column in enumerate(value_columns):
        plot_heatmap_panel(
            axis=axes[0, column_index],
            matrix=heat_matrices[value_column],
            title=top_titles[column_index],
            limits=HEAT_LIMITS[value_column],
            show_ylabels=(column_index == 0),
            cbar_label=colorbar_labels[column_index],
        )

        plot_heatmap_panel(
            axis=axes[1, column_index],
            matrix=cold_matrices[value_column],
            title=bottom_titles[column_index],
            limits=COLD_LIMITS[value_column],
            show_ylabels=(column_index == 0),
            cbar_label=colorbar_labels[column_index],
        )

    axes[0, 0].set_ylabel(
        "Downscaling experiment",
        fontsize=11,
    )
    axes[1, 0].set_ylabel(
        "Downscaling experiment",
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.14,
        top=0.94,
        wspace=0.38,
        hspace=0.38,
    )

    output_file = Path(output_file)
    output_root = output_file.with_suffix("")

    # Save panel matrices for reproducibility.
    heat_matrices["percentarea_diff"].to_csv(
        f"{output_root}_panel_a_heatwave_affected_area.csv"
    )
    heat_matrices["intensity_diff"].to_csv(
        f"{output_root}_panel_b_heatwave_intensity.csv"
    )
    heat_matrices["length_diff"].to_csv(
        f"{output_root}_panel_c_heatwave_duration.csv"
    )
    heat_matrices["count_diff"].to_csv(
        f"{output_root}_panel_d_heatwave_frequency.csv"
    )

    cold_matrices["percentarea_diff"].to_csv(
        f"{output_root}_panel_e_coldwave_affected_area.csv"
    )
    cold_matrices["intensity_diff"].to_csv(
        f"{output_root}_panel_f_coldwave_intensity.csv"
    )
    cold_matrices["length_diff"].to_csv(
        f"{output_root}_panel_g_coldwave_duration.csv"
    )
    cold_matrices["count_diff"].to_csv(
        f"{output_root}_panel_h_coldwave_frequency.csv"
    )

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
# Main
# ============================================================

def main() -> None:
    """Read data, calculate changes, and generate Figure 5."""

    heat_historical_raw, heat_future_raw = read_period_files(
        event_type="heatwave"
    )

    cold_historical_raw, cold_future_raw = read_period_files(
        event_type="coldwave"
    )

    heat_historical = summarize_events(
        heat_historical_raw,
        intensity_prefix="tmax",
    )

    heat_future = summarize_events(
        heat_future_raw,
        intensity_prefix="tmax",
    )

    cold_historical = summarize_events(
        cold_historical_raw,
        intensity_prefix="tmin",
    )

    cold_future = summarize_events(
        cold_future_raw,
        intensity_prefix="tmin",
    )

    heat_changes = calculate_multimodel_changes(
        heat_historical,
        heat_future,
    )

    cold_changes = calculate_multimodel_changes(
        cold_historical,
        cold_future,
    )

    plot_figure(
        heat_changes=heat_changes,
        cold_changes=cold_changes,
        output_file=OUTPUT_FIGURE,
    )


if __name__ == "__main__":
    main()
