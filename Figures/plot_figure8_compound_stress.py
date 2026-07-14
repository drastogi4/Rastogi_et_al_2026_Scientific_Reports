#!/usr/bin/env python3
"""
plot_figure8_compound_stress.py

Generate Figure 8 of the manuscript from the compound-stress summary.

The script creates:

1. A regional bar-and-line figure showing:
   - the magnitude of projected change in the Compound Extremes Stress
     Index (Delta_CESI);
   - the magnitude of projected change in the Cumulative Load Stress
     Index (Delta_CLSI);
   - the amplification factor, defined as Delta_CLSI / Delta_CESI.

2. A scatter plot of Delta_CESI versus Delta_CLSI for each
   ESM-simulation-region combination.

Input
-----
compound_stress_events.csv

Required columns
----------------
ESM
Simulation
Region
CESI_hist
CESI_fut
Delta_CESI
CLSI_hist
CLSI_fut
Delta_CLSI

Optional columns
----------------
REL_Delta_CESI
REL_Delta_CLSI

Outputs
-------
plots_compound/
    figure8_compound_stress.pdf
    figure8_compound_stress.png
    figure8_scatter_CESI_vs_CLSI.pdf
    figure8_scatter_CESI_vs_CLSI.png
    figure8_regional_summary.csv

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

BASE_DIR = Path(".")
COMPOUND_STRESS_CSV = BASE_DIR / "compound_stress_events.csv"

PLOTS_DIR = BASE_DIR / "plots_compound"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

REGION_ORDER = [
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
]

VERBOSE = True


# ============================================================
# Data loading and validation
# ============================================================

def load_compound_stress_data(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Load the compound-stress summary and standardize legacy column names.
    """

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Compound-stress CSV was not found:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    # Support older output files that used ESM, Downscaler, and CSI.
    legacy_names = {
        "ESM": "ESM",
        "Downscaler": "Simulation",
        "CSI_hist": "CESI_hist",
        "CSI_fut": "CESI_fut",
        "Delta_CSI": "Delta_CESI",
        "REL_Delta_CSI": "REL_Delta_CESI",
    }

    dataframe = dataframe.rename(
        columns={
            old_name: new_name
            for old_name, new_name in legacy_names.items()
            if old_name in dataframe.columns
        }
    )

    required_columns = {
        "ESM",
        "Simulation",
        "Region",
        "CESI_hist",
        "CESI_fut",
        "Delta_CESI",
        "CLSI_hist",
        "CLSI_fut",
        "Delta_CLSI",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    numeric_columns = [
        "CESI_hist",
        "CESI_fut",
        "Delta_CESI",
        "CLSI_hist",
        "CLSI_fut",
        "Delta_CLSI",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "ESM",
            "Simulation",
            "Region",
            "Delta_CESI",
            "Delta_CLSI",
        ]
    ).copy()

    if VERBOSE:
        print(
            f"[INFO] Loaded {len(dataframe)} rows from "
            f"{csv_path}"
        )
        print(
            "[INFO] ESMs:",
            sorted(dataframe["ESM"].unique()),
        )
        print(
            "[INFO] Simulations:",
            sorted(dataframe["Simulation"].unique()),
        )

    return dataframe


# ============================================================
# Regional aggregation
# ============================================================

def aggregate_by_region(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average Delta_CESI and Delta_CLSI across ESMs and simulations.

    The amplification factor is calculated from the magnitudes of the
    regionally averaged projected changes:

        amplification = abs(Delta_CLSI) / abs(Delta_CESI)

    This is a dimensionless ratio and does not represent a direct physical
    conversion between heat stress and electricity demand.
    """

    required_columns = {
        "Region",
        "Delta_CESI",
        "Delta_CLSI",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Input dataframe is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    aggregated = (
        dataframe.groupby(
            "Region",
            as_index=False,
            observed=True,
        )
        .agg(
            Delta_CESI=("Delta_CESI", "mean"),
            Delta_CLSI=("Delta_CLSI", "mean"),
        )
    )

    aggregated["Delta_CESI_abs"] = (
        aggregated["Delta_CESI"].abs()
    )

    aggregated["Delta_CLSI_abs"] = (
        aggregated["Delta_CLSI"].abs()
    )

    aggregated["Amplification"] = np.where(
        aggregated["Delta_CESI_abs"] > 0,
        aggregated["Delta_CLSI_abs"]
        / aggregated["Delta_CESI_abs"],
        np.nan,
    )

    available_regions = [
        region
        for region in REGION_ORDER
        if region in aggregated["Region"].unique()
    ]

    aggregated["Region"] = pd.Categorical(
        aggregated["Region"],
        categories=available_regions,
        ordered=True,
    )

    aggregated = aggregated.sort_values(
        "Region"
    ).reset_index(drop=True)

    return aggregated


# ============================================================
# Scatter plot
# ============================================================

def plot_scatter_delta_cesi_vs_clsi(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Plot Delta_CESI against Delta_CLSI.

    Each point represents one ESM-simulation-region combination.
    Points are colored by region and styled by simulation.
    """

    dataframe = dataframe.copy()

    available_regions = [
        region
        for region in REGION_ORDER
        if region in dataframe["Region"].unique()
    ]

    dataframe["Region"] = pd.Categorical(
        dataframe["Region"],
        categories=available_regions,
        ordered=True,
    )

    fig, axis = plt.subplots(
        figsize=(11, 8),
    )

    sns.scatterplot(
        data=dataframe,
        x="Delta_CESI",
        y="Delta_CLSI",
        hue="Region",
        style="Simulation",
        palette="tab20",
        hue_order=available_regions,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        s=90,
        ax=axis,
    )

    axis.axhline(
        0,
        linestyle="--",
        color="gray",
        linewidth=1,
    )

    axis.axvline(
        0,
        linestyle="--",
        color="gray",
        linewidth=1,
    )

    axis.set_xlabel(
        r"$\Delta$CESI (future $-$ historical)",
        fontsize=12,
    )

    axis.set_ylabel(
        r"$\Delta$CLSI (future $-$ historical)",
        fontsize=12,
    )

    axis.set_title(
        "Projected Changes in CESI and CLSI",
        fontsize=14,
        fontweight="bold",
    )

    axis.tick_params(
        axis="both",
        labelsize=11,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()

    pdf_path = (
        output_directory
        / "figure8_scatter_CESI_vs_CLSI.pdf"
    )

    png_path = (
        output_directory
        / "figure8_scatter_CESI_vs_CLSI.png"
    )

    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")


# ============================================================
# Regional CESI, CLSI, and amplification figure
# ============================================================

def plot_log_bar_with_amplification(
    regional_summary: pd.DataFrame,
    output_directory: Path,
    sort_by_clsi: bool = True,
) -> None:
    """
    Plot regional Delta_CESI and Delta_CLSI magnitudes on a logarithmic
    scale, with amplification shown on a secondary y-axis.

    Parameters
    ----------
    regional_summary
        Dataframe containing Region, Delta_CESI, Delta_CLSI,
        Delta_CESI_abs, Delta_CLSI_abs, and Amplification.

    output_directory
        Directory where figure files are written.

    sort_by_clsi
        If True, regions are sorted by increasing absolute Delta_CLSI.
        Otherwise, REGION_ORDER is retained.
    """

    dataframe = regional_summary.copy()

    required_columns = {
        "Region",
        "Delta_CESI_abs",
        "Delta_CLSI_abs",
        "Amplification",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Regional summary is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Avoid zero values on the logarithmic axis.
    positive_values = dataframe[
        ["Delta_CESI_abs", "Delta_CLSI_abs"]
    ].to_numpy()

    positive_values = positive_values[
        np.isfinite(positive_values)
        & (positive_values > 0)
    ]

    if positive_values.size == 0:
        raise ValueError(
            "No positive CESI or CLSI changes are available for plotting."
        )

    epsilon = max(
        positive_values.min() * 0.1,
        1.0e-6,
    )

    dataframe["Delta_CESI_plot"] = dataframe[
        "Delta_CESI_abs"
    ].clip(lower=epsilon)

    dataframe["Delta_CLSI_plot"] = dataframe[
        "Delta_CLSI_abs"
    ].clip(lower=epsilon)

    if sort_by_clsi:
        dataframe = dataframe.sort_values(
            "Delta_CLSI_plot"
        ).reset_index(drop=True)

    regions = dataframe["Region"].astype(str).tolist()
    x_positions = np.arange(len(regions))
    width = 0.36

    fig, left_axis = plt.subplots(
        figsize=(15, 6),
    )

    left_axis.bar(
        x_positions - width / 2,
        dataframe["Delta_CESI_plot"],
        width,
        label=r"$|\Delta\mathrm{CESI}|$",
    )

    left_axis.bar(
        x_positions + width / 2,
        dataframe["Delta_CLSI_plot"],
        width,
        label=r"$|\Delta\mathrm{CLSI}|$",
    )

    left_axis.set_yscale("log")

    left_axis.set_ylabel(
        "Magnitude of projected change (log scale)",
        fontsize=12,
    )

    left_axis.set_xticks(x_positions)

    left_axis.set_xticklabels(
        regions,
        rotation=45,
        ha="right",
        fontsize=11,
    )

    left_axis.tick_params(
        axis="y",
        labelsize=11,
    )

    minimum_value = dataframe[
        ["Delta_CESI_plot", "Delta_CLSI_plot"]
    ].min().min()

    maximum_value = dataframe[
        ["Delta_CESI_plot", "Delta_CLSI_plot"]
    ].max().max()

    left_axis.set_ylim(
        minimum_value * 0.7,
        maximum_value * 1.8,
    )

    left_axis.grid(
        axis="y",
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )

    left_axis.spines["top"].set_visible(False)

    right_axis = left_axis.twinx()

    right_axis.plot(
        x_positions,
        dataframe["Amplification"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label=r"Amplification factor ($|\Delta\mathrm{CLSI}|/|\Delta\mathrm{CESI}|$)",
    )

    right_axis.set_ylabel(
        "Amplification factor",
        fontsize=12,
    )

    right_axis.tick_params(
        axis="y",
        labelsize=11,
    )

    right_axis.axhline(
        1.0,
        color="gray",
        linestyle=":",
        linewidth=1,
    )

    right_axis.spines["top"].set_visible(False)

    handles_left, labels_left = (
        left_axis.get_legend_handles_labels()
    )

    handles_right, labels_right = (
        right_axis.get_legend_handles_labels()
    )

    left_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        frameon=False,
        fontsize=10,
    )

    left_axis.set_title(
        "Projected Regional Changes in CESI and CLSI",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()

    pdf_path = (
        output_directory
        / "figure8_compound_stress.pdf"
    )

    png_path = (
        output_directory
        / "figure8_compound_stress.png"
    )

    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {pdf_path}")
    print(f"[SAVED] {png_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Load compound-stress data and generate Figure 8 outputs."""

    dataframe = load_compound_stress_data(
        COMPOUND_STRESS_CSV
    )

    regional_summary = aggregate_by_region(
        dataframe
    )

    summary_path = (
        PLOTS_DIR
        / "figure8_regional_summary.csv"
    )

    regional_summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"[SAVED] {summary_path}")

    plot_log_bar_with_amplification(
        regional_summary=regional_summary,
        output_directory=PLOTS_DIR,
        sort_by_clsi=True,
    )

    plot_scatter_delta_cesi_vs_clsi(
        dataframe=dataframe,
        output_directory=PLOTS_DIR,
    )

    print(
        f"[DONE] Figure 8 outputs saved in "
        f"{PLOTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
