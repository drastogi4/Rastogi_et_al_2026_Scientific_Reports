"""
plot_uncertainty_diagnostics.py

Visualize:
  - Method Variability Index (MVI) from mvi_all / mvi_by_driver
  - Variance decomposition from variance_decomposition

Inputs (in DATA_DIR/processed-cw):
  - mvi_all.parquet
  - mvi_by_driver.parquet
  - variance_decomposition.parquet

Outputs:
  - figures/ (PNG files)

Customize:
  - DATA_DIR
  - METRICS_TO_PLOT list
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# ================= CONFIG =================
DATA_DIR = Path("./mvi-analysis")   # <-- change this
PROCESSED_DIR = DATA_DIR / "processed-cw"
FIG_DIR = DATA_DIR / "figures-cw"

# Metrics you care most about (must match metric_clean in your files)
METRICS_TO_PLOT = ["tmin_mean", "parea_mean", "duration_mean"]
# ==========================================
MVI_BOUNDS = np.array([
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
])

MVI_BIN_LABELS = [
    "0–0.25",
    "0.25–0.50",
    "0.50–0.75",
    "0.75–1.00",
    "1.00–1.50",
    "1.50–2.00",
    "2.00–2.50",
    "2.50–3.00",
]

def load_data():
    mvi_all = pd.read_parquet(PROCESSED_DIR / "mvi_all_downscaled.parquet")
    mvi_by_driver = pd.read_parquet(PROCESSED_DIR / "mvi_by_driver_downscaled.parquet")
    var_decomp = pd.read_parquet(PROCESSED_DIR / "variance_decomposition.parquet")

    print("[INFO] Loaded:")
    print("  mvi_all:", mvi_all.shape)
    print("  mvi_by_driver:", mvi_by_driver.shape)
    print("  variance_decomposition:", var_decomp.shape)

    return mvi_all, mvi_by_driver, var_decomp


def plot_mvi_bar(mvi_all: pd.DataFrame):
    """
    Bar plots of MVI_RMAD by region for each metric in METRICS_TO_PLOT.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        sub = mvi_all[mvi_all["metric_clean"] == metric].copy()
        if sub.empty:
            print(f"[WARN] No MVI data for metric: {metric}")
            continue

        # sort regions by MVI_RMAD
        sub = sub.sort_values("MVI_RMAD", ascending=False)

        plt.figure(figsize=(8, 4))
        plt.bar(sub["region"], sub["MVI_RMAD"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Method Variability Index (RMAD)")
        plt.title(f"MVI_RMAD across methods for Δ{metric}")
        plt.tight_layout()

        out = FIG_DIR / f"mvi_bar_{metric}.pdf"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out}")


def plot_mvi_heatmap(mvi_all: pd.DataFrame):
    """
    Plot the cold-wave Method Variability Index using discrete color
    intervals shared with the heat-wave MVI figure.

    Rows correspond to regions and columns correspond to:
      - duration_mean
      - parea_mean
      - tmin_mean

    Exact MVI values are annotated in each cell.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "region",
        "metric_clean",
        "MVI_RMAD",
    }

    missing_columns = required_columns.difference(mvi_all.columns)

    if missing_columns:
        raise ValueError(
            "mvi_all is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    metric_order = [
        "duration_mean",
        "parea_mean",
        "tmin_mean",
    ]

    metric_labels = {
        "duration_mean": "Duration",
        "parea_mean": "% area",
        "tmin_mean": "Tmin",
    }

    region_order = [
        "CAL",
        "CAR",
        "CENT",
        "CONUS",
        "FL",
        "MIDA",
        "MIDW",
        "NE",
        "NW",
        "NY",
        "SE",
        "SW",
        "TEN",
        "TEX",
    ]

    sub = mvi_all.loc[
        mvi_all["metric_clean"].isin(metric_order)
    ].copy()

    if sub.empty:
        raise ValueError(
            "No cold-wave MVI data were found for: "
            + ", ".join(metric_order)
        )

    sub["MVI_RMAD"] = pd.to_numeric(
        sub["MVI_RMAD"],
        errors="coerce",
    )

    sub = sub.dropna(
        subset=[
            "region",
            "metric_clean",
            "MVI_RMAD",
        ]
    )

    pivot = sub.pivot_table(
        index="region",
        columns="metric_clean",
        values="MVI_RMAD",
        aggfunc="mean",
    )

    available_metrics = [
        metric
        for metric in metric_order
        if metric in pivot.columns
    ]

    if not available_metrics:
        raise ValueError(
            "None of the requested cold-wave metrics are present "
            "in the MVI pivot table."
        )

    pivot = pivot.reindex(
        index=[
            region
            for region in region_order
            if region in pivot.index
        ],
        columns=available_metrics,
    )

    pivot = pivot.rename(
        columns=metric_labels
    )

    number_of_bins = len(MVI_BOUNDS) - 1

    cmap = plt.get_cmap(
        "viridis",
        number_of_bins,
    )

    norm = BoundaryNorm(
        boundaries=MVI_BOUNDS,
        ncolors=cmap.N,
        clip=True,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 9),
    )

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        annot_kws={
            "fontsize": 11,
        },
        square=True,
    )

    ax.set_title(
        "Method Variability Index",
        fontsize=15,
        pad=12,
    )

    ax.set_xlabel(
        "Mean Change Metric",
        fontsize=12,
    )

    ax.set_ylabel(
        "Region",
        fontsize=12,
    )

    ax.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=10,
    )

    ax.tick_params(
        axis="y",
        labelrotation=0,
        labelsize=10,
    )

    scalar_mappable = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_mappable.set_array([])

    bin_centers = (
        MVI_BOUNDS[:-1]
        + MVI_BOUNDS[1:]
    ) / 2.0

    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        boundaries=MVI_BOUNDS,
        ticks=bin_centers,
        spacing="proportional",
        pad=0.05,
        fraction=0.06,
    )

    colorbar.ax.set_yticklabels(
        MVI_BIN_LABELS,
        fontsize=9,
    )

    colorbar.set_label(
        "MVI",
        rotation=90,
        labelpad=12,
        fontsize=12,
    )

    # Print range for verification against the shared heat-wave scale.
    print(
        "[INFO] Cold-wave MVI range: "
        f"{pivot.min().min():.3f} to "
        f"{pivot.max().max():.3f}"
    )

    print(
        "[INFO] Shared MVI color boundaries:",
        MVI_BOUNDS,
    )

    plt.tight_layout()

    out = FIG_DIR / "mvi_heatmap_updated.pdf"

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    png_out = FIG_DIR / "mvi_heatmap_updated.png"

    plt.savefig(
        png_out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[INFO] Saved {out}")
    print(f"[INFO] Saved {png_out}")

def plot_mvi_by_driver(mvi_by_driver: pd.DataFrame):
    """
    Example: boxplots of MVI_RMAD by ESM for a given metric.
    Shows how sensitive MVI is to driving ESM.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        sub = mvi_by_driver[mvi_by_driver["metric_clean"] == metric].copy()
        if sub.empty:
            print(f"[WARN] No MVI-by-driver data for metric: {metric}")
            continue

        plt.figure(figsize=(10, 4))
        sns.boxplot(
            data=sub,
            x="esm",
            y="MVI_RMAD",
            hue="reference"
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("MVI_RMAD")
        plt.title(f"MVI_RMAD by ESM & reference for Δ{metric}")
        plt.legend(title="Reference", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out = FIG_DIR / f"mvi_by_driver_{metric}.pdf"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out}")


def plot_var_decomp_stacked(var_decomp: pd.DataFrame):
    """
    Stacked bar chart of variance fractions for each region,
    for each metric in METRICS_TO_PLOT.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        sub = var_decomp[var_decomp["metric_clean"] == metric].copy()
        if sub.empty:
            print(f"[WARN] No variance decomposition data for metric: {metric}")
            continue

        # sort regions by method contribution
        sub = sub.sort_values("SS_method_frac", ascending=False)

        regions = sub["region"].tolist()
        x = np.arange(len(regions))

        width = 0.8

        plt.figure(figsize=(10, 5))

        bottom = np.zeros(len(sub))

        for label, col in [
            ("ESM", "SS_esm_frac"),
            ("Method", "SS_method_frac"),
            ("Reference", "SS_reference_frac"),
            ("Residual", "SS_resid_frac"),
        ]:
            if col not in sub.columns:
                continue
            vals = sub[col].fillna(0).values
            plt.bar(regions, vals, bottom=bottom, label=label, width=width)
            bottom += vals

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Fraction of total variance in Δ")
        plt.title(f"Variance decomposition for Δ{metric}")
        plt.legend(title="Source of variance", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out = FIG_DIR / f"variance_decomposition_{metric}.pdf"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out}")


def main():
    mvi_all, mvi_by_driver, var_decomp = load_data()

    # Basic sanity: print available metrics
    print("[INFO] Available MVI metrics:", sorted(mvi_all["metric_clean"].unique()))
    print("[INFO] Available VarDecomp metrics:", sorted(var_decomp["metric_clean"].unique()))

    # Determine metrics present in both datasets
    existing_metrics_mvi = set(mvi_all["metric_clean"].unique())
    existing_metrics_var = set(var_decomp["metric_clean"].unique())
    metrics_to_use = list((set(METRICS_TO_PLOT) & existing_metrics_mvi & existing_metrics_var))

    if not metrics_to_use:
        print("[WARN] None of METRICS_TO_PLOT found in both datasets; using all metrics from MVI.")
        metrics_to_use = sorted(existing_metrics_mvi)

    print("[INFO] Using metrics:", metrics_to_use)

    # ---- Plot MVI ----
    plot_mvi_bar(mvi_all[mvi_all["metric_clean"].isin(metrics_to_use)])
    plot_mvi_heatmap(mvi_all[mvi_all["metric_clean"].isin(metrics_to_use)])
    plot_mvi_by_driver(mvi_by_driver[mvi_by_driver["metric_clean"].isin(metrics_to_use)])

    # ---- Plot variance decomposition ----
    plot_var_decomp_stacked(var_decomp[var_decomp["metric_clean"].isin(metrics_to_use)])

    print(f"[DONE] Figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()

