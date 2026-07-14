"""
plot_uncertainty_diagnostics.py

Visualize:
  - Method Variability Index (MVI) from mvi_all / mvi_by_driver
  - Variance decomposition from variance_decomposition

Inputs (in DATA_DIR/processed):
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

# Shared MVI bins for heat-wave and cold-wave figures
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

# ================= CONFIG =================
DATA_DIR = Path("./mvi-analysis")   # <-- change this
PROCESSED_DIR = DATA_DIR / "processed-hw"
FIG_DIR = DATA_DIR / "figures"

# Metrics you care most about (must match metric_clean in your files)
METRICS_TO_PLOT = ["tmax_mean", "parea_mean", "duration_mean"]
# ==========================================


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
    Heatmap of MVI_RMAD using discrete bins shared with the
    cold-wave MVI heatmap.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metric_order = [
        "duration_mean",
        "parea_mean",
        "tmax_mean",
    ]

    metric_labels = {
        "duration_mean": "Duration",
        "parea_mean": "% area",
        "tmax_mean": "Tmax",
    }

    pivot = mvi_all.pivot_table(
        index="region",
        columns="metric_clean",
        values="MVI_RMAD",
        aggfunc="mean",
    )

    available_metrics = [
        metric for metric in metric_order
        if metric in pivot.columns
    ]

    pivot = pivot[available_metrics]
    pivot = pivot.rename(columns=metric_labels)

    cmap = plt.get_cmap(
        "viridis",
        len(MVI_BOUNDS) - 1,
    )

    norm = BoundaryNorm(
        MVI_BOUNDS,
        ncolors=cmap.N,
        clip=True,
    )

    fig, ax = plt.subplots(
        figsize=(7, 8),
    )

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 11},
    )

    ax.set_aspect("equal")
    ax.set_title(
        "Method Variability Index",
        fontsize=15,
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
        rotation=90,
        labelsize=10,
    )

    ax.tick_params(
        axis="y",
        rotation=0,
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

    cbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        boundaries=MVI_BOUNDS,
        ticks=bin_centers,
        spacing="proportional",
        pad=0.05,
    )

    cbar.ax.set_yticklabels(
        MVI_BIN_LABELS
    )

    cbar.set_label(
        "MVI",
        rotation=90,
        labelpad=12,
        fontsize=12,
    )

    plt.tight_layout()

    out = FIG_DIR / "mvi_heatmap_updated.pdf"

    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[INFO] Saved {out}")

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

