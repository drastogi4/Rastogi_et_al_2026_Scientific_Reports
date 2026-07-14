# Rastogi_et_al_2026_Scientific_Reports

**Code for:** *Evaluating Methodological Uncertainty in High-Resolution Temperature Extremes and Their Regional Implications*

## Overview

This repository contains the code used in:

> **Rastogi et al. (2026).** *Evaluating Methodological Uncertainty in High-Resolution Temperature Extremes and Their Regional Implications.* Scientific Reports.

The repository reproduces the complete analysis workflow used in the manuscript, including:

- Identification of persistent heatwave and cold-wave events over the contiguous United States (CONUS)
- Event tracking using **TempestExtremes**
- Quantification of electricity demand impacts across U.S. Energy Information Administration (EIA) regions
- Historical and future projections of temperature extremes
- Uncertainty and variance decomposition analyses
- Generation of all figures presented in the manuscript

The workflow combines downscaled Earth System Model (ESM) simulations, event-tracking algorithms, observed electricity demand, and statistical analyses.

---

## Repository Structure

```
.
├── Event_Detection/
├── Event_Processing/
├── Uncertainty_Analysis/
├── Figures/
└── README.md
```

---

# Event Detection

Scripts used to identify persistent heatwave and cold-wave events from daily temperature fields using percentile thresholds and the **TempestExtremes** tracking framework.

| Script | Description |
|---------|-------------|
| `calculate_tmax_p95_climatology.sh` | Computes daily 95th percentile climatology of maximum temperature |
| `calculate_tmin_p05_climatology.sh` | Computes daily 5th percentile climatology of minimum temperature |
| `create_tmax_p95_daily_mask.ncl` | Generates daily binary heatwave exceedance masks |
| `create_tmin_p05_daily_mask.ncl` | Generates daily binary cold-wave exceedance masks |
| `run_tempestextremes_heat_events.sh` | Detects and tracks heatwave events using TempestExtremes |
| `run_tempestextremes_cold_events.sh` | Detects and tracks cold-wave events using TempestExtremes |

---

# Event Processing

Scripts for calculating event characteristics including duration, spatial extent, intensity, and regional statistics.

| Script | Description |
|---------|-------------|
| `process_heatwave_events.py` | Processes tracked heatwave events |
| `process_coldwave_events.py` | Processes tracked cold-wave events |
| `prepare_heatwave_data.py` | Combines heatwave metrics across datasets |
| `prepare_coldwave_data.py` | Combines cold-wave metrics across datasets |

---

# Uncertainty Analysis and Projected Changes

Scripts used to quantify uncertainty in future projections and evaluate impacts on regional electricity demand.

| Script | Description |
|---------|-------------|
| `calculate_mvi_heatwave.py` | Computes the Multivariate Vulnerability Index (MVI) for heatwaves |
| `calculate_mvi_coldwave.py` | Computes the Multivariate Vulnerability Index (MVI) for cold waves |
| `variance_decomposition_heatwave.py` | Quantifies predictor contributions to heatwave-related demand variability |
| `variance_decomposition_coldwave.py` | Quantifies predictor contributions to cold-wave demand variability |
| `compound_stress_analysis.py` | Evaluates concurrent temperature and electricity demand stressors |

---

# Figure Generation

Scripts used to reproduce every figure in the manuscript.

## Figure 1

| Script | Figure |
|---------|--------|
| `figure1_observed_demand_scatter.py` | Observed heatwave–electricity demand relationships |
| `figure1_observed_coldwave_scatter.py` | Observed cold-wave–electricity demand relationships |

## Figure 2

| Script | Figure |
|---------|--------|
| `Figure2a_june2022_heatwave_tmax_anomaly.ncl` | June 2022 heatwave temperature anomaly |
| `Figure2bc_june2022_heatwave_heatmap_scatter.py` | Regional heatwave extent and electricity demand response |
| `Figure2d_feb2021_coldwave_tmin_anomaly.ncl` | February 2021 cold-wave temperature anomaly |
| `Figure2de_feb2021_coldwave_heatmap_scatter.py` | Regional cold-wave extent and electricity demand response |

## Figures 3–8

| Script | Figure |
|---------|--------|
| `plot_figure3_heatwave_historical.py` | Historical heatwave analysis |
| `plot_figure4_coldwave_historical.py` | Historical cold-wave analysis |
| `plot_figure5_projected_changes.py` | Future projected changes |
| `plot_figure6_heatwave_uncertainty.py` | Heatwave uncertainty analysis |
| `plot_figure7_coldwave_uncertainty.py` | Cold-wave uncertainty analysis |
| `plot_figure8_compound_stress.py` | Compound stress analysis |

---

# Data Requirements

The datasets analyzed in this study are **not distributed** with this repository because of their size and licensing restrictions.

The workflow uses:

- Downscaled Earth System Model (ESM) temperature data
- U.S. Energy Information Administration (EIA) regional electricity demand data
- Temperature climatologies and threshold masks
- TempestExtremes event-tracking outputs

Please update all input and output paths in the scripts to match your local directory structure before running the workflow.

Additional information on data availability is provided in the accompanying manuscript.

---

# Software Requirements

The workflow was developed and tested using:

- Python 3.10+
- NumPy
- Pandas
- Xarray
- Matplotlib
- Seaborn
- Scikit-learn
- NCL (NCAR Command Language)
- TempestExtremes
- Bash

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<username>/Rastogi_et_al_2026_Scientific_Reports.git
cd Rastogi_et_al_2026_Scientific_Reports
```

Install the required Python packages:

```bash
pip install numpy pandas xarray matplotlib seaborn scikit-learn
```

Install **TempestExtremes** and **NCL** following their official documentation.

---

# Reproducibility

The scripts in this repository reproduce the analyses and figures presented in the manuscript. Intermediate datasets generated during the workflow are not included because of their size but can be recreated by running the scripts in the order described above.

---

# Citation

If you use this code, please cite:

> Rastogi, D., *et al.* (2026). *Evaluating Methodological Uncertainty in High-Resolution Temperature Extremes and Their Regional Implications.* Scientific Reports.

---

# License

This repository is released under the MIT License. See the `LICENSE` file for details.
