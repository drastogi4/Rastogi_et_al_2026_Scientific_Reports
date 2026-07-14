#!/usr/bin/env python3
"""
Generate June 2022 heatwave figures for EIA regions.

Outputs
-------
June_2022_HW_percentarea.pdf
scatter2_June2022.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

THRESHOLD = "t95"
MINIMUM_SIZE = "12500"
GRID = "VIC4"
YEARS = range(2015, 2023)


# ---------------------------------------------------------------------
# Reading functions
# ---------------------------------------------------------------------

def read_ext(year):

    fname = (
        f"../data/DaymetV4/"
        f"DaymetV4_{GRID}_tmax_{year}_{THRESHOLD}"
        f"_numdays_track_3days_{MINIMUM_SIZE}.txt"
    )

    return pd.read_csv(
        fname,
        sep="\t",
        header=None,
        skiprows=[0],
        names=[
            "doy", "minlat", "maxlat", "minlon", "maxlon",
            "meanlon", "meanlat", "clon", "clat", "area"
        ],
    )


def read_area(year):

    fname = (
        f"../data/DaymetV4/"
        f"DaymetV4_{GRID}_tmax_{year}_{THRESHOLD}"
        f"_numdays_track_3days_{MINIMUM_SIZE}_CONUS_percentagearea.txt"
    )

    df = pd.read_csv(
        fname,
        sep=" ",
        header=None,
        skiprows=[0],
        names=[
            "doy","CONUS","CAR","CENT","FL","MIDA","MIDW",
            "NE","NY","SE","TEN","SW","CAL","TEX","NW"
        ],
    )

    df["doy"] += 120
    df["year"] = year
    return df


def read_intensity(year):

    fname = (
        f"../data/DaymetV4/"
        f"DaymetV4_{GRID}_tmax_{year}_{THRESHOLD}"
        f"_numdays_track_3days_{MINIMUM_SIZE}_CONUS_avgtmax.txt"
    )

    df = pd.read_csv(
        fname,
        sep=" ",
        header=None,
        skiprows=[0],
        names=[
            "doy","CONUS","CAR","CENT","FL","MIDA","MIDW",
            "NE","NY","SE","TEN","SW","CAL","TEX","NW"
        ],
    )

    df["doy"] += 120
    df["year"] = year
    return df


def process_ext(df, year):

    ids = []
    counter = 0

    for value in df["clat"]:
        if pd.isna(value):
            counter += 1
        ids.append(counter)

    df = df.copy()
    df["ID"] = ids
    df["year"] = year

    df = (
        df[["ID", "doy", "clat", "clon", "area", "year"]]
        .dropna()
        .reset_index(drop=True)
    )

    df["ID"] = df["ID"].astype(int)
    df["doy"] = df["doy"].astype(int) + 120
    df["duration"] = df.groupby("ID")["ID"].transform("count")

    return df


def eia_regions():

    regions = [
        "NE","NY","MIDA","MIDW","SE","CENT",
        "FLA","CAR","TEN","SW","CAL","TEX","NW"
    ]

    frames = []

    for r in regions:

        df = pd.read_excel(f"../eia/Region_{r}.xlsx")
        df = df.groupby("Local date").mean().reset_index()
        df.rename(columns={"D": f"D_{r}"}, inplace=True)
        frames.append(df[["Local date", f"D_{r}"]])

    out = frames[0]

    for df in frames[1:]:
        out = out.merge(df, on="Local date", how="outer")

    out["Date"] = pd.to_datetime(out["Local date"])
    out["year"] = out["Date"].dt.year
    out["doy"] = out["Date"].dt.dayofyear - 1

    return out[(out.doy > 119) & (out.doy < 273)]


def combine(df1, df2):

    df1 = df1.copy()
    df2 = df2.copy()

    df1["year"] = df1["year"].astype(int)
    df2["year"] = df2["year"].astype(int)

    return df1.merge(df2, how="outer", on=["doy", "year"])


def regional(df, region):

    demand = f"D_{region}" if region != "FL" else "D_FLA"

    out = df[
        [
            "ID",
            "doy",
            "duration",
            "year",
            region,
            f"tmax{region}",
            demand,
            "dayofweek",
            "Month",
        ]
    ].copy()

    out["Region"] = region

    out.rename(
        columns={
            region: "percentarea",
            f"tmax{region}": "intensity",
            demand: "Demand",
        },
        inplace=True,
    )

    scaler = MinMaxScaler()
    out["Demands"] = scaler.fit_transform(out[["Demand"]])

    return out


def plot_heatmap(df):

    order = [
        "NW","CAL","SW","TEX","CENT","MIDA",
        "MIDW","TEN","SE","CAR","FL","NE"
    ]

    df = df.copy()
    df["Region"] = pd.Categorical(
        df["Region"],
        categories=order,
        ordered=True,
    )

    df = df.sort_values("Region")

    pivot = df.pivot(
        index="Region",
        columns="doy",
        values="percentareaHW",
    )

    plt.figure(figsize=(8,5))

    sns.heatmap(
        pivot,
        cmap="gist_heat_r",
        vmin=0,
        vmax=100,
        cbar_kws=dict(
            ticks=[0,10,20,40,60,80,100]
        ),
    )

    plt.tight_layout()
    plt.savefig("June_2022_HW_percentarea.pdf")


def plot_scatter(df):

    hue_order = [
        "NW","CAL","SW","TEX","CENT",
        "MIDA","MIDW","TEN","SE","CAR",
        "FL","NE","CONUS"
    ]

    sns.relplot(
        data=df,
        x="intensitydiff",
        y="Demandperdiff",
        hue="Region",
        hue_order=hue_order,
        size="percentareadiff",
        sizes=(40,400),
        alpha=1,
        palette="muted",
        height=6,
    )

    plt.savefig("scatter2_June2022.pdf")


def main():

    ext = []
    area = []
    intensity = []

    for year in YEARS:

        print(year)

        ext.append(process_ext(read_ext(year), year))
        area.append(read_area(year))
        intensity.append(read_intensity(year))

    dfext = pd.concat(ext, ignore_index=True)
    dfarea = pd.concat(area, ignore_index=True)
    dfintensity = pd.concat(intensity, ignore_index=True)

    dfintensity.rename(
        columns={c: f"tmax{c}" for c in
                 ["CONUS","CAR","CENT","FL","MIDA","MIDW",
                  "NE","NY","SE","TEN","SW","CAL","TEX","NW"]},
        inplace=True,
    )

    dfeia = eia_regions()

    df = combine(dfext, dfeia)
    df = combine(df, dfarea)
    df = combine(df, dfintensity)

    df["dayofweek"] = df["Local date"].dt.dayofweek
    df["Month"] = df["Local date"].dt.month

    regions = [
        "NE","NW","MIDA","MIDW","SE","CENT",
        "FL","CAR","TEN","SW","CAL","TEX"
    ]

    dffinal = pd.concat(
        [regional(df, r) for r in regions],
        ignore_index=True,
    ).dropna().drop_duplicates()

    dfHW = (
        dffinal[(dffinal.ID == 7) & (dffinal.year == 2022)]
        .drop_duplicates()
        .rename(
            columns={
                "percentarea": "percentareaHW",
                "intensity": "intensityHW",
                "Demand": "DemandHW",
            }
        )
    )

    dfclim = dffinal[dffinal.year < 2022]

    clim = (
        dfclim.groupby(["doy","Region"])
        .mean(numeric_only=True)
        .reset_index()
    )

    clim = clim[(clim.doy > 154) & (clim.doy < 177)]

    plot_heatmap(dfHW)

    merged = dfHW.merge(clim, on=["doy","Region"])

    merged["Demanddiff"] = merged["DemandHW"] - merged["Demand"]
    merged["Demandperdiff"] = (
        100 * merged["Demanddiff"] / merged["Demand"]
    )
    merged["intensitydiff"] = (
        merged["intensityHW"] - merged["intensity"]
    )
    merged["percentareadiff"] = (
        merged["percentareaHW"] - merged["percentarea"]
    )

    merged = merged[merged["percentareadiff"] > 0]

    plot_scatter(merged)


if __name__ == "__main__":
    main()
