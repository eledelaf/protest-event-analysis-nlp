"""
Pre vs during vs post COVID distribution of sentiment using VADER compound.
Creates density curves for each period, to see if the distribution shifts..
"""

import os
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# ----------------------------
# Config
# ----------------------------
# MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
MONGO_URI = " "
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

PROTEST_ONLY = True

# COVID Periods
PERIODS = [
    {"name": "Pre-COVID",  "start": None,        "end": "2020-03-10"},
    {"name": "COVID",      "start": "2020-03-11","end": "2022-02-24"},
    {"name": "Post-COVID", "start": "2022-02-25","end": None},
]

# Density estimation settings
BINS = 200  # x-grid resolution
SMOOTHING_SIGMA = 2.5  #  histogram bins

# Output
OUT_DIR = Path("7.2figures")
OUT_FILE = OUT_DIR / "sentiment_density_by_period.png"


def load_df():
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is not set. Export it in your shell, e.g.\n"
            "  export MONGO_URI='mongodb+srv://USER:PASSWORD@HOST/?retryWrites=true&w=majority'"
        )

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    query = {
        "publish_date": {"$exists": True},
        "sentiment.compound": {"$exists": True},
    }
    if PROTEST_ONLY:
        query["hf_label_name"] = "PROTEST"

    projection = {
        "_id": 0,
        "publish_date": 1,
        "sentiment.compound": 1,
    }

    docs = list(col.find(query, projection))
    df = pd.DataFrame(docs)

    if df.empty:
        raise RuntimeError("Query returned 0 documents. Check DB/collection and fields exist.")

    df = df.rename(columns={"sentiment": "sentiment_obj"})
    df["compound"] = df["sentiment_obj"].apply(lambda d: d.get("compound") if isinstance(d, dict) else None)

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["compound"] = pd.to_numeric(df["compound"], errors="coerce")
    df = df.dropna(subset=["publish_date", "compound"]).copy()

    return df


def assign_period(df):
    def which_period(d):
        for p in PERIODS:
            start = pd.to_datetime(p["start"]) if p["start"] else None
            end = pd.to_datetime(p["end"]) if p["end"] else None
            ok_start = True if start is None else (d >= start)
            ok_end = True if end is None else (d <= end)
            if ok_start and ok_end:
                return p["name"]
        return "Other"

    df2 = df.copy()
    df2["period"] = df2["publish_date"].apply(which_period)
    period_names = [p["name"] for p in PERIODS]
    df2 = df2[df2["period"].isin(period_names)].copy()

    if df2.empty:
        raise RuntimeError("After assigning periods, no rows remain. Check your PERIODS dates.")

    return df2


def gaussian_smooth(y, sigma_bins):
    """
    Simple Gaussian smoothing implemented via convolution (no SciPy dependency).
    sigma_bins is in units of histogram bins.
    """
    if sigma_bins <= 0:
        return y

    radius = int(ceil(4 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma_bins ** 2))
    kernel = kernel / kernel.sum()
    return np.convolve(y, kernel, mode="same")


def plot_density(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # x grid over [-1, 1] for VADER compound
    x_min, x_max = -1.0, 1.0
    edges = np.linspace(x_min, x_max, BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))

    period_names = [p["name"] for p in PERIODS]

    for per in period_names:
        vals = df.loc[df["period"] == per, "compound"].values
        if len(vals) < 5:
            continue

        hist, _ = np.histogram(vals, bins=edges, density=True)
        y = gaussian_smooth(hist, SMOOTHING_SIGMA)

        ax.plot(centers, y, linewidth=2, label=f"{per} (n={len(vals)})")

    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_xlim(-1, 1)

    title = "Sentiment distribution by period (VADER compound)"
    if PROTEST_ONLY:
        title += " — PROTEST articles only"
    ax.set_title(title)

    ax.set_xlabel("Sentiment (compound)")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300)
    print(f"Saved: {OUT_FILE.resolve()}")

    plt.show()

def main():
    df = load_df()
    df = assign_period(df)

    print("Loaded rows:", len(df))
    print("Date range:", df["publish_date"].min().date(), "to", df["publish_date"].max().date())
    print("Counts by period:")
    print(df["period"].value_counts().to_string())

    plot_density(df)

if __name__ == "__main__":
    main()
