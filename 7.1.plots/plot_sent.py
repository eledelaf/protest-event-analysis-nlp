"""
Publication date vs sentiment (VADER compound) — PROTEST articles only
Reads from MongoDB and recreates the plot:
- scatter of per-article compound scores
- smoothed trend line over time
- rolling quantile band
- shaded COVID period
- peak month annotations
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# ----------------------------
# Config
# ----------------------------
MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

# Field names in MONGO
DATE_FIELD = "publish_date"          
SENTIMENT_FIELD = "sentiment"     
PROTEST_FILTER = {"hf_label": 1}

# Plot period highlight
COVID_START = "2020-03-01"
COVID_END   = "2022-03-01"

# Smoothing / band parameters
ROLLING_DAYS = 30          # smoothing window on daily series
BAND_Q_LOW = 0.10          # band lower quantile
BAND_Q_HIGH = 0.90         # band upper quantile

# Annotations
TOP_K_PEAKS = 6            # number of peaks to label

# ----------------------------
# Data loading
# ----------------------------
def load_from_mongo():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    projection = {DATE_FIELD: 1, "sentiment": 1, "_id": 0}
    docs = list(col.find(PROTEST_FILTER, projection))
    if not docs:
        raise RuntimeError("Query returned 0 docs. Check PROTEST_FILTER and field names.")

    df = pd.DataFrame(docs)

    # Parse date
    df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD], errors="coerce")

    # Extract VADER compound from nested dict
    df["compound"] = df["sentiment"].apply(
        lambda s: s.get("compound") if isinstance(s, dict) else np.nan
    )
    df["compound"] = pd.to_numeric(df["compound"], errors="coerce")

    df = df.dropna(subset=[DATE_FIELD, "compound"]).copy()
    df = df.sort_values(DATE_FIELD)

    # Keep only valid VADER range
    df = df[(df["compound"] >= -1.0) & (df["compound"] <= 1.0)]

    return df

# ----------------------------
# Peak detection (monthly local maxima)
# ----------------------------
def local_maxima(series: pd.Series) -> pd.Series:
    """
    Return values at local maxima for a 1D series indexed by datetime.
    A point is a local max if it's greater than both neighbors.
    """
    s = series.dropna()
    if len(s) < 3:
        return s.iloc[0:0]

    prev_ = s.shift(1)
    next_ = s.shift(-1)
    is_peak = (s > prev_) & (s > next_)
    return s[is_peak]

# ----------------------------
# Plot
# ----------------------------
def plot(df, outpath):
    dates = df[DATE_FIELD]
    y = df["compound"]

    # Daily mean series
    daily = (
        df.set_index(DATE_FIELD)["compound"]
        .resample("D")
        .mean()
    )

    # Rolling smooth + rolling quantile band
    roll_mean = daily.rolling(ROLLING_DAYS, min_periods=max(7, ROLLING_DAYS // 4)).mean()
    roll_lo = daily.rolling(ROLLING_DAYS, min_periods=max(7, ROLLING_DAYS // 4)).quantile(BAND_Q_LOW)
    roll_hi = daily.rolling(ROLLING_DAYS, min_periods=max(7, ROLLING_DAYS // 4)).quantile(BAND_Q_HIGH)

    # Monthly series for peak annotations
    monthly = df.set_index(DATE_FIELD)["compound"].resample("M").mean()
    peaks = local_maxima(monthly).sort_values(ascending=False).head(TOP_K_PEAKS)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Scatter
    ax.scatter(dates, y, s=18, alpha=0.18)

    # Band + trend line
    ax.fill_between(roll_mean.index, roll_lo.values, roll_hi.values, alpha=0.18)
    ax.plot(roll_mean.index, roll_mean.values, linewidth=2.5)

    # Neutral line
    ax.axhline(0, linestyle="--", linewidth=1.5)

    # COVID shaded period
    covid_start = pd.to_datetime(COVID_START)
    covid_end = pd.to_datetime(COVID_END)
    ax.axvspan(covid_start, covid_end, alpha=0.10)
    ax.text(
        covid_start + (covid_end - covid_start) * 0.05,
        0.95,
        "COVID-19 period",
        fontsize=11,
        va="top")

    # Peak annotations
    for dt, val in peaks.items():
        ax.scatter([dt], [val], s=120, zorder=5) 
        ax.annotate(
            f"{dt:%Y-%m}\n{val:.2f}",
            (dt, val),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=10,
            weight="bold"
        )

    ax.set_title("Publication date vs sentiment (VADER compound) — PROTEST articles only")
    ax.set_xlabel("Publication date")
    ax.set_ylabel("Sentiment (compound)")

    ax.set_ylim(-1.2, 1.1)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    outpath = Path("7.2figures/sentiment_over_time.png")
    df = load_from_mongo()
    plot(df, outpath)
    print(f"Saved: {outpath.resolve()}")

if __name__ == "__main__":
    main()

