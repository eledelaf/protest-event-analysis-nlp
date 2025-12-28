#!/usr/bin/env python3
"""
Sentiment composition over time (VADER label):
Stacked area chart (monthly) showing percentage of articles that are positive / neutral / negative.
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# ----------------------------
# Config 
# ----------------------------
MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"
PROTEST_ONLY = True

MONTH_FREQ = "MS"  # Monthly aggregation frequency

# Optional COVID shading
SHADE_COVID = True
COVID_PERIODS = [{"start": "2020-03-11", "end": "2022-02-24", "label": "COVID-19 period"}]

# Output
OUT_DIR = Path("7.2figures")
OUT_FILE = OUT_DIR / "sentiment_composition_over_time.png"

def load_labels_from_mongo():
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is not set. Export it in your shell, e.g.\n"
            "  export MONGO_URI='mongodb+srv://USER:PASSWORD@HOST/?retryWrites=true&w=majority'"
        )

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    query = {"publish_date": {"$exists": True},
            "sentiment.label": {"$exists": True},}
    
    if PROTEST_ONLY:
        query["hf_label_name"] = "PROTEST"

    projection = {
        "_id": 0,
        "publish_date": 1,
        "sentiment.label": 1,
    }

    docs = list(col.find(query, projection))
    df = pd.DataFrame(docs)

    if df.empty:
        raise RuntimeError(
            "Query returned 0 documents.\n"
            "Check DB_NAME / COLLECTION_NAME, and whether your docs have:\n"
            "- publish_date\n"
            "- sentiment.label\n"
            + ("- hf_label_name == 'PROTEST'\n" if PROTEST_ONLY else "")
        )

    # Here sentiment becomes {'label': ...}
    df = df.rename(columns={"sentiment": "sentiment_obj"})
    df["label"] = df["sentiment_obj"].apply(lambda d: d.get("label") if isinstance(d, dict) else None)

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Keep only expected labels
    keep = {"positive", "neutral", "negative"}
    df = df[df["label"].isin(keep)].dropna(subset=["publish_date"]).copy()
    df = df.sort_values("publish_date")

    if df.empty:
        raise RuntimeError(
            "After filtering to labels {positive, neutral, negative}, no rows remain.\n"
            "Inspect your stored sentiment.label values."
        )

    return df


def plot_composition(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df["month"] = df["publish_date"].dt.to_period("M").dt.to_timestamp()

    # Counts by month per label
    counts = (
        df.groupby(["month", "label"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )

    # Ensure all columns exist
    for col in ["negative", "neutral", "positive"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["negative", "neutral", "positive"]]

    # Convert to percentages
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    x = pct.index.to_pydatetime()
    y_neg = pct["negative"].values
    y_neu = pct["neutral"].values
    y_pos = pct["positive"].values

    ax.stackplot(x, y_neg, y_neu, y_pos, labels=["Negative", "Neutral", "Positive"], alpha=0.85)

    if SHADE_COVID:
        for p in COVID_PERIODS:
            start = pd.to_datetime(p["start"])
            end = pd.to_datetime(p["end"])
            ax.axvspan(start, end, alpha=0.12)
            ax.text(
                start + (end - start) * 0.02,
                98,
                p.get("label", "COVID"),
                fontsize=9,
                va="top",
            )

    title = "Sentiment composition over time (monthly % of articles)"
    if PROTEST_ONLY:
        title += " — PROTEST articles only"
    ax.set_title(title)

    ax.set_xlabel("Month")
    ax.set_ylabel("Share of articles (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300)
    print(f"Saved: {OUT_FILE.resolve()}")
    plt.show()

def main():
    df = load_labels_from_mongo()
    print("Loaded rows:", len(df))
    print("Date range:", df["publish_date"].min().date(), "to", df["publish_date"].max().date())
    print("Label counts:")
    print(df["label"].value_counts().to_string())
    plot_composition(df)

if __name__ == "__main__":
    main()
