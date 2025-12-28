"""
Heatmap: papers × time (monthly mean VADER compound)
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
PROTEST_ONLY = True

# Monthly frequency
MONTH_FREQ = "MS"

# Keep only papers with at least this many documents overall
MIN_DOCS_PER_PAPER = 20

# Optionally limit to top-N papers by volume (keeps heatmap readable)
TOP_N_PAPERS = 20  # set None to disable

# Missing months handling
FILL_MISSING_WITH = np.nan 

# Output
OUT_DIR = Path("7.2figures")
OUT_FILE = OUT_DIR / "sentiment_heatmap_paper_time.png"


def load_df():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is not set. Export it in your shell, e.g.\n"
                "  export MONGO_URI='mongodb+srv://USER:PASSWORD@HOST/?retryWrites=true&w=majority'")

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    query = {
        "publish_date": {"$exists": True},
        "paper": {"$exists": True},
        "sentiment.compound": {"$exists": True},
    }
    if PROTEST_ONLY:
        query["hf_label_name"] = "PROTEST"

    projection = {
        "_id": 0,
        "publish_date": 1,
        "paper": 1,
        "sentiment.compound": 1,
    }

    docs = list(col.find(query, projection))
    df = pd.DataFrame(docs)

    if df.empty:
        raise RuntimeError(
            "Query returned 0 documents.\n"
            "Check DB_NAME / COLLECTION_NAME, and whether your docs have:\n"
            "- publish_date\n"
            "- paper\n"
            "- sentiment.compound\n"
            + ("- hf_label_name == 'PROTEST'\n" if PROTEST_ONLY else "")
        )

    df = df.rename(columns={"sentiment": "sentiment_obj"})
    df["compound"] = df["sentiment_obj"].apply(lambda d: d.get("compound") if isinstance(d, dict) else None)

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["compound"] = pd.to_numeric(df["compound"], errors="coerce")
    df["paper"] = df["paper"].astype(str)

    df = df.dropna(subset=["publish_date", "paper", "compound"]).copy()
    return df


def make_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    # Filter papers with enough docs
    counts = df["paper"].value_counts()
    keep = counts[counts >= MIN_DOCS_PER_PAPER].index
    df = df[df["paper"].isin(keep)].copy()

    if df.empty:
        raise RuntimeError(
            f"After filtering papers with < {MIN_DOCS_PER_PAPER} docs, nothing remains.\n"
            "Lower MIN_DOCS_PER_PAPER or inspect your 'paper' field values."
        )

    # Limit to top-N
    if TOP_N_PAPERS is not None:
        top = df["paper"].value_counts().head(TOP_N_PAPERS).index
        df = df[df["paper"].isin(top)].copy()

    # Quarterly mean per paper
    df["period"] = df["publish_date"].dt.to_period("Q").dt.start_time  # 3-month blocks

    pivot = (
        df.pivot_table(index="paper", columns="period", values="compound", aggfunc="mean")
          .sort_index()
    )

    # Order papers by volume (descending) for readability
    paper_order = df["paper"].value_counts().index.tolist()
    pivot = pivot.reindex(paper_order)

    # Fill missing months
    if not (isinstance(FILL_MISSING_WITH, float) and np.isnan(FILL_MISSING_WITH)):
        pivot = pivot.fillna(FILL_MISSING_WITH)

    return pivot

def plot_heatmap(pivot):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pivot.values

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * pivot.shape[0])))

    # Use imshow; default colormap
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    # Ticks and labels
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    ax.set_xticks(range(pivot.shape[1]))
    # Format months as YYYY-MM for compactness
    xt = [pd.to_datetime(c).strftime("%Y-%m") for c in pivot.columns]
    ax.set_xticklabels(xt, rotation=45, ha="right")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean sentiment (compound)")

    title = "Heatmap of monthly mean sentiment (paper × month)"
    if PROTEST_ONLY:
        title += " — PROTEST articles only"
    ax.set_title(title)
    ax.set_xlabel("Quarterly")
    ax.set_ylabel("Paper")

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300)
    print(f"Saved: {OUT_FILE.resolve()}")

    plt.show()


def main():
    df = load_df()
    print("Loaded rows:", len(df))
    print("Papers (raw):", df["paper"].nunique())
    print("Date range:", df["publish_date"].min().date(), "to", df["publish_date"].max().date())

    pivot = make_heatmap(df)
    print("Heatmap shape (papers × months):", pivot.shape)
    plot_heatmap(pivot)

if __name__ == "__main__":
    main()
