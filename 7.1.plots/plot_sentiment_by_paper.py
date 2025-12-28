"""
Differences between papers and sentiments (VADER compound).
Creates:
A) Faceted time series: monthly mean sentiment line per paper (shared y-axis).
B1) Distribution by paper: violin + boxplot of compound per paper.
B2) COVID angle: distributions by paper faceted by period (pre/during/post).
"""

import os
from math import ceil
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

# Filter: keep only protest-labelled articles.
PROTEST_ONLY = True

# Monthly aggregation frequency for time series
MONTH_FREQ = "MS"  # month-start; alternative: "M" for month-end

# Limit papers in faceted plots, to avoid unreadable figures
MAX_PAPERS_FACET = 12  

# Drop papers with too few documents, helps stability of distributions
MIN_DOCS_PER_PAPER = 20  

# Define periods for the COVID 
PERIODS = [
    {"name": "Pre-COVID",  "start": None,        "end": "2020-03-10"},
    {"name": "COVID",      "start": "2020-03-11","end": "2022-02-24"},
    {"name": "Post-COVID", "start": "2022-02-25","end": None},
]
def get_covid_window():
    covid = next((p for p in PERIODS if p["name"].lower() == "covid"), None)
    if not covid:
        return None, None
    start = pd.to_datetime(covid["start"]) if covid.get("start") else None
    end = pd.to_datetime(covid["end"]) if covid.get("end") else None
    return start, end

COVID_START, COVID_END = get_covid_window()

def shade_covid(ax, add_label=False):
    if COVID_START is None or COVID_END is None:
        return
    ax.axvspan(COVID_START, COVID_END, color="lightsteelblue", alpha=0.20, zorder=0)
    if add_label:
        mid = COVID_START + (COVID_END - COVID_START) / 2
        ax.text(
            mid, 0.92, "COVID-19 period",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9
        )

# Output
OUT_DIR = Path("7.2figures")
OUT_A = OUT_DIR / "sentiment_monthly_mean_facets.png"
OUT_B1 = OUT_DIR / "sentiment_distribution_by_paper.png"
OUT_B2 = OUT_DIR / "sentiment_distribution_by_paper_by_period.png"


def _require_env():
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI is not set. Export it in your shell, e.g.\n"
            "  export MONGO_URI='mongodb+srv://USER:PASSWORD@HOST/?retryWrites=true&w=majority'\n"
            "Or set it directly in the script (not recommended)."
        )


def load_df():
    _require_env()

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

    df = df.dropna(subset=["publish_date", "compound", "paper"]).sort_values("publish_date")
    return df


def filter_papers(df):
    counts = df["paper"].value_counts()
    keep = counts[counts >= MIN_DOCS_PER_PAPER].index
    df2 = df[df["paper"].isin(keep)].copy()

    if df2.empty:
        raise RuntimeError(
            f"After filtering papers with < {MIN_DOCS_PER_PAPER} docs, nothing remains.\n"
            "Lower MIN_DOCS_PER_PAPER or inspect your 'paper' field values."
        )
    return df2


def add_period(df):
    def assign(d):
        for p in PERIODS:
            start = pd.to_datetime(p["start"]) if p["start"] else None
            end = pd.to_datetime(p["end"]) if p["end"] else None
            ok_start = True if start is None else (d >= start)
            ok_end = True if end is None else (d <= end)
            if ok_start and ok_end:
                return p["name"]
        return "Other"

    df2 = df.copy()
    df2["period"] = df2["publish_date"].apply(assign)
    return df2


def plot_faceted_monthly_means(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = (
        df.groupby(["paper", pd.Grouper(key="publish_date", freq=MONTH_FREQ)])["compound"]
          .mean()
          .reset_index()
          .rename(columns={"publish_date": "month"})
    )

    paper_order = df["paper"].value_counts().index.tolist()
    if MAX_PAPERS_FACET is not None and len(paper_order) > MAX_PAPERS_FACET:
        paper_order = paper_order[:MAX_PAPERS_FACET]
        monthly = monthly[monthly["paper"].isin(paper_order)]

    n = len(paper_order)
    fig, axes = plt.subplots(
        nrows=n, ncols=1,
        figsize=(12, 2.6 * n),   
        sharex=True, sharey=True
        )
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, paper in enumerate(paper_order):
        ax = axes[i]
        shade_covid(ax, add_label=(i == 0))
        d = monthly[monthly["paper"] == paper].sort_values("month")
        ax.plot(d["month"], d["compound"], linewidth=2)
        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_title(paper, fontsize=10)
        ax.set_ylim(-1, 1)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    title = "Monthly mean sentiment by paper (VADER compound)"
    if PROTEST_ONLY:
        title += " — PROTEST articles only"
    fig.suptitle(title, y=0.995)

    fig.tight_layout()
    fig.savefig(OUT_A, dpi=300)
    print(f"Saved: {OUT_A.resolve()}")
    plt.show()


def plot_distribution_by_paper(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_order = df["paper"].value_counts().index.tolist()
    data = [df.loc[df["paper"] == p, "compound"].values for p in paper_order]

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(paper_order)), 5))

    ax.violinplot(data, showextrema=False)
    ax.boxplot(data, widths=0.15, showfliers=False)

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_ylim(-1, 1)

    ax.set_xticks(range(1, len(paper_order) + 1))
    ax.set_xticklabels(paper_order, rotation=35, ha="right")

    title = "Sentiment distribution by paper (VADER compound)"
    if PROTEST_ONLY:
        title += " — PROTEST articles only"
    ax.set_title(title)

    ax.set_xlabel("Paper")
    ax.set_ylabel("Sentiment (compound)")

    fig.tight_layout()
    fig.savefig(OUT_B1, dpi=300)
    print(f"Saved: {OUT_B1.resolve()}")
    plt.show()


def plot_distribution_by_paper_by_period(df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfp = add_period(df)
    period_names = [p["name"] for p in PERIODS]
    dfp = dfp[dfp["period"].isin(period_names)].copy()

    paper_order = dfp["paper"].value_counts().index.tolist()

    n = len(period_names)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(max(10, 0.55 * len(paper_order)), 3.2 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, per in zip(axes, period_names):
        sub = dfp[dfp["period"] == per]
        data = [sub.loc[sub["paper"] == p, "compound"].values for p in paper_order]

        ax.boxplot(data, widths=0.5, showfliers=False)
        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_ylim(-1, 1)
        ax.set_title(per, fontsize=10)
        ax.set_ylabel("compound")

    axes[-1].set_xticks(range(1, len(paper_order) + 1))
    axes[-1].set_xticklabels(paper_order, rotation=35, ha="right")
    axes[-1].set_xlabel("Paper")

    title = "Sentiment by paper across periods (VADER compound)"
    if PROTEST_ONLY:
        title += " — PROTEST articles only"
    fig.suptitle(title, y=0.995)

    fig.tight_layout()
    fig.savefig(OUT_B2, dpi=300)
    print(f"Saved: {OUT_B2.resolve()}")
    plt.show()


def main():
    df = load_df()
    df = filter_papers(df)

    print("Loaded rows:", len(df))
    print("Date range:", df["publish_date"].min().date(), "to", df["publish_date"].max().date())
    print("Papers:", df["paper"].nunique())
    print("Docs per paper (top 10):")
    print(df["paper"].value_counts().head(10).to_string())

    plot_faceted_monthly_means(df)
    plot_distribution_by_paper(df)
    plot_distribution_by_paper_by_period(df)


if __name__ == "__main__":
    main()
