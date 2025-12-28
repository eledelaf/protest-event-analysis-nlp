"""
Counts PROTEST articles by COVID period (pre/during/post), overall and by outlet.
"""

from pymongo import MongoClient
import pandas as pd

# ----------------------------
# Config
# ----------------------------
MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

# If True, uses your threshold decision stored in hf_reason 
# If False, uses hf_label_name == "PROTEST"
USE_THRESHOLD_LABEL = True

# ----------------------------
# Periods
# ----------------------------
PERIODS = [
    ("Pre-COVID",  "2020-01-01", "2020-03-11"),
    ("COVID",      "2020-03-11", "2022-02-24"),
    ("Post-COVID", "2022-02-24", "2025-01-01"), 
]

def base_filter():
    f = {
        "status": "done",
        "publish_date": {"$exists": True, "$ne": None},
    }
    if USE_THRESHOLD_LABEL:
        # Matches strings like "... -> PROTEST" in your hf_reason
        f["hf_reason"] = {"$regex": r"->\s*PROTEST\b"}
    else:
        f["hf_label_name"] = "PROTEST"
    return f

def counts_by_paper_for_period(col, start, end):
    f = base_filter()
    f["publish_date"] = {"$gte": start, "$lt": end}

    pipeline = [
        {"$match": f},
        {"$group": {"_id": "$paper", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ]
    rows = list(col.aggregate(pipeline))
    return {r["_id"] if r["_id"] is not None else "UNKNOWN": r["n"] for r in rows}

def overall_count_for_period(col, start, end):
    f = base_filter()
    f["publish_date"] = {"$gte": start, "$lt": end}
    return col.count_documents(f)

def main():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    # Collect counts per paper per period
    all_papers = set()
    per_period_paper_counts = {}

    overall = {}
    for label, start, end in PERIODS:
        overall[label] = overall_count_for_period(col, start, end)

        paper_counts = counts_by_paper_for_period(col, start, end)
        per_period_paper_counts[label] = paper_counts
        all_papers.update(paper_counts.keys())

    # Build dataframe: rows = papers, cols = periods
    papers_sorted = sorted(all_papers)
    df = pd.DataFrame(index=papers_sorted)

    for label, _, _ in PERIODS:
        df[label] = [per_period_paper_counts[label].get(p, 0) for p in papers_sorted]

    df["TOTAL"] = df.sum(axis=1)

    # Add overall row
    overall_row = {label: overall[label] for label, _, _ in PERIODS}
    overall_row["TOTAL"] = sum(overall.values())
    df.loc["ALL_OUTLETS"] = pd.Series(overall_row)

    # ALL_OUTLETS on top
    df = df.loc[["ALL_OUTLETS"] + [p for p in df.index if p != "ALL_OUTLETS"]]

    print("\nPROTEST article counts by COVID period (overall + by outlet)\n")
    print(df.to_string())

    tell = "hf_reason (threshold-based)" if USE_THRESHOLD_LABEL else "hf_label_name"
    print(f"\nLabel source used: {tell}")

    out_csv = "7.3outputs/protest_counts_by_covid_period_and_outlet.csv"
    df.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
