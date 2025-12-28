"""
Compute weekly counts of PROTEST-labelled articles from a MongoDB collection
and report the weeks with the highest number of PROTEST articles.
"""

import argparse
from typing import Dict, Any, List

import pandas as pd
from pymongo import MongoClient

# =========================================================
# CONFIG 
# =========================================================
MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

def build_query(label_source):
    """
    Build a MongoDB query for selecting PROTEST articles.   
    """
    base = {"publish_date": {"$exists": True},
            "paper": {"$exists": True},}

    if label_source == "hf_reason":
        base["hf_reason"] = {"$regex": r"->\s*PROTEST\b"}
    elif label_source == "hf_label_name":
        base["hf_label_name"] = "PROTEST"
    else:
        raise ValueError("label_source must be one of: hf_reason, hf_label_name")

    return base


def fetch_docs(col, query):
    projection = {"_id": 1, "publish_date": 1, "paper": 1}
    rows: List[Dict[str, Any]] = list(col.find(query, projection))
    if not rows:
        return pd.DataFrame(columns=["_id", "publish_date", "paper"])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Find weekly peaks in PROTEST article counts.")
    parser.add_argument("--db", default=DB_NAME, help=f"Database name (default: {DB_NAME}).")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"Collection name (default: {COLLECTION_NAME}).")
    parser.add_argument("--label-source", choices=["hf_reason", "hf_label_name"], default="hf_reason",
                        help="Where to read the final PROTEST label from (default: hf_reason).")
    parser.add_argument("--rule", default="W-MON",
                        help="pandas resample rule for weeks (default: W-MON). Use W-SUN for weeks ending Sunday.")
    parser.add_argument("--top-k", type=int, default=10, help="How many peak weeks to print (default: 10).")
    parser.add_argument("--out-csv", default="7.3outputs/weekly_protest_counts.csv",
                        help="Output CSV path for weekly counts (default: 7.3outputs/weekly_protest_counts.csv).")
    parser.add_argument("--out-pivot-csv", default="7.3outputs/weekly_protest_counts_by_paper.csv",
                        help="Output CSV path for weekly counts by paper (default: 7.3outputsweekly_protest_counts_by_paper.csv).")
    parser.add_argument("--no-pivot", action="store_true",
                        help="If set, do not create the by-paper pivot CSV.")
    args = parser.parse_args()

    if not MONGO_URI or "PASTE_YOUR_MONGO_URI_HERE" in MONGO_URI:
        raise SystemExit(
            "Edit weekly_protest_peaks_with_uri.py and paste your MongoDB URI into MONGO_URI first."
        )

    client = MongoClient(MONGO_URI)
    col = client[args.db][args.collection]

    query = build_query(args.label_source)
    df = fetch_docs(col, query)

    if df.empty:
        print("No documents matched the query. Check label-source, dates, and collection.")
        return

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["publish_date"])
    df = df.set_index("publish_date").sort_index()

    weekly = df.resample(args.rule).size().rename("n_protest_articles").to_frame()
    weekly.index.name = "week_end"
    weekly = weekly.reset_index()

    weekly.to_csv(args.out_csv, index=False)

    top = weekly.sort_values("n_protest_articles", ascending=False).head(args.top_k)
    print("\nTop weeks by PROTEST article count")
    print(top.to_string(index=False))

    if not args.no_pivot:
        pivot = (
            df.reset_index()
              .assign(week_end=lambda x: x["publish_date"].dt.to_period(args.rule).dt.end_time)
              .groupby(["week_end", "paper"])
              .size()
              .rename("n_protest_articles")
              .reset_index()
              .pivot(index="week_end", columns="paper", values="n_protest_articles")
              .fillna(0)
              .astype(int)
              .sort_index()
        )
        pivot.to_csv(args.out_pivot_csv)
        print(f"\nSaved weekly counts by paper to: {args.out_pivot_csv}")

    print(f"\nSaved weekly counts to: {args.out_csv}")

if __name__ == "__main__":
    main()
