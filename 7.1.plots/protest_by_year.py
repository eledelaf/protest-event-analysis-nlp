"""
Determine protest article counts by year and outlet.
"""
import pandas as pd
from pymongo import MongoClient

# =========================
# CONFIG
# =========================
# MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
MONGO_URI = " "
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

# Define what "PROTEST" means:
LABEL_MODE = "reason" 

# If True, only count articles that already have sentiment
REQUIRE_SENTIMENT = False

YEAR_MIN = 2020
YEAR_MAX = 2024

OUT_PIVOT_CSV = "7.3outputs/protest_counts_by_year_outlet.csv"
OUT_LONG_CSV = "7.3outputs/protest_counts_long.csv"

def build_match(label_mode, require_sentiment):
    match = {
        "status": "done",
        "paper": {"$exists": True, "$ne": None, "$ne": ""},
        "publish_date": {"$exists": True, "$ne": None, "$ne": ""},
        "publish_date": {"$regex": r"^\d{4}-\d{2}-\d{2}$"},
    }

    if label_mode == "reason":
        match["hf_reason"] = {"$regex": r"->\s*PROTEST\b"}
    else:
        match["hf_label_name"] = "PROTEST"

    if require_sentiment:
        match["sentiment.compound"] = {"$exists": True}

    return match

def main():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    match = build_match(LABEL_MODE, REQUIRE_SENTIMENT)

    pipeline = [
        {"$match": match},
        {"$addFields": {
            "pub_date": {
                "$dateFromString": {
                    "dateString": "$publish_date",
                    "format": "%Y-%m-%d",
                    "onError": None,
                    "onNull": None
                }
            }
        }},
        {"$match": {"pub_date": {"$ne": None}}},
        {"$addFields": {"year": {"$year": "$pub_date"}}},
        {"$match": {"year": {"$gte": YEAR_MIN, "$lte": YEAR_MAX}}},
        {"$group": {
            "_id": {"paper": "$paper", "year": "$year"},
            "n": {"$sum": 1}
        }},
        {"$sort": {"_id.year": 1, "_id.paper": 1}},
    ]

    results = list(col.aggregate(pipeline, allowDiskUse=True))
    if not results:
        print("No results. Check LABEL_MODE / fields / publish_date format.")
        return

    df_long = pd.DataFrame([
        {"year": r["_id"]["year"], "paper": r["_id"]["paper"], "n_protest_articles": r["n"]}
        for r in results
    ])

    df_pivot = (
        df_long.pivot_table(
            index="year",
            columns="paper",
            values="n_protest_articles",
            aggfunc="sum",
            fill_value=0
        )
        .sort_index()
    )

    # Add totals
    df_pivot["TOTAL"] = df_pivot.sum(axis=1)
    totals_row = df_pivot.sum(axis=0).to_frame().T
    totals_row.index = ["TOTAL"]
    df_pivot = pd.concat([df_pivot, totals_row])

    print("\nCounts of PROTEST articles by year and outlet:")
    print(df_pivot.to_string())

    df_pivot.to_csv(OUT_PIVOT_CSV)
    df_long.to_csv(OUT_LONG_CSV, index=False)

    print(f"\nSaved pivot table to: {OUT_PIVOT_CSV}")
    print(f"Saved long table to:  {OUT_LONG_CSV}")

if __name__ == "__main__":
    main()
