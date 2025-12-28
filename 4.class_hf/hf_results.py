
"""
This script analyzes the results of the Hugging Face classifier.
"""

import os
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

THRESHOLD = 0.57
WINDOW = 0.02          # “near threshold” = +/- 0.02
SAMPLE_N = 200000      # for plotting 

def agg_to_df(coll, pipeline):
    return pd.DataFrame(list(coll.aggregate(pipeline, allowDiskUse=True)))

def main():
    client = MongoClient(MONGO_URI)
    coll = client[DB_NAME][COLLECTION_NAME]

    # 1. Counts by final label
    df_label = agg_to_df(coll, [
        {"$match": {"hf_label_name": {"$exists": True}}},
        {"$group": {"_id": "$hf_label_name", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
        ])
    print("\n=== Counts by hf_label_name ===")
    print(df_label.to_string(index=False) if not df_label.empty else "No hf_label_name found.")

    # 2. Counts by status
    df_status = agg_to_df(coll, [
        {"$match": {"hf_status": {"$exists": True}}},
        {"$group": {"_id": "$hf_status", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ])
    print("\n=== Counts by hf_status ===")
    print(df_status.to_string(index=False) if not df_status.empty else "No hf_status found.")

    # 3. Near-threshold counts
    lo, hi = THRESHOLD - WINDOW, THRESHOLD + WINDOW
    n_near = coll.count_documents({"hf_confidence": {"$gte": lo, "$lt": hi}})
    n_below = coll.count_documents({"hf_confidence": {"$lt": THRESHOLD}})
    n_above = coll.count_documents({"hf_confidence": {"$gte": THRESHOLD}})
    n_total_scored = coll.count_documents({"hf_confidence": {"$type": "number"}})

    print(f"\n=== Confidence sanity check ===")
    print(f"Total with hf_confidence: {n_total_scored}")
    print(f"Below threshold (< {THRESHOLD}): {n_below}")
    print(f"At/above threshold (>= {THRESHOLD}): {n_above}")
    print(f"Near threshold in [{lo:.2f}, {hi:.2f}): {n_near}")

    # 4. Plot distribution (sampled)
    cursor = coll.find(
        {"hf_confidence": {"$type": "number"}},
        {"_id": 0, "hf_confidence": 1}
    ).limit(SAMPLE_N)

    vals = [d["hf_confidence"] for d in cursor if "hf_confidence" in d]

    if not vals:
        print("\nNo hf_confidence values to plot.")
        return

    plt.figure()
    plt.hist(vals, bins=50)
    plt.axvline(THRESHOLD)
    plt.title("Distribution of hf_confidence (sampled)")
    plt.xlabel("hf_confidence = P(PROTEST)")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
