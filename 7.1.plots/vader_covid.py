"""
Analyze VADER sentiment compound scores by COVID period.
"""
import pandas as pd
from pymongo import MongoClient
from pathlib import Path

# ----------------------------
# Config 
# ----------------------------
# MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
MONGO_URI = " "
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

# Choose how to define "final PROTEST"
# - If True: uses hf_reason to keep only docs decided as PROTEST by threshold logic
# - If False: uses hf_label_name == "PROTEST"
USE_FINAL_THRESHOLD_DECISION = True

# COVID period cutoffs 
PRE_START = "2020-01-01"
COVID_START = "2020-03-11"
COVID_END = "2022-02-24"
POST_END = "2024-12-31"  

OUT_DIR = Path("7.3outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def assign_period(d):
    # assumes d is not NaT
    if d < pd.Timestamp(COVID_START):
        return "Pre-COVID"
    elif d < pd.Timestamp(COVID_END):
        return "COVID"
    else:
        return "Post-COVID"

def main():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    query = {
        "publish_date": {"$exists": True},
        "paper": {"$exists": True},
        "sentiment.compound": {"$exists": True},
    }

    query["publish_date"]["$gte"] = PRE_START
    query["publish_date"]["$lte"] = POST_END

    if USE_FINAL_THRESHOLD_DECISION:
        # Keeps only the docs where hf_reason says it ended as PROTEST
        query["hf_reason"] = {"$regex": r"->\s*PROTEST\s*$"}
    else:
        query["hf_label_name"] = "PROTEST"

    projection = {
        "_id": 1,
        "paper": 1,
        "publish_date": 1,
        "sentiment.compound": 1,
    }

    docs = list(col.find(query, projection))
    if not docs:
        print("No documents matched the query. Check filters / dates / fields.")
        return

    df = pd.DataFrame(docs).rename(columns={"_id": "url"})
    df["compound"] = pd.to_numeric(df["sentiment"].apply(lambda x: x.get("compound") if isinstance(x, dict) else None),
                                  errors="coerce")
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    # Clean
    df = df.dropna(subset=["publish_date", "paper", "compound"]).copy()

    # Assign COVID period
    df["covid_period"] = df["publish_date"].apply(assign_period)

    # ----------------------------
    # Overall summary by period
    # ----------------------------
    overall = (
        df.groupby("covid_period")["compound"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values("covid_period")
    )

    # ----------------------------
    # By outlet + period
    # ----------------------------
    by_outlet = (
        df.groupby(["paper", "covid_period"])["compound"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values(["paper", "covid_period"])
    )

    # Pivot table: mean compound (rows=paper, cols=period)
    pivot_mean = by_outlet.pivot(index="paper", columns="covid_period", values="mean")

    # ----------------------------
    # Print
    # ----------------------------
    print("\nMean VADER compound by COVID period (overall):")
    print(overall.to_string(index=False))

    print("\nMean VADER compound by outlet and COVID period:")
    print(by_outlet.to_string(index=False))

    print("\nPivot (mean compound):")
    print(pivot_mean)

    # ----------------------------
    # Save outputs
    # ----------------------------
    overall.to_csv(OUT_DIR / "sentiment_mean_overall_by_covid_period.csv", index=False)
    by_outlet.to_csv(OUT_DIR / "sentiment_mean_by_outlet_and_covid_period.csv", index=False)
    pivot_mean.to_csv(OUT_DIR / "sentiment_mean_pivot_outlet_x_covid_period.csv")

    print(f"\nSaved CSVs to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
