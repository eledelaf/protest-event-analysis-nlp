"""
Check article titles for a specific week end.
"""
import pandas as pd
from pymongo import MongoClient
from datetime import timedelta

MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION = "Texts"

client = MongoClient(MONGO_URI)
col = client[DB_NAME][COLLECTION]

def show_titles_for_week_end(week_end_str, limit=20):
    week_end = pd.to_datetime(week_end_str).tz_localize("UTC")
    start = (week_end - pd.Timedelta(days=6)).date().isoformat()
    end = week_end.date().isoformat()

    query = {
        "hf_reason": {"$regex": r"->\s*PROTEST\s*$"},  # final decision
        "publish_date": {"$gte": start, "$lte": end},
        "title": {"$exists": True},
    }
    proj = {"_id": 0, "publish_date": 1, "paper": 1, "title": 1}
    docs = list(col.find(query, proj).limit(limit))

    df = pd.DataFrame(docs).sort_values(["publish_date", "paper"])
    print(f"\nWeek window: {start} to {end} (week_end={week_end_str})  n_shown={len(df)}")
    print(df.to_string(index=False))
