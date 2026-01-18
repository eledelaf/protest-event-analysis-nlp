"""
Check sentiment distribution across newspapers.
"""

from pymongo.mongo_client import MongoClient
from pprint import pprint

# MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
MONGO_URI = " "
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

pipeline = [
    {"$match": {"hf_label_name": "PROTEST"}},
    {"$group": {"_id": "$paper", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]

for row in col.aggregate(pipeline):
    print(row)

    pipeline = [
    {"$match": {
        "hf_label_name": "PROTEST",
        "sentiment.label": {"$exists": True}
    }},
    {"$group": {
        "_id": {"paper": "$paper", "sentiment": "$sentiment.label"},
        "count": {"$sum": 1}
    }},
    {"$sort": {
        "_id.paper": 1,
        "_id.sentiment": 1
    }}
]

for row in col.aggregate(pipeline):
    print(row["_id"]["paper"], row["_id"]["sentiment"], row["count"])
