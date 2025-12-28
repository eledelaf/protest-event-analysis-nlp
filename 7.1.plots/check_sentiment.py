"""
Check sentiment distribution.
"""
from pymongo.mongo_client import MongoClient
from pprint import pprint

MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
#COLLECTION_NAME = "sample_texts"
COLLECTION_NAME = "Texts"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

pipeline = [
    {"$match": {"sentiment.label": {"$exists": True}}},
    {"$group": {"_id": "$sentiment.label", "count": {"$sum": 1}}},
]

for row in col.aggregate(pipeline):
    pprint(row)
