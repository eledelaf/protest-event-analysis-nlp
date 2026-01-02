"""
This code downloads the data set from MongoDB
"""
import gzip
from pymongo import MongoClient
from bson.json_util import dumps

MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
DB_NAME = "ProjectMaster"
COLLECTION_NAME_TEXTS = "Texts"
COLLECTION_NAME_SAMPLE = "sample_texts"
BATCH_SIZE = 50

def export_collection(collection_name: str, out_path: str):
    client = MongoClient(MONGO_URI)
    coll = client[DB_NAME][collection_name]

    total = coll.estimated_document_count()
    print(f"Exporting {total} docs from {DB_NAME}.{collection_name} -> {out_path}")

    #cursor = coll.find({}, no_cursor_timeout=True).batch_size(BATCH_SIZE)
    cursor = coll.find({}).batch_size(BATCH_SIZE)

    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        try:
            for doc in cursor:
                f.write(dumps(doc))   # keeps ObjectId/datetime safe
                f.write("\n")
                n += 1
                if n % 1000 == 0:
                    print(f"  exported {n} docs...")
        finally:
            cursor.close()
            client.close()

    print(f"Done. Exported {n} documents.")

if __name__ == "__main__":
    export_collection(COLLECTION_NAME_TEXTS, "8.MongoDB/Texts.jsonl.gz")
    export_collection(COLLECTION_NAME_SAMPLE, "8.MongoDB/sample_texts.jsonl.gz")