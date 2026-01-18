import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymongo.mongo_client import MongoClient
from pymongo import UpdateOne
from tqdm import tqdm

# ---------------------------------------------------------------------
# 0. NLTK setup
# ---------------------------------------------------------------------
nltk.download("vader_lexicon")

# ---------------------------------------------------------------------
# 1. MongoDB connection
# ---------------------------------------------------------------------
#MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
MONGO_URI = " "
DB_NAME = "ProjectMaster"
#COLLECTION_NAME = "sample_texts"
COLLECTION_NAME = "Texts"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# ---------------------------------------------------------------------
# 2. VADER initialisation
# ---------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

# ---------------------------------------------------------------------
# 3. Helper to build the text we analyse
# ---------------------------------------------------------------------
def build_text(doc, max_chars=4000):
    """
    Build the text used for sentiment from the MongoDB document.
    You can adapt field names: title/text/content...
    """
    title = doc.get("title", "") or ""
    body = doc.get("text", "") or ""
    combined = (title + "\n\n" + body).strip()
    return combined[:max_chars]


def label_from_compound(c):
    """
    Map VADER compound score to a discrete label.
    """
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"


def main():
    BATCH_SIZE = 100
    # ---------------------------------------------------------------------
    # 4. Query: which docs to sentiment-annotate?
    # ---------------------------------------------------------------------
    query = {
        "hf_label_name": "PROTEST",      
        "sentiment": {"$exists": False},   # only process docs without sentiment yet
        "text": {"$exists": True}          # make sure there is some text
    }

    cursor = col.find(query)

    batch_ops = []
    processed = 0

    for doc in tqdm(cursor):
        text = build_text(doc)
        if not text:
            continue

        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        label = label_from_compound(compound)

        update = UpdateOne(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "sentiment": {
                        "label": label,
                        "neg": float(scores["neg"]),
                        "neu": float(scores["neu"]),
                        "pos": float(scores["pos"]),
                        "compound": float(compound),
                    }
                }
            }
        )
        batch_ops.append(update)
        processed += 1

        if len(batch_ops) >= BATCH_SIZE:
            col.bulk_write(batch_ops)
            batch_ops = []

    if batch_ops:
        col.bulk_write(batch_ops)

    cursor.close()
    print(f"Done. Sentiment added/updated for ~{processed} documents.")


if __name__ == "__main__":
    main()
