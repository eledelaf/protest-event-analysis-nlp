"""
BERTopic topic modelling for PROTEST articles stored in MongoDB Atlas.

Outputs:
  topic_modeling/
    - articles_with_topics.csv
    - topic_info.csv
    - representative_docs.csv
    - topic_share_by_time.csv
    - topic_share_by_paper.csv
    - topic_share_by_paper_time.csv
    - topics_barchart.html
    - topics_map.html 
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dateparser
from pymongo import MongoClient

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# MongoDB connection
# -----------------------------
# MONGO_URI = "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
MONGO_URI = " "
DB_NAME = "ProjectMaster"
COLLECTION_NAME = "Texts"

# -----------------------------
# Config
# -----------------------------
@dataclass
class MongoConfig:
    uri: str
    db_name: str
    collection_name: str

@dataclass
class TopicConfig:
    # Mongo fields
    text_field: str = "text"
    paper_field: str = "paper"
    date_field: str = "publish_date"  
    protest_filter: Dict[str, Any] = None

    # Time binning: "M" month, "Q" quarter (3-month bins)
    time_bin: str = "Q"

    # Data cleaning / filtering
    min_chars: int = 80 

    # Topic model knobs
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_topic_size: int = 40
    nr_topics: Optional[int] = None 

# -----------------------------
# Helpers
# -----------------------------
BOILERPLATE_PATTERNS = [
    r"^Advertisement\s*$",
    r"^Sign up to.*$",
    r"^Subscribe.*$",
    r"^Related articles.*$",
    r"^Read more.*$",
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(said|say|says)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip()

def parse_date(x):
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.to_datetime(x)
    except Exception:
        try:
            return pd.to_datetime(dateparser.parse(str(x)))
        except Exception:
            return None


def to_time_bin(ts: pd.Timestamp, freq):
    if freq == "M":
        return ts.to_period("M").to_timestamp()
    if freq == "Q":
        return ts.to_period("Q").to_timestamp()
    return ts.to_period(freq).to_timestamp()


# -----------------------------
# Data load
# -----------------------------
def load_protest_articles(mcfg, cfg):
    client = MongoClient(mcfg.uri, serverSelectionTimeoutMS=8000)
    # quick connectivity check (gives much clearer errors)
    client.admin.command("ping")
    print("✅ MongoDB connected")

    col = client[mcfg.db_name][mcfg.collection_name]

    if cfg.protest_filter is None:
        raise ValueError("TopicConfig.protest_filter must be set.")

    projection = {
        "_id": 1,
        cfg.text_field: 1,
        cfg.paper_field: 1,
        cfg.date_field: 1,
    }

    rows = list(col.find(cfg.protest_filter, projection=projection))
    if not rows:
        raise RuntimeError("No documents returned. Check protest_filter or field names.")

    df = pd.DataFrame(rows).rename(
        columns={
            "_id": "url",
            cfg.text_field: "text",
            cfg.paper_field: "paper",
            cfg.date_field: "published_date",
        }
    )

    df["published_date"] = df["published_date"].apply(parse_date)
    df = df.dropna(subset=["published_date"]).copy()

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() >= cfg.min_chars].copy()

    df["paper"] = df["paper"].fillna("Unknown")
    df["time_bin"] = df["published_date"].apply(lambda x: to_time_bin(x, cfg.time_bin))

    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df):,} protest docs after filtering (min_chars={cfg.min_chars})")
    return df

# -----------------------------
# Topic modelling
# -----------------------------
def fit_bertopic(df, cfg):
    docs = df["text"].tolist()

    embedder = SentenceTransformer(cfg.embedding_model_name)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5)

    topic_model = BERTopic(
        min_topic_size=cfg.min_topic_size,
        nr_topics=cfg.nr_topics,
        calculate_probabilities=True,
        verbose=True,
        vectorizer_model=vectorizer_model,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    if probs is None:
        max_prob = [None] * len(docs)
    else:
        max_prob = [float(p.max()) if hasattr(p, "max") else None for p in probs]

    return topic_model, topics, max_prob


def build_topic_tables(df):
    # shares by time bin
    tmp = df.groupby(["time_bin", "topic"]).size().rename("n").reset_index()
    tmp["share"] = tmp.groupby("time_bin")["n"].transform(lambda x: x / x.sum())
    topic_by_time = tmp.pivot(index="time_bin", columns="topic", values="share").fillna(0.0)

    # shares by paper
    tmp = df.groupby(["paper", "topic"]).size().rename("n").reset_index()
    tmp["share"] = tmp.groupby("paper")["n"].transform(lambda x: x / x.sum())
    topic_by_paper = tmp.pivot(index="paper", columns="topic", values="share").fillna(0.0)

    # shares by paper + time
    tmp = df.groupby(["paper", "time_bin", "topic"]).size().rename("n").reset_index()
    tmp["share"] = tmp.groupby(["paper", "time_bin"])["n"].transform(lambda x: x / x.sum())
    topic_by_paper_time = tmp.pivot_table(
        index=["paper", "time_bin"], columns="topic", values="share", fill_value=0.0
    )

    return topic_by_time, topic_by_paper, topic_by_paper_time


def export_results(out_dir, df, topic_model: BERTopic):
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "articles_with_topics.csv"), index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(out_dir, "topic_info.csv"), index=False)

    rep_docs = topic_model.get_representative_docs()
    rep_rows = []
    for topic_id, docs in rep_docs.items():
        for i, d in enumerate(docs[:5]):
            rep_rows.append({"topic": topic_id, "rank": i + 1, "doc_snippet": d[:400]})
    pd.DataFrame(rep_rows).to_csv(os.path.join(out_dir, "representative_docs.csv"), index=False)

    try:
        topic_model.visualize_barchart(top_n_topics=20).write_html(os.path.join(out_dir, "topics_barchart.html"))
    except Exception:
        pass

    try:
        topic_model.visualize_topics().write_html(os.path.join(out_dir, "topics_map.html"))
    except Exception:
        pass


def main():
    mongo = MongoConfig(uri=MONGO_URI, db_name=DB_NAME, collection_name=COLLECTION_NAME)

    cfg = TopicConfig(
        text_field="text",
        paper_field="paper",
        date_field="publish_date",
        protest_filter={"hf_label_name": "PROTEST", "status": "done"},
        time_bin="Q",          
        min_chars=80,          
        min_topic_size=40,
        nr_topics=None,
    )

    out_dir = "6.Topic_analysis/topic_modeling"

    df = load_protest_articles(mongo, cfg)

    topic_model, topics, max_prob = fit_bertopic(df, cfg)
    df["topic"] = topics
    df["topic_confidence"] = max_prob

    topic_by_time, topic_by_paper, topic_by_paper_time = build_topic_tables(df)
    os.makedirs(out_dir, exist_ok=True)
    topic_by_time.to_csv(os.path.join(out_dir, "topic_share_by_time.csv"))
    topic_by_paper.to_csv(os.path.join(out_dir, "topic_share_by_paper.csv"))
    topic_by_paper_time.to_csv(os.path.join(out_dir, "topic_share_by_paper_time.csv"))

    export_results(out_dir, df, topic_model)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
