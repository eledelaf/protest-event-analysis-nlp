"""
Run a Hugging Face zero-shot classifier over a MongoDB collection.
If an entry has hf_confidence, reclasify it with the new threshold.
In other case, classify it, and new hf_confidence.

Threshold default to 0.65.
"""

from __future__ import annotations

from typing import Dict, Any, List
import argparse
import os

from pymongo.mongo_client import MongoClient
from pymongo import UpdateOne
from pymongo.errors import PyMongoError
from tqdm import tqdm

import re
from bson.decimal128 import Decimal128

# ----------------------------
# CONFIG
# ----------------------------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB"
)

DB_NAME_DEFAULT = "ProjectMaster"
#COLLECTION_NAME_DEFAULT = "Texts"
COLLECTION_NAME_DEFAULT = "sample_texts"
BATCH_SIZE = 50

def parse_args():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=str, default=DB_NAME_DEFAULT)
    p.add_argument("--collection", type=str, default=COLLECTION_NAME_DEFAULT)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--force", action="store_true", help="Re-classify all docs even if hf_status=='ok'.")
    p.add_argument("--min_length", type=int, default=200)
    p.add_argument("--max_chars", type=int, default=4000)
    p.add_argument("--relabel_only",action="store_true", help="Only update hf_label/hf_label_name using existing hf_confidence and --threshold. No HF inference.",)
    p.add_argument("--debug_id", type=str, default=None, help="Print before/after for one _id (URL).")
    p.add_argument("--dry_run", action="store_true", help="Compute but do not write updates.")
    p.add_argument("--limit", type=int, default=0, help="Limit relabel docs for testing (0 = no limit).")
    p.add_argument("--hybrid",action="store_true", help="Relabel docs that already have hf_confidence; classify only docs missing hf_confidence.")
    return p.parse_args()

_TOP_RE = re.compile(r"Top='(?P<label>.*?)'\s*\((?P<score>[0-9]*\.?[0-9]+)\)")

def _to_float(x):
    """
    Convert a value to float.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, Decimal128):
        return float(x.to_decimal())
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None

def _extract_top_from_reason(reason):
    """
    Extract top label and score from reason string.
    """
    if not reason:
        return None, None
    m = _TOP_RE.search(reason)
    if not m:
        return None, None
    return m.group("label"), _to_float(m.group("score"))

def relabel_from_confidence(col, threshold, *, debug_id=None, dry_run=False, limit=0):
    """
    Recompute hf_label, hf_label_name, hf_reason for ALL docs with hf_confidence.
    """
    if debug_id:
        before = col.find_one(
            {"_id": debug_id},
            {"_id": 1, "hf_confidence": 1, "hf_label": 1, "hf_label_name": 1, "hf_reason": 1},
        )
        print("\n[debug] BEFORE:", before)

        conf = _to_float((before or {}).get("hf_confidence"))
        if conf is None:
            print("[debug] This doc has no numeric hf_confidence; cannot relabel.")
            return

        is_protest = conf >= threshold
        label_name = "PROTEST" if is_protest else "NOT PROTEST"
        label_int = 1 if is_protest else 0

        old_reason = (before or {}).get("hf_reason") or ""
        top_label, top_score = _extract_top_from_reason(old_reason)
        if top_label is not None and top_score is not None:
            new_reason = (
                f"Top='{top_label}' ({top_score:.3f}); "
                f"P(PROTEST)={conf:.3f}; threshold={threshold:.2f} -> {label_name}"
            )
        else:
            new_reason = f"P(PROTEST)={conf:.3f}; threshold={threshold:.2f} -> {label_name}"

        if not dry_run:
            col.update_one(
                {"_id": debug_id},
                {"$set": {"hf_label": label_int, "hf_label_name": label_name, "hf_reason": new_reason}},
            )

        after = col.find_one(
            {"_id": debug_id},
            {"_id": 1, "hf_confidence": 1, "hf_label": 1, "hf_label_name": 1, "hf_reason": 1},
        )
        print("\n[debug] AFTER:", after)
        return

    base_query = {"hf_confidence": {"$exists": True, "$ne": None}}
    projection = {"_id": 1, "hf_confidence": 1, "hf_reason": 1}

    READ_BATCH = 1000  # how many docs to read per chunk
    last_id = None

    scanned = 0
    convertible = 0

    while True:
        q = dict(base_query)
        if last_id is not None:
            q["_id"] = {"$gt": last_id}

        batch = list(
            col.find(q, projection)
              .sort("_id", 1)
              .limit(READ_BATCH)
        )
        if not batch:
            break

        ops: List[UpdateOne] = []

        for doc in batch:
            scanned += 1
            last_id = doc["_id"]

            if limit and scanned > limit:
                break

            conf = _to_float(doc.get("hf_confidence"))
            if conf is None:
                continue
            convertible += 1

            is_protest = conf >= threshold
            label_name = "PROTEST" if is_protest else "NOT PROTEST"
            label_int = 1 if is_protest else 0

            old_reason = doc.get("hf_reason") or ""
            top_label, top_score = _extract_top_from_reason(old_reason)

            if top_label is not None and top_score is not None:
                new_reason = (
                    f"Top='{top_label}' ({top_score:.3f}); "
                    f"P(PROTEST)={conf:.3f}; threshold={threshold:.2f} -> {label_name}"
                )
            else:
                new_reason = f"P(PROTEST)={conf:.3f}; threshold={threshold:.2f} -> {label_name}"

            ops.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"hf_label": label_int, "hf_label_name": label_name, "hf_reason": new_reason}},
                )
            )

            if len(ops) >= BATCH_SIZE:
                if not dry_run:
                    res = col.bulk_write(ops, ordered=False)
                    print(f"[relabel] batch modified={res.modified_count}")
                ops = []

        if ops and not dry_run:
            res = col.bulk_write(ops, ordered=False)
            print(f"[relabel] batch modified={res.modified_count}")

        if limit and scanned >= limit:
            break

    print(f"[relabel] scanned={scanned}  convertible_confidence={convertible}  dry_run={dry_run}")

def _flush(col, ops, *, tag):
    """
    Write a batch of UpdateOne ops safely.
    """
    if not ops:
        return
    try:
        res = col.bulk_write(ops, ordered=False)
        # print only when something actually changed (avoids spam)
        if res.modified_count:
            print(f"[{tag}] modified={res.modified_count} matched={res.matched_count}")
    except PyMongoError as e:
        print(f"[Mongo] bulk_write error: {e}")

def classify_missing_confidence(col, args):
    """
    Classify ONLY documents that don't have hf_confidence yet.
    """
    from hf_class import classify_article_with_hf

    base_query: Dict[str, Any] = {"text": {"$exists": True, "$ne": None, "$ne": ""}}

    if not args.force:
        # only docs that are missing confidence (or have it as null)
        base_query["$or"] = [
            {"hf_confidence": {"$exists": False}},
            {"hf_confidence": None},
        ]

    projection = {"_id": 1, "title": 1, "text": 1, "hf_confidence": 1, "hf_status": 1}
    READ_BATCH = 250
    last_id = None

    scanned = 0
    attempted = 0

    while True:
        q = dict(base_query)
        if last_id is not None:
            q["_id"] = {"$gt": last_id}

        batch = list(col.find(q, projection).sort("_id", 1).limit(READ_BATCH))
        if not batch:
            break

        ops: List[UpdateOne] = []

        for doc in batch:
            scanned += 1
            last_id = doc["_id"]

            if args.limit and scanned > args.limit:
                break

            doc_id = doc["_id"]
            title = doc.get("title") or ""
            text = doc.get("text") or ""

            try:
                attempted += 1
                res = classify_article_with_hf(
                    title,
                    text,
                    protest_threshold=args.threshold,
                    min_length=args.min_length,
                    max_chars=args.max_chars,
                )
            except Exception as e:
                ops.append(
                    UpdateOne(
                        {"_id": doc_id},
                        {"$set": {"hf_status": "error", "hf_error_message": str(e)}},
                    )
                )
            else:
                if res is None:
                    ops.append(
                        UpdateOne(
                            {"_id": doc_id},
                            {
                                "$set": {
                                    "hf_status": "skipped_short_text",
                                    "hf_reason": f"Skipped: text shorter than min_length={args.min_length}",
                                }
                            },
                        )
                    )
                else:
                    # store full classification output
                    payload = {
                        "hf_confidence": res["confidence"],
                        "hf_label": res["label"],
                        "hf_label_name": res["label_name"],
                        "hf_model": res["model"],
                        "hf_reason": res["reason"],
                        "hf_status": "ok",
                    }
                    # optional if classifier returns it
                    if "top_label" in res:
                        payload["hf_top_label"] = res["top_label"]
                    if "top_score" in res:
                        payload["hf_top_score"] = res["top_score"]

                    ops.append(UpdateOne({"_id": doc_id}, {"$set": payload}))

            if len(ops) >= BATCH_SIZE:
                if not args.dry_run:
                    #_flush(col, ops)
                    _flush(col, ops, tag="classify_missing")
                ops = []

        if ops and not args.dry_run:
            #_flush(col, ops)
            _flush(col, ops, tag="classify_missing")

        if args.limit and scanned >= args.limit:
            break

    print(
        f"[classify_missing] scanned={scanned} attempted_inference={attempted} "
        f"dry_run={args.dry_run} force={args.force}"
    )

def main() -> None:
    args = parse_args()
    print(f"[run_hf] threshold={args.threshold:.2f}  db={args.db}  collection={args.collection}")

    client = MongoClient(MONGO_URI)
    col = client[args.db][args.collection]

    if args.hybrid:
        print("[run_hf] Hybrid mode:")
        print("  1) Relabel docs that already have hf_confidence using current threshold")
        print("  2) Classify docs missing hf_confidence (newly scraped)")

        # 1. relabel existing confidence
        relabel_from_confidence(
            col,
            args.threshold,
            debug_id=args.debug_id,
            dry_run=args.dry_run,
            limit=args.limit,
        )

        # 2. classify missing confidence
        classify_missing_confidence(col, args)

        print("[run_hf] Done (hybrid).")
        return

    # --- Existing modes preserved ---
    if args.relabel_only:
        print("[run_hf] Relabel-only mode: updating labels from existing hf_confidence (no HF inference).")
        relabel_from_confidence(
            col,
            args.threshold,
            debug_id=args.debug_id,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        return

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += ["--hybrid", "--threshold", "0.65"]
    main()


