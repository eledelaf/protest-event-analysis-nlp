"""
This code computes the best threshold for a binary classifier.
Optimizes F0.5 score.
"""
import os
from pymongo import MongoClient

#MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://eledelaf:Ly5BX57aSXIzJVde@articlesprotestdb.bk5rtxs.mongodb.net/?retryWrites=true&w=majority&appName=ArticlesProtestDB")
MONGO_URI = ""
DB_NAME = "ProjectMaster"
COLLECTION = "sample_texts"

def metrics(tp, fp, tn, fn):
    """
    Computes various metrics from confusion matrix values.
    """
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    beta2 = 0.25  # beta = 0.5
    denom = beta2 * prec + rec
    f05 = ((1 + beta2) * prec * rec / denom) if denom else 0.0
    return acc, prec, rec, f1, f05

def confusion(y_true, scores, t):
    """
    Computes confusion matrix values for a given threshold.
    """
    tp = fp = tn = fn = 0
    for h, s in zip(y_true, scores):
        pred = 1 if s >= t else 0
        if h == 1 and pred == 1: tp += 1
        elif h == 0 and pred == 1: fp += 1
        elif h == 0 and pred == 0: tn += 1
        elif h == 1 and pred == 0: fn += 1
    return tp, fp, tn, fn

def main():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    docs = list(col.find(
        {"human_label": {"$exists": True}, "hf_confidence": {"$type": "number"}},
        {"_id": 0, "human_label": 1, "hf_confidence": 1},
    ))

    y_true = [int(d["human_label"]) for d in docs]
    scores = [float(d["hf_confidence"]) for d in docs]

    print(f"Loaded {len(docs)} validation docs with human_label + hf_confidence.")

    best = {"t": None, "f05": -1}
    for i in range(5, 91):  # 0.05..0.90
        t = i / 100.0
        tp, fp, tn, fn = confusion(y_true, scores, t)
        acc, prec, rec, f1, f05 = metrics(tp, fp, tn, fn)

        print(f"t={t:.2f}  F0.5={f05:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  Acc={acc:.3f}  (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

        if f05 > best["f05"]:
            best = {"t": t, "f05": f05, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "acc": acc, "prec": prec, "rec": rec, "f1": f1}

    print("\n=== Best threshold (max F0.5) ===")
    print(f"Best t = {best['t']:.2f}")
    print(f"TP={best['tp']} FP={best['fp']} TN={best['tn']} FN={best['fn']}")
    print(f"Accuracy={best['acc']:.3f} Precision={best['prec']:.3f} Recall={best['rec']:.3f} F1={best['f1']:.3f} F0.5={best['f05']:.3f}")

if __name__ == "__main__":
    main()
