"""
Hugging Face (Transformers) zero-shot classifier helper.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from transformers import pipeline

# ----------------------------
# Model + labels
# ----------------------------
HF_MODEL_NAME = "facebook/bart-large-mnli"

# Make the labels descriptive so the model understands the task.
PROTEST_LABEL = "a concrete real-world protest event"
OTHER_LABEL = "something else (no specific protest event)"
CANDIDATE_LABELS = [PROTEST_LABEL, OTHER_LABEL]

_zsc = pipeline(
    task="zero-shot-classification",
    model=HF_MODEL_NAME,
)


def classify_article_with_hf( title, text, *, protest_threshold: float = 0.65, max_chars: int = 4000, min_length: int = 200):
    """
    Classify a document as PROTEST / NOT PROTEST.
    Returns None when text is missing or too short.
   """
    if not text or len(text.strip()) < min_length:
        return None

    truncated_text = text[:max_chars]
    sequence = f"Title: {title}\n\nArticle:\n{truncated_text}"

    result = _zsc(
        sequence,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="The main focus of this article is {}.",
        multi_label=False,
    )

    labels = result.get("labels", [])
    scores = result.get("scores", [])
    if not labels or not scores or len(labels) != len(scores):
        raise ValueError(f"Unexpected classifier output: {result}")

    top_label = str(labels[0])
    top_score = float(scores[0])

    # Always treat "confidence" as P(PROTEST).
    try:
        protest_idx = labels.index(PROTEST_LABEL)
    except ValueError as e:
        raise ValueError(f"PROTEST_LABEL not found in returned labels: {labels}") from e

    confidence = float(scores[protest_idx])

    is_protest = confidence >= protest_threshold
    label_int = 1 if is_protest else 0
    label_name = "PROTEST" if is_protest else "NOT PROTEST"

    reason = (
        f"Top='{top_label}' ({top_score:.3f}); "
        f"P(PROTEST)={confidence:.3f}; "
        f"threshold={protest_threshold:.2f} -> {label_name}"
    )

    return {
        "confidence": confidence,
        "label": label_int,
        "label_name": label_name,
        "top_label": top_label,
        "top_score": top_score,
        "model": HF_MODEL_NAME,
        "reason": reason,
    }
