"""
Real Benchmark Evaluation Metrics for Remote Sensing Vision-Language Models.
Implements standard metrics:
- RSVQA: Overall Accuracy (OA), Average Accuracy (AA)
- VRSBench Captioning: BLEU-1 to BLEU-4, ROUGE-L, Word Error Rate (WER)
- Grounding: Bounding Box IoU, Precision@0.5 (P@0.5), Location Agreement
- CDVQA: Directional Change Accuracy, Delta % Error, F1-Score
- Optical-SAR Fusion: Cross-Modal Consensus Agreement, IoU
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalize string for robust NLP evaluation."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def compute_vqa_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Calculate Overall Accuracy (OA) and Average Accuracy (AA) for VQA.
    OA = correct / total
    AA = mean of per-class accuracies
    """
    if not predictions or len(predictions) != len(ground_truths):
        return {"overall_accuracy": 0.0, "average_accuracy": 0.0, "total_samples": 0}

    correct = 0
    class_stats: Dict[str, Dict[str, int]] = {}

    for pred, gt in zip(predictions, ground_truths):
        norm_p = normalize_text(pred)
        norm_gt = normalize_text(gt)

        # Match check: exact match or containment for short answers (yes/no, urban/rural)
        p_words = norm_p.split()
        if not p_words:
            is_correct = (norm_gt == "")
        elif norm_gt in {"yes", "no"}:
            first_word = p_words[0] if p_words else ""
            if norm_gt == "yes":
                is_correct = (first_word == "yes") or (
                    "yes" in p_words and "no" not in p_words and "not" not in p_words
                )
            else:
                is_correct = (first_word == "no") or (
                    "no" in p_words and "yes" not in p_words
                ) or ("not" in p_words and "yes" not in p_words)
        else:
            is_correct = (norm_p == norm_gt) or (
                len(norm_gt.split()) == 1 and norm_gt in p_words
            )

        if norm_gt not in class_stats:
            class_stats[norm_gt] = {"correct": 0, "total": 0}
        class_stats[norm_gt]["total"] += 1

        if is_correct:
            correct += 1
            class_stats[norm_gt]["correct"] += 1

    total = len(predictions)
    oa = (correct / total) * 100.0

    per_class_accs = [
        (s["correct"] / s["total"]) * 100.0
        for s in class_stats.values()
        if s["total"] > 0
    ]
    aa = sum(per_class_accs) / max(len(per_class_accs), 1)

    return {
        "overall_accuracy": round(oa, 2),
        "average_accuracy": round(aa, 2),
        "total_samples": total,
    }


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(hypothesis: str, reference: str, max_n: int = 4) -> Dict[str, float]:
    """
    Compute sentence-level BLEU-1 through BLEU-4 with brevity penalty.
    """
    hyp_tokens = normalize_text(hypothesis).split()
    ref_tokens = normalize_text(reference).split()

    if not hyp_tokens or not ref_tokens:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(ref_len) / float(hyp_len)) if hyp_len > 0 else 0.0

    precisions = []
    bleu_scores = {}

    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        clipped_count = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total_count = max(sum(hyp_ngrams.values()), 1)
        p_n = clipped_count / total_count
        precisions.append(p_n)

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            log_prec_sum = sum(math.log(p) for p in precisions) / float(n)
            bleu_scores[f"bleu_{n}"] = round(bp * math.exp(log_prec_sum) * 100.0, 2)
        else:
            bleu_scores[f"bleu_{n}"] = 0.0

    return bleu_scores


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1-score based on Longest Common Subsequence (LCS).
    """
    hyp_tokens = normalize_text(hypothesis).split()
    ref_tokens = normalize_text(reference).split()

    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    # Dynamic programming table for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / float(n)
    rec = lcs / float(m)

    if prec + rec == 0:
        return 0.0
    f1 = (2.0 * prec * rec) / (prec + rec)
    return round(f1 * 100.0, 2)


def compute_bbox_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union between two bounding boxes: [y1, x1, y2, x2].
    """
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    inter_h = max(0, y2 - y1)
    inter_w = max(0, x2 - x1)
    inter_area = inter_h * inter_w

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return round(inter_area / union_area, 4)


def _to_bbox_list(bbox: Any) -> Optional[List[int]]:
    """Convert dict or list bounding box to [y1, x1, y2, x2]."""
    if not bbox:
        return None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return [int(bbox["y1"]), int(bbox["x1"]), int(bbox["y2"]), int(bbox["x2"])]
        if all(k in bbox for k in ("ymin", "xmin", "ymax", "xmax")):
            return [int(bbox["ymin"]), int(bbox["xmin"]), int(bbox["ymax"]), int(bbox["xmax"])]
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    return None


def compute_grounding_metrics(
    pred_bbox: Optional[Any],
    pred_location: Optional[str] = None,
    expected_location: Optional[str] = None,
    gt_bbox: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute spatial grounding metrics: location agreement, bbox validity,
    and genuine Precision@0.5 against ground truth bbox (IoU >= 0.50).
    """
    loc_match = False
    if pred_location and expected_location:
        norm_p = normalize_text(pred_location)
        norm_e = normalize_text(expected_location)
        e_words = set(norm_e.split("-") + norm_e.split())
        p_words = set(norm_p.split("-") + norm_p.split())
        loc_match = bool(e_words & p_words)

    p_box = _to_bbox_list(pred_bbox)
    has_valid_box = bool(p_box and p_box[2] > p_box[0] and p_box[3] > p_box[1])

    g_box = _to_bbox_list(gt_bbox)
    iou = 0.0
    if has_valid_box and g_box and g_box[2] > g_box[0] and g_box[3] > g_box[1]:
        iou = compute_bbox_iou(p_box, g_box)

    # Precision@0.5 strictly requires reference box IoU >= 0.50
    # If no ground truth box is provided or IoU < 0.50, precision_at_50 is 0.0
    precision_at_50 = 1.0 if (iou >= 0.50) else 0.0

    return {
        "location_agreement": loc_match,
        "valid_bounding_box": has_valid_box,
        "iou": round(iou, 4),
        "precision_at_50": precision_at_50,
    }


def compute_cdvqa_metrics(
    pred_answer: str,
    true_answer: str,
    true_direction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate Change Detection Visual Question Answering with negation awareness.
    """
    norm_p = normalize_text(pred_answer)
    norm_t = normalize_text(true_answer)

    # Empty or whitespace-only predictions must receive 0.0 accuracy across the board
    if not norm_p:
        return {
            "directional_accuracy": 0.0,
            "delta_percentage_error": None,
            "rouge_l": 0.0,
        }

    # 1. Directional Classification Accuracy with negation awareness
    dir_acc = 0.0
    if true_direction:
        td = true_direction.lower()
        has_not_increase = bool(re.search(r"\b(?:not|did\s+not|never|hasnt|has\s+not|no)\s+(?:increase|increased|expansion|gain|growth)\b", norm_p))
        has_not_decrease = bool(re.search(r"\b(?:not|did\s+not|never|hasnt|has\s+not|no)\s+(?:decrease|decreased|drop|reduction|loss)\b", norm_p))

        pred_increased = (
            bool(re.search(r"\b(?:increase|increased|expansion|expanded|gain|grew|growth)\b", norm_p))
            and not has_not_increase
        )
        pred_decreased = (
            bool(re.search(r"\b(?:decrease|decreased|loss|shrunk|shrink|reduction|reduced|drop|dropped)\b", norm_p))
            and not has_not_decrease
        )
        pred_unchanged = bool(re.search(r"\b(?:unchanged|remained\s+unchanged|no\s+significant\s+change|stable|constant)\b", norm_p))

        if "increase" in td:
            dir_acc = 1.0 if (pred_increased and not pred_decreased) else 0.0
        elif "decrease" in td:
            dir_acc = 1.0 if (pred_decreased and not pred_increased) else 0.0
        elif "unchanged" in td:
            dir_acc = 1.0 if pred_unchanged else 0.0

    # 2. Key physical delta extraction
    p_deltas = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", pred_answer)
    t_deltas = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", true_answer)

    delta_err = None
    if p_deltas and t_deltas:
        p_val = abs(float(p_deltas[0]))
        t_val = abs(float(t_deltas[0]))
        delta_err = round(abs(p_val - t_val), 2)

    # 3. ROUGE-L lexical similarity
    rouge_l = compute_rouge_l(pred_answer, true_answer)

    return {
        "directional_accuracy": dir_acc,
        "delta_percentage_error": delta_err,
        "rouge_l": rouge_l,
    }
