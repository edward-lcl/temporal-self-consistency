"""
TSCT Evaluation Pipeline — Week 2 Deliverables
================================================
Jason's evaluation infrastructure for the TSCT project.

Contains:
1. ECE pipeline (equal-frequency binning, softmax confidence)
2. Accuracy metrics (exact match + F1)
3. Volatility-split breakdowns (Fast / Slow / Immutable)
4. Bonferroni correction framework (alpha ~ 0.007, 7 comparisons)
5. Volatility discrimination scoring
6. Temporal generalization gap metric
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


# ============================================================
# 1. ECE PIPELINE — Equal-frequency binning
# ============================================================

# Hedge token -> confidence mapping (eval-time fixed scalars)
HEDGE_TO_CONFIDENCE = {
    "[CONFIDENT]":       0.95,
    "[COND_CONFIDENT]":  0.75,
    "[TEMPORAL_HEDGE]":  0.45,
    "[UNKNOWN]":         0.10,
}


def compute_ece(predictions: List[Dict], n_bins: int = 10) -> Dict:
    """
    Compute Expected Calibration Error with equal-frequency binning.

    Each prediction dict must have:
        - "predicted_hedge": one of the 4 hedge tokens
        - "correct": bool (was the factual answer correct)

    During training, confidence = softmax probability of emitted hedge token.
    During evaluation (here), confidence = fixed scalar from HEDGE_TO_CONFIDENCE.

    Returns dict with:
        - ece: float (the ECE value, lower is better)
        - bins: list of bin dicts with avg_confidence, accuracy, count, bin_range
        - n_total: int
    """
    if not predictions:
        return {"ece": 0.0, "bins": [], "n_total": 0}

    # Map each prediction to (confidence, correctness)
    pairs = []
    for pred in predictions:
        hedge = pred["predicted_hedge"]
        conf = HEDGE_TO_CONFIDENCE.get(hedge, 0.5)
        correct = 1.0 if pred["correct"] else 0.0
        pairs.append((conf, correct))

    # Sort by confidence for equal-frequency binning
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)

    # Equal-frequency: each bin has the same number of examples
    bin_size = max(1, n // n_bins)
    bins = []
    ece = 0.0

    for i in range(0, n, bin_size):
        bin_pairs = pairs[i:i + bin_size]
        if not bin_pairs:
            continue

        confs = [p[0] for p in bin_pairs]
        accs  = [p[1] for p in bin_pairs]

        avg_conf = np.mean(confs)
        avg_acc  = np.mean(accs)
        count    = len(bin_pairs)
        weight   = count / n

        gap = abs(avg_acc - avg_conf)
        ece += weight * gap

        bins.append({
            "avg_confidence": round(avg_conf, 4),
            "accuracy":       round(avg_acc, 4),
            "gap":            round(avg_acc - avg_conf, 4),
            "count":          count,
            "weight":         round(weight, 4),
            "bin_range":      (round(min(confs), 4), round(max(confs), 4)),
        })

    return {
        "ece":     round(ece, 6),
        "bins":    bins,
        "n_total": n,
    }


# ============================================================
# 2. ACCURACY METRICS — Exact Match + F1
# ============================================================

def normalize_answer(text: str) -> str:
    """Lowercase, strip whitespace/punctuation for matching."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def exact_match(predicted: str, gold: str) -> bool:
    """Case-insensitive exact match after normalization."""
    return normalize_answer(predicted) == normalize_answer(gold)


def token_f1(predicted: str, gold: str) -> float:
    """
    Token-level F1 between predicted and gold answers.
    Standard SQuAD-style F1.
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def compute_accuracy_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute EM and F1 across a list of predictions.

    Each prediction dict must have:
        - "predicted_answer": str
        - "gold_answer": str

    Returns dict with em, f1, and count.
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "count": 0}

    ems = []
    f1s = []
    for pred in predictions:
        predicted = pred["predicted_answer"]
        gold      = pred["gold_answer"]
        ems.append(1.0 if exact_match(predicted, gold) else 0.0)
        f1s.append(token_f1(predicted, gold))

    return {
        "em":    round(np.mean(ems), 4),
        "f1":    round(np.mean(f1s), 4),
        "count": len(predictions),
    }


# ============================================================
# 3. VOLATILITY-SPLIT BREAKDOWNS
# ============================================================

def volatility_breakdown(predictions: List[Dict]) -> Dict:
    """
    Compute ECE and accuracy metrics broken down by volatility class.

    Each prediction dict must have:
        - "volatility": "fast" | "slow" | "immutable"
        - "predicted_hedge", "correct" (for ECE)
        - "predicted_answer", "gold_answer" (for accuracy)

    Returns dict keyed by volatility class, each with ece + accuracy metrics.
    """
    by_vol = defaultdict(list)
    for pred in predictions:
        vol = pred.get("volatility", "unknown")
        by_vol[vol].append(pred)

    results = {}
    for vol, preds in sorted(by_vol.items()):
        results[vol] = {
            "count":    len(preds),
            "ece":      compute_ece(preds),
            "accuracy": compute_accuracy_metrics(preds),
        }

    return results


# ============================================================
# 4. BONFERRONI CORRECTION FRAMEWORK
# ============================================================

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.
    Uses pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if s_pooled == 0:
        return 0.0

    return round((m1 - m2) / s_pooled, 4)


def paired_t_test(group1: List[float], group2: List[float]) -> float:
    """
    Two-sample t-test for ECE comparison.
    Returns p-value.
    """
    from scipy import stats
    if len(group1) < 2 or len(group2) < 2:
        return 1.0
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return round(p_val, 6)


def bonferroni_ece_comparison(
    tsct_ece_scores: List[float],
    baseline_ece_scores: Dict[str, List[float]],
    alpha: float = 0.05,
    n_comparisons: int = 7,
) -> Dict:
    """
    Run 7 pairwise ECE comparisons between TSCT and each baseline,
    with Bonferroni correction.

    Args:
        tsct_ece_scores: list of ECE values from TSCT across random seeds
        baseline_ece_scores: dict mapping baseline name -> list of ECE values
            Expected baselines:
            1. Base LLM
            2. Self-consistency
            3. Temperature scaling
            4. Label smoothing SFT
            5. SFT only (no TCL)
            6. RAG
            7. Oracle

    Returns dict with per-comparison results including:
        - p_value (raw)
        - significant_bonferroni (bool, using corrected alpha)
        - cohens_d
        - practical_significance (bool, ECE reduction >= 0.01)
        - mean_tsct_ece
        - mean_baseline_ece
    """
    corrected_alpha = alpha / n_comparisons  # ~0.007

    results = {
        "alpha":           alpha,
        "n_comparisons":   n_comparisons,
        "corrected_alpha": round(corrected_alpha, 4),
        "comparisons":     {},
    }

    for name, baseline_scores in baseline_ece_scores.items():
        p_val = paired_t_test(tsct_ece_scores, baseline_scores)
        d     = cohens_d(baseline_scores, tsct_ece_scores)  # positive d = TSCT is lower (better)

        mean_tsct     = round(np.mean(tsct_ece_scores), 4)
        mean_baseline = round(np.mean(baseline_scores), 4)
        ece_reduction = round(mean_baseline - mean_tsct, 4)

        results["comparisons"][name] = {
            "mean_tsct_ece":           mean_tsct,
            "mean_baseline_ece":       mean_baseline,
            "ece_reduction":           ece_reduction,
            "p_value":                 p_val,
            "significant_bonferroni":  p_val < corrected_alpha,
            "cohens_d":                d,
            "practical_significance":  abs(ece_reduction) >= 0.01,
            "interpretation":          _interpret_comparison(
                                           p_val, corrected_alpha, ece_reduction, d
                                       ),
        }

    return results


def _interpret_comparison(p_val, alpha, ece_reduction, d):
    """Generate human-readable interpretation of a comparison."""
    if p_val >= alpha:
        return "Not statistically significant after Bonferroni correction."

    if abs(ece_reduction) < 0.01:
        return (
            f"Statistically significant (p={p_val:.4f}) but NOT practically "
            f"meaningful — ECE reduction of {ece_reduction:.4f} is below 0.01 "
            f"threshold. Do not claim this as a meaningful improvement."
        )

    size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    direction = "improvement" if ece_reduction > 0 else "regression"
    return (
        f"Statistically significant (p={p_val:.4f}) and practically meaningful "
        f"({size} effect, d={d:.2f}). ECE {direction} of {ece_reduction:.4f}."
    )


# ============================================================
# 5. VOLATILITY DISCRIMINATION SCORING
# ============================================================

def volatility_discrimination(predictions: List[Dict]) -> Dict:
    """
    Compare model's predicted volatility class against ground truth.
    Produces a confusion matrix and per-class precision/recall/F1.

    Each prediction dict must have:
        - "predicted_volatility": "fast" | "slow" | "immutable"
        - "true_volatility":     "fast" | "slow" | "immutable"
    """
    classes = ["immutable", "slow", "fast"]

    # Build confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for pred in predictions:
        true = pred["true_volatility"]
        predicted = pred["predicted_volatility"]
        matrix[true][predicted] += 1

    # Compute per-class metrics
    per_class = {}
    for cls in classes:
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in classes if other != cls)
        fn = sum(matrix[cls][other] for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }

    # Overall accuracy
    total   = sum(matrix[t][p] for t in classes for p in classes)
    correct = sum(matrix[c][c] for c in classes)
    accuracy = correct / total if total > 0 else 0.0

    # Format confusion matrix for display
    cm_display = {}
    for true_cls in classes:
        cm_display[true_cls] = {
            pred_cls: matrix[true_cls][pred_cls] for pred_cls in classes
        }

    return {
        "accuracy":         round(accuracy, 4),
        "per_class":        per_class,
        "confusion_matrix": cm_display,
        "total":            total,
    }


# ============================================================
# 6. TEMPORAL GENERALIZATION GAP
# ============================================================

def temporal_generalization_gap(
    predictions: List[Dict],
    cutoff_year: int = 2022,
    split_months: int = 12,
) -> Dict:
    """
    Measure performance change across time splits from training cutoff.
    Groups test examples into 12-month buckets by how far past cutoff
    the fact changed, then computes ECE and accuracy per bucket.

    Each prediction dict must have:
        - "change_year": int (year the fact changed)
        - "predicted_hedge", "correct" (for ECE)
        - "predicted_answer", "gold_answer" (for accuracy)

    Returns dict with per-bucket metrics showing how performance
    degrades (or doesn't) with distance from cutoff.
    """
    # Group by months-past-cutoff
    buckets = defaultdict(list)
    for pred in predictions:
        change_year = pred.get("change_year")
        if change_year is None:
            continue
        months_past = (int(change_year) - cutoff_year) * 12
        # Round to nearest split_months bucket
        bucket = (months_past // split_months) * split_months
        bucket_label = f"{bucket}-{bucket + split_months} months"
        buckets[bucket_label].append(pred)

    results = {}
    for bucket_label in sorted(buckets.keys(),
                                key=lambda x: int(x.split("-")[0])):
        preds = buckets[bucket_label]
        results[bucket_label] = {
            "count":    len(preds),
            "ece":      compute_ece(preds),
            "accuracy": compute_accuracy_metrics(preds),
        }

    # Compute generalization gap: difference between closest and farthest buckets
    if len(results) >= 2:
        bucket_keys = sorted(results.keys(),
                              key=lambda x: int(x.split("-")[0]))
        closest  = results[bucket_keys[0]]
        farthest = results[bucket_keys[-1]]

        gap = {
            "ece_gap":      round(farthest["ece"]["ece"] - closest["ece"]["ece"], 4),
            "em_gap":       round(farthest["accuracy"]["em"] - closest["accuracy"]["em"], 4),
            "f1_gap":       round(farthest["accuracy"]["f1"] - closest["accuracy"]["f1"], 4),
            "closest_bucket":  bucket_keys[0],
            "farthest_bucket": bucket_keys[-1],
        }
    else:
        gap = {"note": "Insufficient buckets for gap computation"}

    return {
        "cutoff_year":  cutoff_year,
        "split_months": split_months,
        "buckets":      results,
        "generalization_gap": gap,
    }


# ============================================================
# FULL EVALUATION RUNNER
# ============================================================

def run_full_evaluation(predictions: List[Dict], model_name: str = "model") -> Dict:
    """
    Run all evaluation metrics on a set of predictions.
    Returns a complete results dict suitable for the results table.

    Each prediction dict should have:
        - "predicted_answer": str
        - "gold_answer": str
        - "predicted_hedge": str (one of the 4 hedge tokens)
        - "correct": bool
        - "volatility": str
        - "change_year": int (optional, for temporal generalization)
    """
    return {
        "model": model_name,
        "overall": {
            "ece":      compute_ece(predictions),
            "accuracy": compute_accuracy_metrics(predictions),
        },
        "by_volatility":        volatility_breakdown(predictions),
        "temporal_generalization": temporal_generalization_gap(predictions),
    }


# ============================================================
# EXAMPLE USAGE / SMOKE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TSCT Evaluation Pipeline — Smoke Test")
    print("=" * 60)

    # Create synthetic test predictions
    np.random.seed(42)
    test_preds = []
    for i in range(200):
        vol = np.random.choice(["fast", "slow", "immutable"], p=[0.6, 0.3, 0.1])

        if vol == "fast":
            hedge = np.random.choice(
                ["[TEMPORAL_HEDGE]", "[UNKNOWN]", "[CONFIDENT]"],
                p=[0.7, 0.2, 0.1]
            )
            correct = np.random.random() < 0.3
        elif vol == "slow":
            hedge = np.random.choice(
                ["[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[CONFIDENT]"],
                p=[0.6, 0.2, 0.2]
            )
            correct = np.random.random() < 0.6
        else:
            hedge = np.random.choice(
                ["[CONFIDENT]", "[COND_CONFIDENT]"],
                p=[0.8, 0.2]
            )
            correct = np.random.random() < 0.9

        test_preds.append({
            "predicted_answer": "Alice" if correct else "Bob",
            "gold_answer":      "Alice",
            "predicted_hedge":  hedge,
            "correct":          correct,
            "volatility":       vol,
            "change_year":      np.random.choice([2022, 2023, 2024]),
            "predicted_volatility": vol if np.random.random() < 0.7 else "slow",
            "true_volatility":  vol,
        })

    # 1. ECE
    ece_result = compute_ece(test_preds)
    print(f"\n1. ECE: {ece_result['ece']:.4f}")
    print(f"   Bins: {len(ece_result['bins'])}")
    for b in ece_result["bins"]:
        print(f"     conf={b['avg_confidence']:.2f}  acc={b['accuracy']:.2f}  "
              f"gap={b['gap']:+.2f}  n={b['count']}")

    # 2. Accuracy
    acc = compute_accuracy_metrics(test_preds)
    print(f"\n2. Accuracy: EM={acc['em']:.4f}  F1={acc['f1']:.4f}")

    # 3. Volatility breakdown
    vb = volatility_breakdown(test_preds)
    print(f"\n3. Volatility breakdown:")
    for vol, metrics in vb.items():
        print(f"   {vol:10s}: ECE={metrics['ece']['ece']:.4f}  "
              f"EM={metrics['accuracy']['em']:.4f}  n={metrics['count']}")

    # 4. Bonferroni comparison (synthetic)
    tsct_scores = [0.08, 0.09, 0.07]  # 3 seeds
    baselines = {
        "Base LLM":          [0.22, 0.24, 0.21],
        "Self-consistency":  [0.18, 0.19, 0.17],
        "Temperature scaling":[0.15, 0.16, 0.14],
        "Label smoothing":   [0.13, 0.14, 0.12],
        "SFT only":          [0.11, 0.12, 0.10],
        "RAG":               [0.12, 0.13, 0.11],
        "Oracle":            [0.02, 0.02, 0.03],
    }
    bonf = bonferroni_ece_comparison(tsct_scores, baselines)
    print(f"\n4. Bonferroni ECE comparisons (corrected alpha={bonf['corrected_alpha']}):")
    for name, comp in bonf["comparisons"].items():
        sig = "***" if comp["significant_bonferroni"] else "   "
        prac = "P" if comp["practical_significance"] else " "
        print(f"   {sig}{prac} {name:25s}: TSCT={comp['mean_tsct_ece']:.3f} vs "
              f"{comp['mean_baseline_ece']:.3f}  "
              f"d={comp['cohens_d']:+.2f}  p={comp['p_value']:.4f}")

    # 5. Volatility discrimination
    vd = volatility_discrimination(test_preds)
    print(f"\n5. Volatility discrimination (accuracy={vd['accuracy']:.4f}):")
    for cls, metrics in vd["per_class"].items():
        print(f"   {cls:10s}: P={metrics['precision']:.2f}  "
              f"R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  "
              f"n={metrics['support']}")
    print(f"   Confusion matrix:")
    for true_cls, row in vd["confusion_matrix"].items():
        print(f"     {true_cls:10s} -> {dict(row)}")

    # 6. Temporal generalization gap
    tg = temporal_generalization_gap(test_preds)
    print(f"\n6. Temporal generalization gap:")
    for bucket, metrics in tg["buckets"].items():
        print(f"   {bucket:20s}: ECE={metrics['ece']['ece']:.4f}  "
              f"EM={metrics['accuracy']['em']:.4f}  n={metrics['count']}")
    gap = tg["generalization_gap"]
    if "ece_gap" in gap:
        print(f"   Gap (closest -> farthest): ECE={gap['ece_gap']:+.4f}  "
              f"EM={gap['em_gap']:+.4f}")

    print(f"\n{'=' * 60}")
    print("All 6 pipeline components operational.")
    print("=" * 60)
