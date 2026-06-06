"""
Week 3 Full Results Analysis
Runs all 24 prediction files through the eval pipeline.
"""
import os
import json
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pipeline import (
    compute_ece, compute_accuracy_metrics, volatility_breakdown,
    bonferroni_ece_comparison, run_full_evaluation
)

UPLOADS = os.environ.get("TSCT_PREDICTIONS_DIR", "./predictions")
BENCHMARKS = ["temporal_delta", "mmlu", "freshqa", "stress_test"]
CONDITIONS = ["sft", "tsct"]
SEEDS = [42, 123, 456]

def load(condition, seed, benchmark):
    exp = "exp2_sft" if condition == "sft" else "exp3_tsct"
    fname = f"{exp}_seed{seed}_{benchmark}_predictions.jsonl"
    with open(f"{UPLOADS}/{fname}") as f:
        return [json.loads(l) for l in f]

def normalize_volatility(preds):
    """Convert 'fast-changing' etc to 'fast' to match pipeline."""
    vol_map = {
        "fast-changing": "fast",
        "slow-changing": "slow",
        "never-changing": "immutable",
        "fast": "fast", "slow": "slow", "immutable": "immutable",
    }
    for p in preds:
        v = p.get("volatility", "unknown")
        p["volatility"] = vol_map.get(v, v)
    return preds

# ============================================================
# OVERALL RESULTS TABLE
# ============================================================
print("=" * 80)
print("FULL RESULTS — All conditions × All benchmarks")
print("=" * 80)
print(f"\n{'Condition':<25} {'Benchmark':<18} {'ECE':>8} {'EM':>8} {'F1':>8} {'N':>6}")
print("-" * 80)

all_results = defaultdict(dict)

for condition in CONDITIONS:
    for seed in SEEDS:
        for benchmark in BENCHMARKS:
            preds = normalize_volatility(load(condition, seed, benchmark))
            ece = compute_ece(preds)
            acc = compute_accuracy_metrics(preds)
            all_results[f"{condition}_seed{seed}"][benchmark] = {
                "ece": ece["ece"],
                "em": acc["em"],
                "f1": acc["f1"],
                "n": acc["count"],
                "predictions": preds,
            }
            print(f"{condition.upper()} seed {seed:<13} {benchmark:<18} "
                  f"{ece['ece']:>8.4f} {acc['em']:>8.4f} {acc['f1']:>8.4f} {acc['count']:>6}")
    print("-" * 80)

# ============================================================
# AGGREGATE ACROSS SEEDS — mean ± std per condition
# ============================================================
print("\n" + "=" * 80)
print("AGGREGATE METRICS (mean ± std across 3 seeds)")
print("=" * 80)

agg = {}
for condition in CONDITIONS:
    agg[condition] = {}
    for benchmark in BENCHMARKS:
        eces = [all_results[f"{condition}_seed{s}"][benchmark]["ece"] for s in SEEDS]
        ems  = [all_results[f"{condition}_seed{s}"][benchmark]["em"] for s in SEEDS]
        f1s  = [all_results[f"{condition}_seed{s}"][benchmark]["f1"] for s in SEEDS]
        agg[condition][benchmark] = {
            "ece_mean": np.mean(eces), "ece_std": np.std(eces),
            "em_mean":  np.mean(ems),  "em_std":  np.std(ems),
            "f1_mean":  np.mean(f1s),  "f1_std":  np.std(f1s),
            "ece_scores": eces,
        }

print(f"\n{'Benchmark':<18} {'Metric':<8} {'SFT':<22} {'TSCT':<22} {'Δ (TSCT-SFT)':<15}")
print("-" * 90)
for benchmark in BENCHMARKS:
    for metric in ["ece", "em", "f1"]:
        sft_m  = agg["sft"][benchmark][f"{metric}_mean"]
        sft_s  = agg["sft"][benchmark][f"{metric}_std"]
        tsct_m = agg["tsct"][benchmark][f"{metric}_mean"]
        tsct_s = agg["tsct"][benchmark][f"{metric}_std"]
        delta  = tsct_m - sft_m
        direction = "↓" if (metric == "ece" and delta < 0) or (metric != "ece" and delta > 0) else "↑"
        good = (metric == "ece" and delta < 0) or (metric != "ece" and delta > 0)
        marker = "✓" if good else " "
        print(f"{benchmark:<18} {metric.upper():<8} "
              f"{sft_m:.4f} ± {sft_s:.4f}     "
              f"{tsct_m:.4f} ± {tsct_s:.4f}     "
              f"{delta:+.4f} {marker}")
    print()

# ============================================================
# VOLATILITY BREAKDOWN (temporal_delta benchmark — primary)
# ============================================================
print("=" * 80)
print("VOLATILITY BREAKDOWN — temporal_delta test set")
print("=" * 80)

for condition in CONDITIONS:
    print(f"\n{condition.upper()} (averaged across 3 seeds):")
    by_vol = defaultdict(lambda: defaultdict(list))
    for seed in SEEDS:
        preds = all_results[f"{condition}_seed{seed}"]["temporal_delta"]["predictions"]
        vb = volatility_breakdown(preds)
        for vol, metrics in vb.items():
            by_vol[vol]["ece"].append(metrics["ece"]["ece"])
            by_vol[vol]["em"].append(metrics["accuracy"]["em"])
            by_vol[vol]["count"].append(metrics["count"])

    print(f"  {'Class':<12} {'ECE':>10} {'EM':>10} {'Count':>8}")
    for vol in sorted(by_vol.keys()):
        m = by_vol[vol]
        print(f"  {vol:<12} {np.mean(m['ece']):>10.4f} {np.mean(m['em']):>10.4f} "
              f"{int(np.mean(m['count'])):>8}")

# ============================================================
# BONFERRONI: TSCT vs SFT comparison
# ============================================================
print("\n" + "=" * 80)
print("BONFERRONI-CORRECTED COMPARISON — TSCT vs SFT")
print("(α = 0.05, corrected α ≈ 0.007 across 7 comparisons)")
print("=" * 80)

# For now we only have SFT and TSCT — full 7-comparison needs baselines from Logan
# But we can show the TSCT vs SFT primary result for each benchmark
print(f"\n{'Benchmark':<18} {'SFT ECE':>10} {'TSCT ECE':>12} {'Δ':>8} {'Cohen d':>10}")
print("-" * 70)
for benchmark in BENCHMARKS:
    sft_ece  = agg["sft"][benchmark]["ece_scores"]
    tsct_ece = agg["tsct"][benchmark]["ece_scores"]

    sft_mean  = np.mean(sft_ece)
    tsct_mean = np.mean(tsct_ece)
    delta = tsct_mean - sft_mean

    pooled = np.sqrt((np.var(sft_ece, ddof=1) + np.var(tsct_ece, ddof=1)) / 2)
    d = (sft_mean - tsct_mean) / pooled if pooled > 0 else 0

    print(f"{benchmark:<18} {sft_mean:>10.4f} {tsct_mean:>12.4f} "
          f"{delta:>+8.4f} {d:>+10.2f}")

# ============================================================
# HEDGE DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("HEDGE TOKEN DISTRIBUTION — temporal_delta")
print("=" * 80)
from collections import Counter

for condition in CONDITIONS:
    print(f"\n{condition.upper()} (averaged hedge usage across 3 seeds):")
    all_hedges = Counter()
    total = 0
    for seed in SEEDS:
        preds = all_results[f"{condition}_seed{seed}"]["temporal_delta"]["predictions"]
        for p in preds:
            all_hedges[p["predicted_hedge"]] += 1
            total += 1

    for hedge, count in sorted(all_hedges.items()):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {hedge:<22} {count:>6} ({pct:>5.1f}%)  {bar}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
