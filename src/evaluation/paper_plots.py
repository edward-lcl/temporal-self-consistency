"""
TSCT Paper Plots - Complete Set
================================
All figures needed for the paper. Currently uses placeholder data;
swap in real numbers from eval pipeline when available.

Plots produced:
1. Calibration curves (reliability diagrams) - per condition
2. ECE bar chart across all 8 conditions
3. Temporal generalization curve (ECE/EM vs months past cutoff)
4. Volatility confusion matrix (heatmap)
5. Hedge distribution by volatility class (grouped bar)
6. Per-benchmark results comparison

Output: figures/*.png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Style — clean, publication quality
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Consistent color scheme across all plots
COLORS = {
    "Base LLM":             "#888888",
    "Self-consistency":     "#A6611A",
    "Temperature scaling":  "#DFC27D",
    "Label smoothing":      "#80CDC1",
    "RAG":                  "#5AB4AC",
    "SFT only":             "#2C7BB6",
    "TSCT":                 "#D7191C",
    "SFT+TCL+DPO":          "#FF7F00",
    "Oracle":               "#000000",
}

os.makedirs("figures", exist_ok=True)


# ==========================================================
# Helper: load real results if present, else placeholder
# ==========================================================
def load_or_placeholder():
    """Load real eval results if available, else return placeholder."""
    if os.path.exists("results.json"):
        with open("results.json") as f:
            return json.load(f), True
    # Placeholder structure mimics what eval pipeline produces
    return {
        "Base LLM":             {"ece_per_seed": [0.22, 0.24, 0.21], "em": 0.18, "f1": 0.24},
        "Self-consistency":     {"ece_per_seed": [0.18, 0.19, 0.17], "em": 0.21, "f1": 0.28},
        "Temperature scaling":  {"ece_per_seed": [0.15, 0.16, 0.14], "em": 0.18, "f1": 0.24},
        "Label smoothing":      {"ece_per_seed": [0.13, 0.14, 0.12], "em": 0.19, "f1": 0.26},
        "RAG":                  {"ece_per_seed": [0.12, 0.13, 0.11], "em": 0.34, "f1": 0.41},
        "SFT only":             {"ece_per_seed": [0.11, 0.12, 0.10], "em": 0.22, "f1": 0.29},
        "TSCT":                 {"ece_per_seed": [0.08, 0.09, 0.07], "em": 0.23, "f1": 0.30},
        "SFT+TCL+DPO":          {"ece_per_seed": [0.07, 0.08, 0.06], "em": 0.24, "f1": 0.31},
        "Oracle":               {"ece_per_seed": [0.02, 0.02, 0.03], "em": 0.18, "f1": 0.24},
    }, False


# ==========================================================
# 1. CALIBRATION CURVES (reliability diagram)
# ==========================================================
def plot_calibration_curves(bin_data=None):
    """
    Each line = one model condition.
    X = mean predicted confidence in bin
    Y = empirical accuracy in bin
    Perfect calibration = diagonal.
    """
    if bin_data is None:
        # Placeholder: simulate well-calibrated TSCT vs overconfident base
        bin_data = {
            "Base LLM":  {"conf": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                          "acc":  [0.08, 0.18, 0.32, 0.45, 0.55, 0.62]},
            "SFT only":  {"conf": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                          "acc":  [0.10, 0.25, 0.42, 0.58, 0.72, 0.80]},
            "TSCT":      {"conf": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                          "acc":  [0.12, 0.30, 0.48, 0.68, 0.85, 0.92]},
        }

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")

    for name, data in bin_data.items():
        color = COLORS.get(name, "#444444")
        ax.plot(data["conf"], data["acc"], "o-", lw=2, markersize=7,
                color=color, label=name)

    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration curves on temporal_delta test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/calibration_curves.png")
    plt.close()
    print("  ✓ figures/calibration_curves.png")


# ==========================================================
# 2. ECE BAR CHART — all 8 conditions
# ==========================================================
def plot_ece_bars(results=None):
    if results is None:
        results, _ = load_or_placeholder()

    # Order matters for storytelling: baselines → ours → oracle
    order = ["Base LLM", "Self-consistency", "Temperature scaling",
             "Label smoothing", "RAG", "SFT only",
             "TSCT", "SFT+TCL+DPO", "Oracle"]

    names = [n for n in order if n in results]
    means = [np.mean(results[n]["ece_per_seed"]) for n in names]
    stds  = [np.std(results[n]["ece_per_seed"])  for n in names]
    colors = [COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), means, yerr=stds,
                  color=colors, capsize=4, edgecolor="black", linewidth=0.5)

    # Highlight TSCT
    for i, name in enumerate(names):
        if name == "TSCT":
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("ECE on volatile-fact subset — lower is better")
    ax.axhline(y=results.get("Oracle", {"ece_per_seed": [0]})["ece_per_seed"][0],
               color="gray", linestyle=":", lw=1, label="Oracle (ceiling)")

    # Annotate bars with values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/ece_bar_chart.png")
    plt.close()
    print("  ✓ figures/ece_bar_chart.png")


# ==========================================================
# 3. TEMPORAL GENERALIZATION CURVE
# ==========================================================
def plot_temporal_generalization(curves=None):
    """X = months past training cutoff; Y = ECE per condition."""
    if curves is None:
        months = [0, 6, 12, 18, 24, 30, 36]
        curves = {
            "Base LLM":  [0.20, 0.22, 0.25, 0.30, 0.36, 0.42, 0.49],
            "SFT only":  [0.10, 0.12, 0.15, 0.18, 0.23, 0.28, 0.34],
            "TSCT":      [0.08, 0.09, 0.10, 0.12, 0.14, 0.17, 0.20],
        }
    else:
        months = curves.pop("months")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, values in curves.items():
        color = COLORS.get(name, "#444444")
        ax.plot(months, values, "o-", lw=2, markersize=7, color=color, label=name)

    ax.set_xlabel("Months past training cutoff")
    ax.set_ylabel("ECE")
    ax.set_title("Temporal generalization — degradation with horizon")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xticks(months)
    plt.tight_layout()
    plt.savefig("figures/temporal_generalization.png")
    plt.close()
    print("  ✓ figures/temporal_generalization.png")


# ==========================================================
# 4. VOLATILITY CONFUSION MATRIX
# ==========================================================
def plot_volatility_confusion(matrix=None):
    """Heatmap of predicted vs true volatility class."""
    classes = ["immutable", "slow", "fast"]
    if matrix is None:
        # Placeholder — TSCT's volatility predictions
        matrix = np.array([
            [180,  20,   0],   # true: immutable
            [ 15, 250,  35],   # true: slow
            [  5,  40, 455],   # true: fast
        ])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Volatility discrimination — TSCT")

    # Annotate cells with counts + percentages
    for i in range(len(classes)):
        row_sum = matrix[i].sum()
        for j in range(len(classes)):
            n = matrix[i][j]
            pct = 100 * n / row_sum if row_sum else 0
            color = "white" if n > matrix.max() / 2 else "black"
            ax.text(j, i, f"{n}\n({pct:.0f}%)", ha="center", va="center",
                    color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("figures/volatility_confusion.png")
    plt.close()
    print("  ✓ figures/volatility_confusion.png")


# ==========================================================
# 5. HEDGE DISTRIBUTION BY VOLATILITY
# ==========================================================
def plot_hedge_distribution(dist=None):
    """Stacked bar: for each volatility class, what hedge tokens did the model emit?"""
    if dist is None:
        # Placeholder TSCT distribution
        dist = {
            "immutable": {"[CONFIDENT]": 165, "[COND_CONFIDENT]": 25,
                          "[TEMPORAL_HEDGE]": 8, "[UNKNOWN]": 2},
            "slow":      {"[CONFIDENT]": 80,  "[COND_CONFIDENT]": 180,
                          "[TEMPORAL_HEDGE]": 35, "[UNKNOWN]": 5},
            "fast":      {"[CONFIDENT]": 50,  "[COND_CONFIDENT]": 90,
                          "[TEMPORAL_HEDGE]": 280, "[UNKNOWN]": 80},
        }

    hedge_order = ["[CONFIDENT]", "[COND_CONFIDENT]",
                   "[TEMPORAL_HEDGE]", "[UNKNOWN]"]
    hedge_colors = ["#1B7837", "#7FBC41", "#DE77AE", "#8E0152"]

    vols = list(dist.keys())
    x = np.arange(len(vols))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(vols))

    for hedge, color in zip(hedge_order, hedge_colors):
        counts = np.array([dist[v].get(hedge, 0) for v in vols])
        # Convert to %
        totals = np.array([sum(dist[v].values()) for v in vols])
        pcts = 100 * counts / totals
        ax.bar(x, pcts, width, bottom=bottoms, label=hedge, color=color)
        bottoms += pcts

    ax.set_xticks(x)
    ax.set_xticklabels(vols)
    ax.set_xlabel("True volatility class")
    ax.set_ylabel("% of model outputs")
    ax.set_title("TSCT hedge token distribution by fact volatility")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig("figures/hedge_distribution.png")
    plt.close()
    print("  ✓ figures/hedge_distribution.png")


# ==========================================================
# 6. PER-BENCHMARK COMPARISON (grouped bar)
# ==========================================================
def plot_benchmark_comparison(per_bench=None):
    """ECE per benchmark per condition - grouped bars."""
    if per_bench is None:
        per_bench = {
            "TemporalDelta": {"Base LLM": 0.22, "SFT only": 0.11, "TSCT": 0.08},
            "FreshQA":       {"Base LLM": 0.25, "SFT only": 0.14, "TSCT": 0.10},
            "TLQA":          {"Base LLM": 0.20, "SFT only": 0.13, "TSCT": 0.09},
            "TDBench":       {"Base LLM": 0.23, "SFT only": 0.12, "TSCT": 0.09},
            "MMLU stable":   {"Base LLM": 0.05, "SFT only": 0.06, "TSCT": 0.05},
        }

    conditions = list(next(iter(per_bench.values())).keys())
    benchmarks = list(per_bench.keys())
    x = np.arange(len(benchmarks))
    width = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conditions):
        vals = [per_bench[b][cond] for b in benchmarks]
        offset = (i - len(conditions) / 2 + 0.5) * width
        color = COLORS.get(cond, "#444444")
        ax.bar(x + offset, vals, width, color=color, label=cond,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("ECE")
    ax.set_title("ECE across benchmarks")
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig("figures/benchmark_comparison.png")
    plt.close()
    print("  ✓ figures/benchmark_comparison.png")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("Generating all paper figures...\n")
    plot_calibration_curves()
    plot_ece_bars()
    plot_temporal_generalization()
    plot_volatility_confusion()
    plot_hedge_distribution()
    plot_benchmark_comparison()
    print("\nAll figures saved to figures/")
    print("Currently using placeholder data. When eval results are ready, pass")
    print("real data dicts into each function or save results.json.")
