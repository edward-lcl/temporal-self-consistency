"""
TSCT Final Results Table Generator
====================================
Produces a publication-ready results table from all eval pipeline outputs.

Columns: ECE | EM | F1 | ECE (fast) | ECE (slow) | ECE (immutable)
Rows: 8 conditions in canonical order

Outputs:
  - results_table.md    (Markdown for paper)
  - results_table.tex   (LaTeX booktabs-style for paper)
  - results_table.csv   (CSV for analysis / spreadsheets)
"""
import json
import os
import sys
import numpy as np

CONDITIONS = [
    "Base LLM",
    "Self-consistency",
    "Temperature scaling",
    "Label smoothing",
    "RAG",
    "SFT only",
    "TSCT",
    "SFT+TCL+DPO",
    "Oracle",
]


def fmt(mean, std=None):
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def make_table(results):
    """results = dict of condition_name -> dict with metrics."""
    rows = []
    for cond in CONDITIONS:
        if cond not in results:
            rows.append({"name": cond, "missing": True})
            continue
        r = results[cond]
        rows.append({
            "name":         cond,
            "ece":          fmt(r["ece_mean"], r["ece_std"]),
            "em":           fmt(r["em_mean"], r["em_std"]),
            "f1":           fmt(r["f1_mean"], r["f1_std"]),
            "ece_fast":     fmt(r["ece_fast_mean"], r["ece_fast_std"]),
            "ece_slow":     fmt(r["ece_slow_mean"], r["ece_slow_std"]),
            "ece_immut":    fmt(r["ece_immut_mean"], r["ece_immut_std"]),
            "sig":          r.get("significant", ""),
        })
    return rows


def write_markdown(rows, output="results_table.md"):
    lines = [
        "| Condition | ECE ↓ | EM ↑ | F1 ↑ | ECE Fast ↓ | ECE Slow ↓ | ECE Immut ↓ | Sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['name']} | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['name']} | {r['ece']} | {r['em']} | {r['f1']} | "
            f"{r['ece_fast']} | {r['ece_slow']} | {r['ece_immut']} | {r['sig']} |"
        )
    with open(output, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {output}")


def write_latex(rows, output="results_table.tex"):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main results across 8 conditions. ECE: Expected Calibration Error "
        "(lower better). EM/F1: accuracy metrics (higher better). $\\dagger$ marks "
        "statistical significance after Bonferroni correction ($\\alpha=0.007$).}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Condition & ECE $\\downarrow$ & EM $\\uparrow$ & F1 $\\uparrow$ & "
        "ECE Fast & ECE Slow & ECE Immut \\\\",
        "\\midrule",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(f"{r['name']} & — & — & — & — & — & — \\\\")
            continue
        sig = "$^\\dagger$" if r["sig"] else ""
        lines.append(
            f"{r['name']}{sig} & {r['ece']} & {r['em']} & {r['f1']} & "
            f"{r['ece_fast']} & {r['ece_slow']} & {r['ece_immut']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(output, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {output}")


def write_csv(rows, output="results_table.csv"):
    import csv
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition", "ECE", "EM", "F1",
                    "ECE_Fast", "ECE_Slow", "ECE_Immutable", "Significant"])
        for r in rows:
            if r.get("missing"):
                w.writerow([r["name"]] + ["—"] * 7)
                continue
            w.writerow([r["name"], r["ece"], r["em"], r["f1"],
                        r["ece_fast"], r["ece_slow"], r["ece_immut"], r["sig"]])
    print(f"  ✓ {output}")


# Placeholder data structure showing what's expected
PLACEHOLDER_RESULTS = {
    "Base LLM": {
        "ece_mean": 0.223, "ece_std": 0.012, "em_mean": 0.182, "em_std": 0.008,
        "f1_mean": 0.245, "f1_std": 0.009, "ece_fast_mean": 0.31, "ece_fast_std": 0.01,
        "ece_slow_mean": 0.18, "ece_slow_std": 0.01, "ece_immut_mean": 0.05,
        "ece_immut_std": 0.008, "significant": "",
    },
    "SFT only": {
        "ece_mean": 0.112, "ece_std": 0.008, "em_mean": 0.224, "em_std": 0.011,
        "f1_mean": 0.291, "f1_std": 0.013, "ece_fast_mean": 0.15, "ece_fast_std": 0.01,
        "ece_slow_mean": 0.09, "ece_slow_std": 0.008, "ece_immut_mean": 0.04,
        "ece_immut_std": 0.005, "significant": "",
    },
    "TSCT": {
        "ece_mean": 0.082, "ece_std": 0.006, "em_mean": 0.228, "em_std": 0.013,
        "f1_mean": 0.302, "f1_std": 0.012, "ece_fast_mean": 0.10, "ece_fast_std": 0.008,
        "ece_slow_mean": 0.07, "ece_slow_std": 0.006, "ece_immut_mean": 0.04,
        "ece_immut_std": 0.005, "significant": "†",
    },
    "Oracle": {
        "ece_mean": 0.023, "ece_std": 0.005, "em_mean": 0.180, "em_std": 0.008,
        "f1_mean": 0.245, "f1_std": 0.009, "ece_fast_mean": 0.02, "ece_fast_std": 0.003,
        "ece_slow_mean": 0.02, "ece_slow_std": 0.003, "ece_immut_mean": 0.01,
        "ece_immut_std": 0.002, "significant": "",
    },
}


if __name__ == "__main__":
    if os.path.exists("results.json"):
        with open("results.json") as f:
            results = json.load(f)
        print("Using real results.json")
    else:
        results = PLACEHOLDER_RESULTS
        print("Using placeholder data — substitute real results when eval completes.")

    rows = make_table(results)
    print("\nGenerating tables...")
    write_markdown(rows)
    write_latex(rows)
    write_csv(rows)
    print("\nTo use real data: drop a results.json with the same structure as PLACEHOLDER_RESULTS.")
