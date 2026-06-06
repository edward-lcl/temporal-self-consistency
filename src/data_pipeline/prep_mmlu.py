"""
Prepare MMLU stable subset for regression-check evaluation.

Extracts time-stable MMLU subjects (math, physics, chemistry, etc.) where
answers do not change over time. TSCT must not degrade accuracy on these.

Usage:
    python prep_mmlu.py
Output:
    mmlu_stable_subset.jsonl
"""
from datasets import load_dataset
import json

# Time-stable subjects — answers don't change over time
STABLE_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_mathematics", "college_physics",
    "computer_science", "conceptual_physics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts",
    "high_school_biology", "high_school_chemistry", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "logical_fallacies",
    "machine_learning", "mathematical_logic", "philosophy",
    "prehistory", "world_religions",
]


def main():
    ds = load_dataset("cais/mmlu", "all")
    stable = []

    for row in ds["test"]:
        if row["subject"] in STABLE_SUBJECTS:
            choices = row["choices"]
            answer_idx = row["answer"]
            stable.append({
                "question":    row["question"],
                "choices":     choices,
                "gold_answer": choices[answer_idx],
                "subject":     row["subject"],
                "volatility":  "immutable",
            })

    with open("mmlu_stable_subset.jsonl", "w") as f:
        for r in stable:
            f.write(json.dumps(r) + "\n")

    print(f"MMLU stable subset: {len(stable)} questions "
          f"across {len(STABLE_SUBJECTS)} subjects")


if __name__ == "__main__":
    main()
