"""
Prepare extended-horizon stress test (18-36 months post-cutoff).

Pulls facts from the master triples file that changed in 2024-2025,
well after the LLaMA-3 8B March 2023 cutoff. These are the hardest
test cases — the model cannot have seen these answers in pretraining.

Usage:
    python prep_stress_horizon.py
Requires:
    all_triples.jsonl  (master TemporalDelta file in working dir)
Output:
    stress_test_extended_horizon.jsonl
"""
import json


def main():
    stress = []
    with open("all_triples.jsonl") as f:
        for line in f:
            r = json.loads(line)
            t_end = r.get("t_end")
            if t_end and t_end in ("2024", "2025"):
                r["months_post_cutoff"] = (int(t_end) - 2023) * 12
                stress.append(r)

    with open("stress_test_extended_horizon.jsonl", "w") as f:
        for r in stress:
            f.write(json.dumps(r) + "\n")

    print(f"Stress test set: {len(stress)} facts (18-36 months post-cutoff)")


if __name__ == "__main__":
    main()
