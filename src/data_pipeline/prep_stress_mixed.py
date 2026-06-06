"""
Mixed-Paragraph Stress Test Generator
======================================
Constructs paragraphs containing BOTH stable and volatile claims.
Tests whether the model selectively hedges only the volatile facts
rather than blanket-hedging the entire paragraph or being uniformly confident.

Output: stress_mixed_paragraphs.jsonl

Input format:
{
    "passage": "...",
    "claims": [
        {"text": "...", "volatility": "fast/slow/immutable", "expected_hedge": "[TOKEN]"},
        ...
    ]
}
"""
import json
import random

random.seed(42)

# Stable claims (immutable / very slow) — model should be CONFIDENT
stable_claims = [
    "The speed of light in vacuum is approximately 299,792,458 metres per second.",
    "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
    "Mount Everest is the tallest mountain on Earth above sea level.",
    "The chemical formula for water is H2O.",
    "Carbon has an atomic number of 6.",
    "Shakespeare wrote the play Romeo and Juliet.",
    "Apollo 11 landed on the Moon in 1969.",
    "The Berlin Wall fell in 1989.",
    "The Amazon is the longest river in South America.",
    "Jupiter is the largest planet in the solar system.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "World War 2 ended in 1945.",
]

# Volatile claims (fast-changing, where model should HEDGE)
volatile_claims = [
    "The current CEO of Nike is John Donahoe.",
    "The current President of the United States is Joe Biden.",
    "The prime minister of the United Kingdom is Rishi Sunak.",
    "The CEO of OpenAI is Sam Altman.",
    "Apple's CEO is Tim Cook.",
    "Microsoft is led by CEO Satya Nadella.",
    "The Secretary-General of the United Nations is António Guterres.",
    "France's president is Emmanuel Macron.",
    "Germany's chancellor is Olaf Scholz.",
    "The chair of the Federal Reserve is Jerome Powell.",
]

# Templates for combining claims into mixed paragraphs
def make_mixed_paragraph(stable, volatile, n_stable=2, n_volatile=2):
    chosen_stable = random.sample(stable, n_stable)
    chosen_volatile = random.sample(volatile, n_volatile)

    all_claims = []
    for c in chosen_stable:
        all_claims.append({
            "text": c,
            "volatility": "immutable" if "world war" in c.lower() or "wrote" in c.lower() or "atomic" in c.lower() else "slow",
            "expected_hedge": "[CONFIDENT]",
        })
    for c in chosen_volatile:
        all_claims.append({
            "text": c,
            "volatility": "fast",
            "expected_hedge": "[TEMPORAL_HEDGE]",
        })

    random.shuffle(all_claims)
    passage = " ".join(c["text"] for c in all_claims)

    return {
        "passage": passage,
        "claims": all_claims,
        "n_stable": n_stable,
        "n_volatile": n_volatile,
    }

paragraphs = []
for _ in range(40):  # 40 mixed paragraphs
    n_s = random.choice([1, 2, 3])
    n_v = random.choice([1, 2, 3])
    paragraphs.append(make_mixed_paragraph(stable_claims, volatile_claims, n_s, n_v))

# Also include some all-stable and all-volatile control paragraphs
for _ in range(10):
    p = make_mixed_paragraph(stable_claims, volatile_claims, 4, 0)
    p["type"] = "all_stable_control"
    paragraphs.append(p)
for _ in range(10):
    p = make_mixed_paragraph(stable_claims, volatile_claims, 0, 4)
    p["type"] = "all_volatile_control"
    paragraphs.append(p)

print(f"Generated {len(paragraphs)} mixed-paragraph stress test cases")
print(f"  Mixed:              40")
print(f"  All-stable control: 10")
print(f"  All-volatile ctrl:  10")

with open("stress_mixed_paragraphs.jsonl", "w") as f:
    for p in paragraphs:
        f.write(json.dumps(p) + "\n")

print("\nWritten to stress_mixed_paragraphs.jsonl")
print("\nEvaluation goal: model should hedge SELECTIVELY — emit [CONFIDENT] for")
print("stable claims in the paragraph and [TEMPORAL_HEDGE] for volatile ones,")
print("not blanket-hedge everything or blanket-confident-everything.")
