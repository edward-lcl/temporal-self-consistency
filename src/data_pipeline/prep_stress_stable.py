"""
Adversarial Stable Facts Stress Test
=====================================
Tests whether the model over-hedges on facts that should be confident.
A model that hedges EVERYTHING gets low ECE on volatile facts but fails this test.

Generates a stress set of immutable + slow-stable facts where the model
should emit [CONFIDENT] or [COND_CONFIDENT], never [TEMPORAL_HEDGE]/[UNKNOWN].

Output: stress_stable_facts.jsonl
"""
import json

stable_facts = [
    # Physical constants
    {"question": "What is the speed of light in vacuum?", "gold_answer": "299792458 metres per second", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the boiling point of water at standard atmospheric pressure?", "gold_answer": "100 degrees Celsius", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the freezing point of water at standard pressure?", "gold_answer": "0 degrees Celsius", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the atomic number of carbon?", "gold_answer": "6", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the atomic number of oxygen?", "gold_answer": "8", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the chemical formula of water?", "gold_answer": "H2O", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the chemical formula of carbon dioxide?", "gold_answer": "CO2", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "How many protons does a hydrogen atom have?", "gold_answer": "1", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the gravitational acceleration on Earth?", "gold_answer": "9.81 metres per second squared", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is absolute zero in Celsius?", "gold_answer": "-273.15 degrees Celsius", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},

    # Mathematical facts
    {"question": "What is the value of pi to two decimal places?", "gold_answer": "3.14", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "How many sides does a hexagon have?", "gold_answer": "6", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "How many degrees are in a triangle?", "gold_answer": "180", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the square root of 144?", "gold_answer": "12", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the value of e to two decimal places?", "gold_answer": "2.72", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},

    # Historical events (immutable past dates)
    {"question": "In what year did World War 2 end?", "gold_answer": "1945", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year did the Berlin Wall fall?", "gold_answer": "1989", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year did Apollo 11 land on the moon?", "gold_answer": "1969", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year was the Magna Carta signed?", "gold_answer": "1215", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year was the Declaration of Independence signed?", "gold_answer": "1776", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year did the Titanic sink?", "gold_answer": "1912", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year did the French Revolution begin?", "gold_answer": "1789", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},

    # Geographic / structural slow-changing facts
    {"question": "What is the capital of France?", "gold_answer": "Paris", "volatility": "slow", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the capital of Japan?", "gold_answer": "Tokyo", "volatility": "slow", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the capital of Italy?", "gold_answer": "Rome", "volatility": "slow", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the capital of Egypt?", "gold_answer": "Cairo", "volatility": "slow", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the capital of Brazil?", "gold_answer": "Brasilia", "volatility": "slow", "expected_hedge": "[CONFIDENT]"},
    {"question": "On which continent is Egypt located?", "gold_answer": "Africa", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the largest ocean on Earth?", "gold_answer": "Pacific", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the longest river in South America?", "gold_answer": "Amazon", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the tallest mountain on Earth above sea level?", "gold_answer": "Mount Everest", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},

    # Cultural/authorship (immutable)
    {"question": "Who wrote Romeo and Juliet?", "gold_answer": "William Shakespeare", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "Who painted the Mona Lisa?", "gold_answer": "Leonardo da Vinci", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "Who composed the Ninth Symphony?", "gold_answer": "Ludwig van Beethoven", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "Who wrote the novel 1984?", "gold_answer": "George Orwell", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "Who developed the theory of general relativity?", "gold_answer": "Albert Einstein", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "Who painted the Sistine Chapel ceiling?", "gold_answer": "Michelangelo", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},

    # Scientific knowledge (immutable)
    {"question": "What is the powerhouse of the cell?", "gold_answer": "Mitochondria", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "How many chromosomes does a typical human cell have?", "gold_answer": "46", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What gas do plants absorb from the atmosphere during photosynthesis?", "gold_answer": "Carbon dioxide", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the largest planet in the solar system?", "gold_answer": "Jupiter", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "How many moons does Mars have?", "gold_answer": "2", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the closest star to Earth?", "gold_answer": "The Sun", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the chemical symbol for gold?", "gold_answer": "Au", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the chemical symbol for silver?", "gold_answer": "Ag", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "What is the chemical symbol for sodium?", "gold_answer": "Na", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},

    # Long-stable country borders/founding
    {"question": "In what year was the United States founded?", "gold_answer": "1776", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year did the Soviet Union dissolve?", "gold_answer": "1991", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
    {"question": "In what year was the United Nations founded?", "gold_answer": "1945", "volatility": "immutable", "expected_hedge": "[CONFIDENT]"},
]

print(f"Total stable stress facts: {len(stable_facts)}")

with open("stress_stable_facts.jsonl", "w") as f:
    for r in stable_facts:
        f.write(json.dumps(r) + "\n")

print("Written to stress_stable_facts.jsonl")
print(f"\nGoal: model should emit [CONFIDENT] or [COND_CONFIDENT] for ALL of these.")
print(f"Any [TEMPORAL_HEDGE] or [UNKNOWN] = over-hedging failure mode.")
