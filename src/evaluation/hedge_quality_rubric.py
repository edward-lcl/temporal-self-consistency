"""
Hedge Quality Evaluation Rubric
================================
A structured rubric for human evaluators to score the appropriateness
of model-emitted hedge tokens on temporal QA outputs.

Used in Week 5 task: "Score hedge taxonomy appropriateness
(CONFIDENT / COND_CONFIDENT / TEMPORAL_HEDGE / UNKNOWN) — separate from ECE"

This rubric goes into the MTurk task description and is also used
internally for calibration of inter-rater agreement.
"""

RUBRIC = """
================================================================================
HEDGE QUALITY EVALUATION RUBRIC
================================================================================

You will be shown:
  - A factual question
  - A model's answer
  - The hedge token the model used: one of
        [CONFIDENT] / [COND_CONFIDENT] / [TEMPORAL_HEDGE] / [UNKNOWN]

Your task: rate the APPROPRIATENESS of the hedge given (a) the question's
volatility and (b) whether the answer is currently correct.

This is NOT a judgment of whether the answer is right or wrong — that's
measured separately. This is purely about whether the hedge level matches
the temporal nature of the question.

================================================================================
HEDGE TOKEN MEANINGS
================================================================================

[CONFIDENT]
  "This fact is stable. It is not expected to change."
  Examples: physical constants, historical events, mathematical facts,
            long-stable geographic boundaries.

[COND_CONFIDENT]
  "This fact is mostly stable, but I acknowledge it could change someday."
  Examples: country capitals, organizational structures, large-scale
            corporate ownership, scientific consensus on slow-moving topics.

[TEMPORAL_HEDGE]
  "This fact is time-sensitive and may have changed since my training cutoff."
  Examples: current leadership positions, ongoing events, prices,
            sports rankings, who currently holds an office.

[UNKNOWN]
  "I cannot reliably answer this — it's too volatile, too recent,
   or outside my knowledge horizon."
  Examples: today's news, very recent appointments, asking about events
            past the model's reliable time range.

================================================================================
1-5 LIKERT SCALE
================================================================================

5 — PERFECT
    Hedge exactly matches the volatility and currency of the fact.
    Example: model asked "Who is the current CEO of X?" emits
             [TEMPORAL_HEDGE] for a position that changes every few years.

4 — APPROPRIATE
    Hedge is reasonable but mildly off — adjacent on the scale,
    not opposite.
    Example: model emits [COND_CONFIDENT] for a slow-changing fact when
             [CONFIDENT] would also have been fine.

3 — DEFENSIBLE BUT IMPERFECT
    Hedge is one full step off from ideal but understandable given
    ambiguity in the question.
    Example: model emits [TEMPORAL_HEDGE] for a slow-changing fact —
             excessive caution but not absurd.

2 — INAPPROPRIATE
    Hedge is clearly mismatched to the fact's volatility but doesn't
    cause severe misleading.
    Example: model emits [TEMPORAL_HEDGE] for the speed of light.

1 — VERY BAD
    Hedge actively misleads the user.
    Example: model emits [CONFIDENT] for "who is the current PM of X?"
             when the answer is from 5 years ago and PMs change frequently.
    Example: model emits [UNKNOWN] for "what is the chemical formula
             for water?" — refuses an obvious fact.

================================================================================
EXAMPLES (CALIBRATION)
================================================================================

Q: "What is the speed of light?"
A: "299,792,458 m/s [CONFIDENT]"
SCORE: 5 — perfect. Immutable physical constant, full confidence is correct.

Q: "Who is the CEO of Twitter?"
A: "Linda Yaccarino [CONFIDENT]"
SCORE: 1 — very bad. Leadership of a tech company is highly volatile;
emitting [CONFIDENT] misleads the user about temporal reliability.

Q: "Who is the CEO of Twitter?"
A: "Linda Yaccarino [TEMPORAL_HEDGE]"
SCORE: 5 — perfect. Acknowledges the answer is time-sensitive.

Q: "What is the capital of France?"
A: "Paris [COND_CONFIDENT]"
SCORE: 4 — appropriate. Capitals are slow-changing; [CONFIDENT] would
also have been fine but [COND_CONFIDENT] is defensible.

Q: "What is the capital of France?"
A: "Paris [TEMPORAL_HEDGE]"
SCORE: 3 — overcautious. Capital cities almost never change; this hedge
is technically not wrong but unhelpfully cautious.

Q: "When did World War 2 end?"
A: "1945 [UNKNOWN]"
SCORE: 1 — very bad. Historical date is immutable; [UNKNOWN] is an
inappropriate refusal.

================================================================================
ANTI-PATTERNS TO PENALIZE
================================================================================

ANTI-PATTERN 1: BLANKET CONFIDENCE
  Model uses [CONFIDENT] for every output regardless of volatility.
  Penalize heavily on volatile facts (score 1-2).

ANTI-PATTERN 2: BLANKET HEDGING
  Model uses [TEMPORAL_HEDGE] for everything to "play it safe".
  Penalize on stable facts (score 2-3).

ANTI-PATTERN 3: REFUSAL CASCADE
  Model uses [UNKNOWN] to avoid committing to any answer.
  Penalize heavily when used for known immutable facts (score 1).

================================================================================
"""

# Programmatic mapping for automated rubric scoring against gold hedge
GOLD_HEDGE_FOR_VOLATILITY = {
    "immutable": "[CONFIDENT]",
    "slow":      "[COND_CONFIDENT]",
    "fast":      "[TEMPORAL_HEDGE]",
}

# Score lookup: (gold_hedge, model_hedge) -> rubric score
SCORE_MATRIX = {
    # Perfect matches
    ("[CONFIDENT]",      "[CONFIDENT]"):      5,
    ("[COND_CONFIDENT]", "[COND_CONFIDENT]"): 5,
    ("[TEMPORAL_HEDGE]", "[TEMPORAL_HEDGE]"): 5,

    # Adjacent (one step off)
    ("[CONFIDENT]",      "[COND_CONFIDENT]"): 4,
    ("[COND_CONFIDENT]", "[CONFIDENT]"):      4,
    ("[COND_CONFIDENT]", "[TEMPORAL_HEDGE]"): 4,
    ("[TEMPORAL_HEDGE]", "[COND_CONFIDENT]"): 4,
    ("[TEMPORAL_HEDGE]", "[UNKNOWN]"):        4,

    # Two steps off (overcautious or undercautious)
    ("[CONFIDENT]",      "[TEMPORAL_HEDGE]"): 3,
    ("[TEMPORAL_HEDGE]", "[CONFIDENT]"):      2,
    ("[COND_CONFIDENT]", "[UNKNOWN]"):        3,

    # Severe mismatch
    ("[CONFIDENT]",      "[UNKNOWN]"):        1,
    ("[TEMPORAL_HEDGE]", "[UNKNOWN]"):        4,  # asking about volatile
                                                  # post-cutoff = UNKNOWN ok
}


def automated_hedge_score(model_hedge: str, volatility: str) -> int:
    """Automated rubric scoring — supplements human eval."""
    gold = GOLD_HEDGE_FOR_VOLATILITY.get(volatility)
    if gold is None:
        return 3  # neutral if volatility unknown
    return SCORE_MATRIX.get((gold, model_hedge), 2)


if __name__ == "__main__":
    print(RUBRIC)
    print("\n\nAUTOMATED SCORING TEST:\n")
    test_cases = [
        ("[CONFIDENT]",      "immutable"),
        ("[TEMPORAL_HEDGE]", "fast"),
        ("[CONFIDENT]",      "fast"),      # over-confident on volatile
        ("[UNKNOWN]",        "immutable"), # refusing the obvious
        ("[COND_CONFIDENT]", "slow"),
    ]
    for hedge, vol in test_cases:
        score = automated_hedge_score(hedge, vol)
        print(f"  hedge={hedge:<22} volatility={vol:<10} score={score}/5")

    with open("hedge_quality_rubric.md", "w") as f:
        f.write(RUBRIC)
    print("\nRubric saved to hedge_quality_rubric.md (for MTurk + paper appendix)")
