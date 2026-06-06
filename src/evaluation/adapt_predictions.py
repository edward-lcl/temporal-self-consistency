"""
Prediction Format Adapter
=========================
Normalizes prediction files from different team members into the
canonical format expected by eval_pipeline.py.

The eval pipeline expects each prediction dict to have:
    predicted_answer : str
    gold_answer      : str
    predicted_hedge  : str   one of [CONFIDENT]/[COND_CONFIDENT]/[TEMPORAL_HEDGE]/[UNKNOWN]
    correct          : bool
    volatility       : str   fast / slow / immutable
    change_year      : int   (optional, for temporal generalization)

This adapter handles the known format variants we've encountered:
    - Logan's baseline format (model_answer, correct_answer, fact_type, hedge without brackets)
    - Tanvi's prediction format (predicted_hedge present but no predicted_answer text)
    - NaN model answers (parse failures)
"""
import json
import math

# Volatility label normalization
VOLATILITY_MAP = {
    "fast-changing":  "fast",
    "slow-changing":  "slow",
    "never-changing": "immutable",
    "fast":           "fast",
    "slow":           "slow",
    "immutable":      "immutable",
}

# Hedge token normalization (some files omit brackets)
HEDGE_MAP = {
    "CONFIDENT":        "[CONFIDENT]",
    "COND_CONFIDENT":   "[COND_CONFIDENT]",
    "TEMPORAL_HEDGE":   "[TEMPORAL_HEDGE]",
    "UNKNOWN":          "[UNKNOWN]",
    "conf":             "[CONFIDENT]",
    "cond":             "[COND_CONFIDENT]",
    "temp":             "[TEMPORAL_HEDGE]",
    "unk":              "[UNKNOWN]",
    "[CONFIDENT]":      "[CONFIDENT]",
    "[COND_CONFIDENT]": "[COND_CONFIDENT]",
    "[TEMPORAL_HEDGE]": "[TEMPORAL_HEDGE]",
    "[UNKNOWN]":        "[UNKNOWN]",
}


def safe_str(value, default="PARSE_FAILED"):
    """Handle NaN / None / non-string answer fields."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if not isinstance(value, str):
        return str(value)
    return value


def normalize_record(r):
    """Convert a single record from any known variant to canonical format."""
    # Answer fields — try several key names
    predicted = (r.get("predicted_answer")
                 or r.get("model_answer")
                 or r.get("answer"))
    gold = (r.get("gold_answer")
            or r.get("correct_answer")
            or r.get("gold"))

    # Hedge field
    raw_hedge = r.get("predicted_hedge") or r.get("hedge") or "[UNKNOWN]"
    hedge = HEDGE_MAP.get(raw_hedge, "[UNKNOWN]")

    # Volatility field
    raw_vol = r.get("volatility") or r.get("fact_type") or "unknown"
    volatility = VOLATILITY_MAP.get(raw_vol, raw_vol)

    # Correctness — recompute if missing
    if "correct" in r:
        correct = bool(r["correct"])
    else:
        correct = (safe_str(predicted).strip().lower()
                   == safe_str(gold).strip().lower())

    # Change year for temporal generalization
    change_year = r.get("change_year")
    if change_year is None and r.get("t_end"):
        try:
            change_year = int(r["t_end"])
        except (ValueError, TypeError):
            change_year = None

    out = {
        "predicted_answer": safe_str(predicted),
        "gold_answer":      safe_str(gold),
        "predicted_hedge":  hedge,
        "correct":          correct,
        "volatility":       volatility,
    }
    if change_year is not None:
        out["change_year"] = change_year
    return out


def adapt_file(input_path, output_path=None):
    """Adapt a whole prediction file. Handles both JSONL and JSON-array inputs."""
    with open(input_path) as f:
        content = f.read().strip()

    # Detect format: JSON array vs JSONL
    if content.startswith("["):
        records = json.loads(content)
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]

    adapted = [normalize_record(r) for r in records]

    # Report stats
    n_parse_fail = sum(1 for a in adapted if a["predicted_answer"] == "PARSE_FAILED")
    if n_parse_fail:
        print(f"  WARNING: {n_parse_fail}/{len(adapted)} records had unparseable answers")

    if output_path:
        with open(output_path, "w") as f:
            for a in adapted:
                f.write(json.dumps(a) + "\n")
        print(f"  Wrote {len(adapted)} records to {output_path}")

    return adapted


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python adapt_predictions.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    adapt_file(inp, out)
