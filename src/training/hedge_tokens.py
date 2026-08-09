"""Hedge token definitions shared by the training and eval sides.

Mirrors the HEDGE_TO_CONFIDENCE mapping in src/evaluation/eval_pipeline.py --
kept as a separate constant here (not imported) because the training and
eval codepaths are intentionally decoupled (different repos upstream), but
the four values must stay in sync.
"""

HEDGE_TOKENS = ["[CONFIDENT]", "[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[UNKNOWN]"]

HEDGE_TO_CONFIDENCE = {
    "[CONFIDENT]": 0.95,
    "[COND_CONFIDENT]": 0.75,
    "[TEMPORAL_HEDGE]": 0.45,
    "[UNKNOWN]": 0.10,
}


def add_hedge_tokens(tokenizer, model):
    """Add the 4 hedge tokens as special/atomic tokens and resize embeddings.

    Returns the list of new token ids (aligned with HEDGE_TOKENS order).
    Adding them via `additional_special_tokens` guarantees each hedge token
    tokenizes to exactly one id (required -- the TCL loss indexes a single
    logit position per hedge token), and that the tokenizer never silently
    splits e.g. "[TEMPORAL_HEDGE]" into sub-word pieces.
    """
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in HEDGE_TOKENS if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))

    hedge_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in HEDGE_TOKENS]
    for tok, tid in zip(HEDGE_TOKENS, hedge_token_ids):
        assert tid is not None and tid != tokenizer.unk_token_id, (
            f"hedge token {tok!r} did not tokenize to a dedicated id"
        )
    return hedge_token_ids
