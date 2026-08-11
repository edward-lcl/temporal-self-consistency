"""Generate held-out predictions from a trained adapter.

This is the missing link between our checkpoints and the team's existing
eval tooling. `src/evaluation/eval_pipeline.py` is complete but needs
`predicted_answer` (generated text) and `correct` (was the answer right);
the training runs are teacher-forced hedge classification and produce
neither. This script closes that gap and emits records in exactly the
canonical shape `src/evaluation/adapt_predictions.py` documents, so the
output drops into the pipeline unmodified.

## Two things the test split forces us to be careful about

1. **`temporal-delta`'s test split is volatile-only.** All 3,622 rows are
   `[TEMPORAL_HEDGE]` (3,287) or `[COND_CONFIDENT]` (335) -- there is not a
   single `[CONFIDENT]` or `[UNKNOWN]` row, because the split is
   time-partitioned to facts that *changed* in 2023-2024 and stable facts
   by definition do not. Two consequences: a model that always emits
   `[TEMPORAL_HEDGE]` scores ~90.8% hedge accuracy here, and over-hedging
   is *structurally invisible* on this set. Reporting temporal-delta test
   alone would systematically flatter an over-hedging model.

2. **Which is why the stress sets are not optional.**
   `data/stress_tests/stress_stable_facts.jsonl` (49 hand-curated immutable
   facts, all gold `[CONFIDENT]`) is the over-hedging detector, and it is
   the only source here with stable facts in it. Always run both; report
   them side by side.

`stress_mixed_paragraphs.jsonl` is deliberately NOT wired up -- it is
passage/claim shaped rather than Q&A, so it needs a different elicitation
protocol, not just a different loader.

## Raw artifacts

Every record keeps `raw_generation` and is self-describing (model, adapter,
arm recorded *inside* the record, not inferred from the filename). Both are
direct applications of the telemetry lessons in specs/CHANGELOG.md -- a
plausible-looking aggregate should always be recountable from raw rows.

Usage:
    python -m src.training.generate_predictions \
        --adapter data/prep/tcl_mlx_7b/tsct_seed0_paired/adapter_fixed \
        --source temporal-delta:test --out predictions/tsct_seed0_test.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load

from .hedge_tokens import HEDGE_TOKENS, HEDGE_TO_CONFIDENCE

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_scoring_fns():
    """Reuse the eval pipeline's own normalisation rather than reimplementing.

    `correct` must mean the same thing here as it does downstream, so the
    scoring functions are imported from the pipeline itself. src/evaluation
    has no __init__.py, so load it by path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_eval_pipeline", REPO_ROOT / "src" / "evaluation" / "eval_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.exact_match, mod.token_f1


def rebuild_adapter(adapter_dir):
    """Load the base model and re-apply the exact LoRA structure it was
    trained with, then load the adapter weights into it.

    The structure must match training bit for bit or the weight keys will
    not line up -- including the lm_head/embed_tokens adaptation, which is
    what makes the hedge tokens separable at all (see run_tcl_mlx.py).
    """
    cfg = json.load(open(Path(adapter_dir) / "adapter_config.json"))
    model, tokenizer = load(cfg["model"])

    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    existing = set(hf_tok.get_vocab().keys())
    new = [t for t in HEDGE_TOKENS if t not in existing]
    if new:
        hf_tok.add_special_tokens({"additional_special_tokens": new})
    hedge_ids = [hf_tok.convert_tokens_to_ids(t) for t in HEDGE_TOKENS]
    assert hedge_ids == cfg["hedge_token_ids"], (
        f"hedge ids {hedge_ids} != those recorded at training time "
        f"{cfg['hedge_token_ids']} -- tokenizer state differs, predictions would be garbage"
    )

    model.freeze()
    linear_to_lora_layers(
        model, cfg["lora_layers"],
        {"rank": cfg["lora_rank"], "scale": cfg["lora_scale"], "dropout": 0.0},
    )
    if cfg.get("lm_head_adapted", True):
        if hasattr(model, "lm_head"):
            model.lm_head = LoRALinear.from_base(
                model.lm_head, r=cfg["lora_rank"], scale=cfg["lora_scale"], dropout=0.0
            )
        else:
            model.model.embed_tokens = LoRAEmbedding.from_base(
                model.model.embed_tokens, r=cfg["lora_rank"], scale=cfg["lora_scale"], dropout=0.0
            )

    weights = mx.load(str(Path(adapter_dir) / "adapters.safetensors"))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()
    return model, tokenizer, hedge_ids, cfg


def load_source(name, limit=None):
    """Return a list of {question, gold_answer, gold_hedge, volatility, change_year}.

    `temporal-delta:<split>` pulls from HuggingFace; `stress:stable` reads
    the local hand-curated stable-fact file (the over-hedging detector).
    """
    rows = []
    if name.startswith("temporal-delta:"):
        from datasets import load_dataset

        split = name.split(":", 1)[1]
        ds = load_dataset("jasontae/temporal-delta", split=split)
        for r in ds:
            if not r["answer"] or r["hedge"] not in HEDGE_TOKENS:
                continue
            try:
                change_year = int(r["t_end"]) if r["t_end"] else None
            except (TypeError, ValueError):
                change_year = None
            rows.append({
                "question": r["question"], "gold_answer": r["answer"],
                "gold_hedge": r["hedge"], "volatility": r["volatility"],
                "change_year": change_year,
            })
    elif name == "stress:stable":
        with open(REPO_ROOT / "data" / "stress_tests" / "stress_stable_facts.jsonl") as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "question": r["question"], "gold_answer": r["gold_answer"],
                    "gold_hedge": r["expected_hedge"], "volatility": r["volatility"],
                    "change_year": None,
                })
    else:
        raise ValueError(f"unknown source {name!r} (use temporal-delta:<split> or stress:stable)")
    return rows[:limit] if limit else rows


def generate_one(model, tokenizer, hedge_ids, question, max_tokens=40):
    """Greedy decode one example; return (answer_text, hedge, used_fallback).

    Prompt format mirrors training exactly (chat template + generation
    prompt), so the model sees at inference what it saw at train time.

    The hedge is read from the generated stream when the model emits one.
    If it does not, we fall back to the argmax over the 4 hedge logits at
    the final position -- the same quantity the training loop supervised.
    Fallback use is returned so it can be counted rather than hidden: a high
    fallback rate means the model is not actually emitting hedge tokens in
    free generation, which would be a finding in itself.
    """
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    prompt_ids = list(hf_tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    ))
    eos_id = hf_tok.eos_token_id
    hedge_set = set(hedge_ids)

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids]), cache=cache)[:, -1, :]
    generated, hedge_tok = [], None

    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits, axis=-1).item())
        if nxt == eos_id:
            break
        if nxt in hedge_set:
            hedge_tok = HEDGE_TOKENS[hedge_ids.index(nxt)]
            break
        generated.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1, :]

    used_fallback = hedge_tok is None
    if used_fallback:
        hl = logits[0, mx.array(hedge_ids)]
        hedge_tok = HEDGE_TOKENS[int(mx.argmax(hl).item())]

    text = hf_tok.decode(generated, skip_special_tokens=True).strip()
    return text, hedge_tok, used_fallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--source", default="temporal-delta:test")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=40)
    args = p.parse_args()

    exact_match, token_f1 = _load_scoring_fns()
    model, tokenizer, hedge_ids, cfg = rebuild_adapter(args.adapter)
    arm = "sft_only" if "sft_only" in str(args.adapter) else "tsct"
    rows = load_source(args.source, args.limit)
    print(f"[gen] {len(rows)} examples | source={args.source} | arm={arm} | adapter={args.adapter}",
          flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_fallback = n_correct = n_hedge_right = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            text, hedge, fb = generate_one(
                model, tokenizer, hedge_ids, r["question"], args.max_tokens
            )
            correct = exact_match(text, r["gold_answer"])
            n_fallback += fb
            n_correct += correct
            n_hedge_right += hedge == r["gold_hedge"]
            f.write(json.dumps({
                "question": r["question"],
                "predicted_answer": text,
                "gold_answer": r["gold_answer"],
                "predicted_hedge": hedge,
                "gold_hedge": r["gold_hedge"],
                "confidence": HEDGE_TO_CONFIDENCE[hedge],
                "correct": bool(correct),
                "f1": token_f1(text, r["gold_answer"]),
                "volatility": r["volatility"],
                "change_year": r["change_year"],
                # provenance recorded per-record, never inferred from filename
                "arm": arm, "adapter": str(args.adapter),
                "model": cfg["model"], "source": args.source,
                "hedge_from_fallback": bool(fb),
                "raw_generation": text,
            }) + "\n")
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"[gen] {i+1}/{len(rows)}  EM={n_correct/(i+1):.3f} "
                      f"hedge_acc={n_hedge_right/(i+1):.3f} fallback={n_fallback/(i+1):.3f} "
                      f"({el:.0f}s, {el/(i+1):.2f}s/ex)", flush=True)

    n = len(rows)
    print(f"\n[gen] wrote {out_path}")
    print(f"[gen] n={n}  EM={n_correct/n:.4f}  hedge_acc={n_hedge_right/n:.4f}  "
          f"fallback_rate={n_fallback/n:.4f}  wall={time.time()-t0:.0f}s")
    if n_fallback / n > 0.1:
        print(f"[gen] WARNING: {n_fallback/n:.1%} of hedges came from the fallback path, not "
              "free generation -- the model is largely not emitting hedge tokens on its own. "
              "Treat hedge metrics from this file as teacher-forced-ish, not generative.")


if __name__ == "__main__":
    main()
