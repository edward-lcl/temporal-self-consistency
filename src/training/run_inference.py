"""
Inference Runner (template)
===========================
Runs a fine-tuned checkpoint on a benchmark set and produces predictions
in the canonical eval-pipeline format.

This is the script that produces the prediction files the eval pipeline
consumes. Designed to run on Colab Pro+ / any CUDA GPU.

CRITICAL NOTES (learned the hard way during this project):
  - Use the INSTRUCT model for the base-LLM baseline (it needs to follow
    the output format). Use the BASE model only as the starting point for
    fine-tuning.
  - Use the tokenizer's chat template, not raw string prompts.
  - For the hedge token: extract the differentiable softmax probability
    over the four hedge token IDs, do NOT greedy-decode then look up.
    (Greedy decoding collapses every output to [CONFIDENT].)

Usage:
    python run_inference.py --checkpoint <hf_repo_or_path> \\
        --benchmark temporal_delta_test.jsonl \\
        --output predictions/tsct_seed42_temporal_delta.jsonl
"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HEDGE_TOKENS = ["[CONFIDENT]", "[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[UNKNOWN]"]


def load_model(checkpoint, subfolder=None):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if subfolder:
        kwargs["subfolder"] = subfolder
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    tok = AutoTokenizer.from_pretrained(checkpoint, subfolder=subfolder)
    return model, tok


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system",
         "content": "You answer factual questions and emit a hedge token."},
        {"role": "user",
         "content": (f"Answer this question, then emit exactly one hedge token "
                     f"from {HEDGE_TOKENS}.\n\nQuestion: {question}\n\n"
                     f"Format: Answer: <text> Hedge: <token>")},
    ]
    return tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )


def parse_output(text):
    """Extract answer text and hedge token from model output."""
    answer, hedge = text, "[UNKNOWN]"
    if "Answer:" in text:
        answer = text.split("Answer:", 1)[1]
    for tok in HEDGE_TOKENS:
        if tok in text:
            hedge = tok
            answer = answer.split("Hedge:")[0] if "Hedge:" in answer else answer
            break
    return answer.strip(), hedge


def run(checkpoint, benchmark_path, output_path, subfolder=None):
    model, tokenizer = load_model(checkpoint, subfolder)
    model.eval()

    predictions = []
    with open(benchmark_path) as f:
        rows = [json.loads(l) for l in f]

    for row in rows:
        question = row["question"]
        gold = row.get("gold_answer", "")
        inputs = build_prompt(tokenizer, question).to(model.device)

        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id, do_sample=False,
            )
        new_tokens = out[0][inputs.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        answer, hedge = parse_output(decoded)
        correct = answer.strip().lower() == str(gold).strip().lower()

        rec = {
            "predicted_answer": answer,
            "gold_answer":      gold,
            "predicted_hedge":  hedge,
            "correct":          correct,
            "volatility":       row.get("volatility", "unknown"),
        }
        if row.get("t_end"):
            try:
                rec["change_year"] = int(row["t_end"])
            except (ValueError, TypeError):
                pass
        predictions.append(rec)

    with open(output_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args.checkpoint, args.benchmark, args.output, args.subfolder)
