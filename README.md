# TSCT: Temporal Self-Consistency Training

Training language models to natively distinguish time-stable from time-volatile facts and express calibrated temporal uncertainty — without retrieval at inference time.

## Overview

Language models answer questions about a changing world using frozen weights. They confidently assert facts that may have changed since training (current CEOs, political leaders, prices) with no signal of temporal uncertainty. **Temporal Self-Consistency Training (TSCT)** is a fine-tuning method that teaches a model to emit one of four hedge tokens with every answer, calibrated to the fact's volatility:

| Hedge token | Confidence | Use case |
|---|---|---|
| `[CONFIDENT]` | 0.95 | Immutable facts (constants, history) |
| `[COND_CONFIDENT]` | 0.75 | Slow-changing (capitals, org structure) |
| `[TEMPORAL_HEDGE]` | 0.45 | Fast-changing (leadership, prices) |
| `[UNKNOWN]` | 0.10 | Beyond reliable knowledge horizon |

**Research question:** Can a language model be trained to reliably distinguish time-stable from time-volatile factual claims and express calibrated uncertainty about the latter, without access to external retrieval at inference time?

## Repository structure

```
tsct-temporal-calibration/
├── data/
│   ├── prep/               # (gitignored) large generated datasets
│   ├── stress_tests/       # adversarial + mixed-paragraph stress sets
│   └── samples/            # small samples for inspection
├── src/
│   ├── data_pipeline/      # dataset construction + stress-test generation
│   │   ├── prep_mmlu.py            # MMLU stable subset (regression check)
│   │   ├── prep_stress_horizon.py  # 18-36 month post-cutoff facts
│   │   ├── prep_stress_stable.py   # over-hedging detector set
│   │   └── prep_stress_mixed.py    # mixed stable/volatile paragraphs
│   ├── evaluation/         # the metrics pipeline
│   │   ├── eval_pipeline.py        # ECE, EM/F1, volatility, Bonferroni, etc.
│   │   ├── adapt_predictions.py    # normalize team output formats
│   │   ├── full_analysis.py        # run all prediction files
│   │   ├── generate_results_table.py  # MD/LaTeX/CSV results tables
│   │   ├── hedge_quality_rubric.py # human-eval rubric + auto scoring
│   │   └── paper_plots.py          # all 6 paper figures
│   └── training/           # training-side reference code
│       ├── tcl_loss.py             # Temporal Calibration Loss (reference)
│       └── run_inference.py        # checkpoint -> predictions
├── figures/                # generated paper figures
├── paper/                  # paper draft
├── docs/                   # proposal, status, design notes
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Build evaluation datasets
python src/data_pipeline/prep_mmlu.py
python src/data_pipeline/prep_stress_stable.py
python src/data_pipeline/prep_stress_mixed.py
# prep_stress_horizon.py needs all_triples.jsonl in the working dir

# 2. Run a checkpoint on a benchmark (needs GPU)
python src/training/run_inference.py \
    --checkpoint <hf_repo> --subfolder exp3_tsct_seed42 \
    --benchmark temporal_delta_test.jsonl \
    --output predictions/tsct_seed42_temporal_delta.jsonl

# 3. Normalize any team prediction file to the canonical format
python src/evaluation/adapt_predictions.py raw_preds.jsonl clean_preds.jsonl

# 4. Run the full evaluation across all prediction files
TSCT_PREDICTIONS_DIR=./predictions python src/evaluation/full_analysis.py

# 5. Generate figures and the results table
python src/evaluation/paper_plots.py
python src/evaluation/generate_results_table.py
```

## Prediction format

Every prediction the eval pipeline consumes must be a JSON object with:

```json
{
  "predicted_answer": "John Donahoe",
  "gold_answer": "John Donahoe",
  "predicted_hedge": "[TEMPORAL_HEDGE]",
  "correct": true,
  "volatility": "fast",
  "change_year": 2024
}
```

`adapt_predictions.py` converts the known team output variants into this format.

## Evaluation metrics

1. **ECE** — Expected Calibration Error, equal-frequency binning
2. **Accuracy** — exact match + token-level F1
3. **Volatility breakdown** — ECE/accuracy split by fast/slow/immutable
4. **Bonferroni comparison** — TSCT vs 7 baselines, corrected alpha approximately 0.007
5. **Volatility discrimination** — confusion matrix vs ground-truth volatility
6. **Temporal generalization gap** — performance vs months past cutoff

## Datasets

| Dataset | Role |
|---|---|
| TemporalDelta (ours) | Training + test — Wikidata 11-property SPARQL extraction |
| PAT-Questions | Training contrastive pairs + eval |
| FreshQA | Primary eval — fast-changing facts |
| TLQA | Eval — list-based temporal QA |
| TDBench | Eval — time-accuracy metric |
| MMLU stable subset | Regression check — accuracy must not degrade |

## Status

See `docs/STATUS.md` for the live task tracker. As of the latest update, the evaluation infrastructure is complete and verified; model checkpoints are being debugged (see `docs/tcl_debugging.md`).

## Team

Jason (project lead + evaluation), Tanvi (training), Logan (baselines), Aarav (data).
