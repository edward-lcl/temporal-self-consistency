# TCL Fix Validation — Small Proxy Model Diagnostic

_Written 2026-08-09. Validates the differentiable-`c_hat` fix documented in
`docs/tcl_debugging.md` before committing to a full 8B run, per
`specs/tsct-project-state.md`'s immediate-next-actions list._

> ## ⚠️ CORRECTION NOTICE — added 2026-08-11
>
> This document was written from a **single seed (seed 0)**. Replication across
> seeds 1-3 confirmed one of its two headline results and **refuted the other**.
> The original text is left unedited below so the record stands; read it with
> these two corrections in hand. Full detail: `specs/CLAIMS-LEDGER.md` A1/A2.
>
> **1. The gradient-connectivity result HOLDS — stronger than claimed.**
> `broken` gives exactly 0.0 at 78/78 steps in all four seeds; `fixed` gives
> nonzero at 78/78 in all four (means 3.69-4.47). It also holds at 7B:
> 1202/1202 probes nonzero across 3 seeds. This is the load-bearing result and
> it is now well replicated.
>
> **2. The hedge-collapse result DOES NOT REPLICATE.** The section
> "Hedge-token collapse check" below, and the sentence in "Verdict" beginning
> _"the collapse check reproduces the original failure mode…"_, are **not
> supported**. Across seeds:
>
> | seed | `broken` post-train | `fixed` post-train |
> |---|---|---|
> | 0 (this doc) | `[CONFIDENT]` 99.0% | `[TEMPORAL_HEDGE]` 66.3% |
> | 1 | `[CONFIDENT]` 100% | `[TEMPORAL_HEDGE]` **100%** |
> | 2 | `[CONFIDENT]` 100% | `[TEMPORAL_HEDGE]` 96.2% |
> | 3 | **`[TEMPORAL_HEDGE]` 62.5%** | **`[CONFIDENT]` 100%** |
>
> At seed 3 both directions invert: `fixed` collapses to 100% `[CONFIDENT]` —
> the exact pathology the fix is meant to prevent — while `broken` fails to
> collapse to it at all. At 78 steps on 104 examples the post-train
> distribution is seed noise, not signal.
>
> **The verdict "the fix is confirmed working" still stands**, but on **one**
> load-bearing result rather than two.
>
> **3. One caveat in this document proved understated.** It warns against using
> aggregate `tcl_total` drift as a gate-check. That was right, and the reason is
> sharper than stated: at 7B the calibration terms are ~1% of the summed loss,
> so `tcl_total` is ~94% cross-entropy. See CLAIMS-LEDGER B2.

## What this is

A from-scratch, standalone reimplementation of the TCL training loop under
`src/training/`, built specifically to validate the gradient-path fix. The
original `src/training/tcl_loss.py` referenced in `docs/tcl_debugging.md`
lives in a separate, unlinked repo owned by the training lead (Tanvi) and is
not available here — this is not a recovery of that code, it's an
independent reimplementation built to test one specific claim: that
replacing the argmax+lookup `c_hat` computation with a softmax-weighted
expectation reconnects the calibration loss's gradient to the model.

Every design choice not explicitly specified in `docs/tcl_debugging.md` (the
concrete forms of `L_over`, `L_under`, `R_hedge`) is documented as an
assumption in `src/training/tcl_loss.py`'s module docstring — treat those as
placeholders for the real formulas if/when the original loss code surfaces.
**The differentiability fix itself is formula-agnostic** and is the thing
actually under test here.

## Setup

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (proxy for the target LLaMA-3 8B —
  small enough to iterate in seconds/minutes on Apple Silicon MPS, same
  causal-LM architecture family).
- **Hedge tokens:** added as `additional_special_tokens`, embeddings resized.
  Verified each of the 4 tokens tokenizes to exactly one id (required for
  the loss to index a single logit position per hedge token).
- **Data:** 104 real examples pulled live from `jasontae/temporal-delta` on
  HuggingFace (train split), volatility-balanced (48 fast / 48 slow / 8
  immutable — immutable is capped at 8 because only 8 immutable-labeled
  rows exist in the full 14,206-row train split). Not synthetic.
- **Hyperparameters:** the revised set from `docs/tcl_debugging.md`'s table
  — epochs=3, lr=5e-5, lambda_over=0.5, lambda_under=0.5, lambda_hedge=0.3.
  Batch size 4 → 78 steps per condition (within the doc's own 50–200 step
  diagnostic-run guidance).
- **Two conditions, same data/hparams/seed, run back to back:**
  - `broken` — reproduces the original bug verbatim (argmax-select hedge
    token, then discrete lookup into the confidence table).
  - `fixed` — the documented fix (softmax-weighted expectation over hedge
    logits).
- **Logging:** every step logs `ce`, `l_over`, `l_under`, `r_hedge`,
  `tcl_total`, `n_volatile` (batch composition), `pred_confident_frac`, and
  `c_hat_grad_norm` as **separate columns** — this separation is exactly
  what the original debugging notes flagged as missing from the first run.
  Full log: `data/prep/tcl_diagnostic/loss_log.csv` (gitignored, see
  `data/prep/*.jsonl` pattern — csv added as an exception, see note below).

## The direct gradient-connectivity test

Beyond just watching `tcl_total`, each step also isolates
`c_hat_loss = lambda_over * L_over + lambda_under * L_under` (the two terms
that are pure functions of `c_hat` — `R_hedge` is built straight from
`hedge_probs` and stays differentiable in both conditions, so it's not a
useful signal for this specific bug) and backprops it alone into the LM
head weight, before the real optimizer step, to measure `c_hat_grad_norm`.

This is the single clearest result:

| condition | c_hat_grad_norm | interpretation |
|---|---|---|
| `broken` | **0.0** at every one of 78 steps | argmax detaches `c_hat` from the autograd graph — exactly the bug as diagnosed |
| `fixed` | mean 4.26, max 14.65, **nonzero at 100% of steps** | softmax expectation keeps the path connected end to end |

This is not a subtle statistical difference — it's exactly 0 vs. never 0,
which is what "gradient severed at a discrete op" vs. "gradient connected"
looks like when measured directly rather than inferred from loss curves.

## Hedge-token collapse check

Full-dataset predicted-hedge distribution before and after training:

| | `[CONFIDENT]` | `[COND_CONFIDENT]` | `[TEMPORAL_HEDGE]` | `[UNKNOWN]` |
|---|---|---|---|---|
| pre-train (both conditions, same init) | 70.2% | 3.8% | 5.8% | 20.2% |
| **broken**, post-train | **99.0%** | 0.0% | 1.0% | 0.0% |
| **fixed**, post-train | 33.7% | 0.0% | **66.3%** | 0.0% |

`broken` reproduces the reported failure mode almost exactly: collapse to
~100% `[CONFIDENT]`, the same pathology that made all 6 original checkpoints
unusable. `fixed` does **not** collapse to a single token — it moves toward
a distribution dominated by `[TEMPORAL_HEDGE]`, which is the correct
direction given the training slice's label composition (46% of examples
are labeled `[TEMPORAL_HEDGE]` from the fast-volatility rows; 8 immutable
examples out of 104 give little signal to also learn `[COND_CONFIDENT]`,
so its complete absence post-train is expected from this data slice, not a
symptom of the bug).

## Loss magnitudes (context, not the primary evidence)

`tcl_total` drift across the run (first step vs. last step) was ~93% in
**both** conditions — CE dominates the summed loss for a freshly
resized 0.5B model and drops sharply regardless of whether the calibration
path is connected, so **aggregate `tcl_total` drift is not a reliable
diagnostic signal at this scale/setup** (it was the signal that worked in
the original 8B run because that model's CE had already converged going
into the TCL phase, leaving `tcl_total`'s movement — or lack of it — cleanly
attributable to the calibration terms). This is a real deviation worth
flagging: **don't rely on aggregate `tcl_total` drift alone as a
gate-check on the 8B run either** — use the per-term breakdown and the
isolated `c_hat_grad_norm` check, which are scale-independent and were
unambiguous here.

`l_over`/`l_under` magnitudes themselves were small and noisy over only 78
steps on 104 examples (as expected for a diagnostic run, not a convergence
run) — not treated as evidence either way; the gradient-connectivity check
and the collapse check are the load-bearing results.

Full per-step numbers: `data/prep/tcl_diagnostic/loss_log.csv`.
Run config: `data/prep/tcl_diagnostic/run_meta.json`.

## Deviations from the documented fix / assumptions made

1. **`L_over`/`L_under`/`R_hedge` formulas are inferred, not recovered.**
   See `src/training/tcl_loss.py` docstring for the exact forms used
   (asymmetric squared-hinge over/under penalties vs. `c_gold`, plus an
   entropy-based anti-collapse regularizer). If the original formulas
   differ, re-validate with the real ones — but the differentiability fix
   itself doesn't depend on which forms are used.
2. **TCL restricted to volatile (fast/slow) examples only**, per the
   debugging doc's diagnostic checklist. `n_volatile` was logged per batch
   and never hit 0 in either condition (min 2, max 4 out of batch size 4),
   ruling out the "batch accidentally has ~0 volatile examples" false
   positive the original debugging process had to check for.
3. Trained in fp32 on MPS, not bf16/LoRA — deliberate for the diagnostic
   (full fine-tune of a 0.5B model is cheap and removes LoRA-adapter
   gradient-flow as a confound while isolating the c_hat bug specifically).
4. `data/prep/tcl_diagnostic/loss_log.csv` and `run_meta.json` are
   committed despite the existing `data/prep/*.jsonl` gitignore pattern
   (csv/json aren't matched by that glob) — kept as the primary evidence
   artifact for this doc.

## Verdict

**The fix is confirmed working at the small-proxy-model scale.** The
gradient-connectivity test gives an unambiguous, mechanism-level result
(exactly 0 vs. never 0), independent of the inferred-loss-formula caveat
above, and the collapse check reproduces the original failure mode in the
`broken` condition while showing no collapse in the `fixed` condition on
the same data/hparams/seed.

## Recommendation: local M5 Pro vs. cloud for the full 7-8B run

**Recommend cloud GPU compute for the full run, not local M5 Pro.**

Reasoning:
- This diagnostic (0.5B, full fine-tune, fp32, 104 examples, 3 epochs = 78
  steps) took **47s** on MPS for the `fixed` condition. Scaling naively by
  parameter count (8B / 0.5B ≈ 16x) and accounting for the fact that a real
  8B run needs LoRA (not full fine-tune) plus the full ~14,206-example
  train split rather than 104 examples, and 3 epochs over that — the
  proxy run does not scale to a "few extra minutes" estimate; it scales to
  **hours-to-a-day-plus territory** on a single MPS device with no CUDA
  (bitsandbytes 4-bit quant unavailable on MPS, so this would be bf16/fp16
  LoRA, slower than the team's likely original CUDA setup, per
  `specs/tsct-project-state.md`).
- Kiran's team close-out is ~2026-08-20. A multi-day per-attempt local loop
  (with only one seed at a time, no parallelism across the SFT×3/TSCT×3
  seed matrix the original run needed) does not fit that timeline, and any
  bug found after a long local run costs a full day to iterate on.
- Cloud GPU (A100/H100 class) gets bitsandbytes 4-bit QLoRA back, running
  6 seeded checkpoints in parallel or in a tight sequential loop, likely in
  under a day total.

**Next command to scale up** (once cloud compute is provisioned): reuse
`src/training/run_tcl_diagnostic.py`'s structure directly — swap
`--model Qwen/Qwen2.5-0.5B-Instruct` for the target 8B checkpoint, wire in
LoRA (`peft`) around `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)`,
point `load_slice()` at the full train split instead of a 48/48/8-capped
balanced slice, and drop the `broken` condition (it's only useful for this
validation, not for production training). The `fixed`-mode `compute_c_hat`
path, `tcl_terms`, and the per-step separated logging (including
`c_hat_grad_norm` as an ongoing gradient-health check) should carry over
unchanged.
