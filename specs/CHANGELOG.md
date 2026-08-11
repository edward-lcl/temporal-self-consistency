# TSCT Working Log

Running record of everything we've found and changed since joining the project (2026-08-08). Newest entries on top. This is the log Edward asked for -- keep every fix, before/after state, and decision here going forward, not just in chat.

_Housekeeping: the "newest on top" convention has drifted -- the two 2026-08-09 entries below the "Open threads" block were appended at the bottom. Left in place rather than silently reordered; worth a cleanup pass._

---

## 2026-08-10 -- Re-verification of the TCL fix: gradient result holds at n=4 seeds, collapse result does NOT

**Ask:** Edward asked to catch up on the OpenClaw analysis and kick off the scale-up experiments before end of day.

**Re-ran the committed proxy diagnostic (seed 0), bit for bit.** `fixed` mean `c_hat_grad_norm` 4.2564 vs the committed 4.2565 (float rounding), max 14.6506 identical, both hedge distributions identical. `specs/tcl-fix-validation.md` reproduces exactly. Environment is sound.

**Then replicated across seeds 1-3, which the original run did not do.** Two very different outcomes for the two headline claims:

| seed | `broken` grad | `fixed` grad (mean) | `broken` post-train top token | `fixed` post-train top token |
|---|---|---|---|---|
| 0 | 0/78 nonzero | 78/78, 4.2564 | `[CONFIDENT]` 99.0% | `[TEMPORAL_HEDGE]` 66.3% |
| 1 | 0/78 nonzero | 78/78, 3.6946 | `[CONFIDENT]` 100% | `[TEMPORAL_HEDGE]` **100%** |
| 2 | 0/78 nonzero | 78/78, 4.4739 | `[CONFIDENT]` 100% | `[TEMPORAL_HEDGE]` 96.2% |
| 3 | 0/78 nonzero | 78/78, 4.1158 | **`[TEMPORAL_HEDGE]` 62.5%** | **`[CONFIDENT]` 100%** |

1. **The gradient-connectivity result is rock solid.** `broken` gives exactly 0.0 at every one of 78 steps in all 4 seeds; `fixed` is nonzero at 100% of steps in all 4, mean 3.69-4.47. This is the mechanism-level claim and it replicates without exception.
2. **The hedge-collapse result does not replicate and should not be reported as a finding.** `specs/tcl-fix-validation.md` states `fixed` "does **not** collapse to a single token." At seed 1 it collapses to 100% `[TEMPORAL_HEDGE]`, and at seed 3 it collapses to **100% `[CONFIDENT]`** -- the exact pathology the fix is supposed to prevent -- while `broken` at that same seed fails to collapse to `[CONFIDENT]` at all. Both directions invert. At 78 steps on 104 examples the post-train distribution is seed noise, not signal.

**Consequence:** the verdict "fix confirmed working" still stands, but it rests on **one** load-bearing result, not two. `specs/tcl-fix-validation.md` should be amended -- its "Hedge-token collapse check" section and the second half of its "Verdict" currently overclaim from a single seed. Not edited yet; flagging rather than rewriting someone else's evidence doc unilaterally.

**Where:** `data/prep/tcl_diagnostic_seeds/` (to be committed alongside this entry).

## 2026-08-11 -- FIRST REAL ECE. The paper's headline effect looks like an artifact of a broken baseline.

First held-out ECE numbers this project has ever had, computed by Jason's own `eval_pipeline.py` on generated predictions from both arms.

**temporal-delta test (n=3,622, volatile-only by construction):**

| arm | ECE | EM | F1 | hedge acc | mean asserted conf |
|---|---|---|---|---|---|
| SFT-only | 0.4552 | 0.0226 | 0.0402 | 0.9997 | 0.4778 |
| TSCT | **0.4506** | 0.0273 | 0.0454 | 0.9994 | 0.4780 |

**stress_stable_facts (n=49, all gold `[CONFIDENT]` -- the over-hedging detector, never run before):**

| arm | ECE | EM | hedge acc | predicted distribution |
|---|---|---|---|---|
| SFT-only | 0.1520 | 0.8571 | 1.0000 | 49 `[CONFIDENT]` |
| TSCT | 0.1602 | 0.8776 | 0.9388 | 46 `[CONFIDENT]`, 3 `[COND_CONFIDENT]` |

### 1. TCL's ECE benefit is 0.0046. It is not there.

On the benchmark the paper is built around, TSCT and SFT are indistinguishable.

### 2. Why the Doc's number was so much bigger -- and why we should be careful about it

The Slack table (recorded in the 2026-08-09 entry below) reports temporal_delta **TSCT 0.42 vs SFT 0.85**, i.e. `ECE_reduction=+0.4350, p=0.0099`.

**Our TSCT arm reproduces theirs almost exactly: 0.4506 vs their 0.42.** Our SFT arm does not: 0.4552 vs their 0.85.

An SFT baseline collapsed to `[CONFIDENT]` would assert 0.95 against ~8% accuracy -> ECE ~0.87, which is their 0.85. Our SFT does not collapse; it emits `[TEMPORAL_HEDGE]` correctly 99.97% of the time. So the most parsimonious reading is that **the reported effect is the distance between a broken baseline and a working model, not the contribution of TCL.** Train both arms correctly and the gap disappears.

Stated carefully, because this contradicts a written Discussion section: n=1 seed, a different base model (Qwen2.5-7B vs their LLaMA-3 8B), and our reimplemented loss with inferred `L_over`/`L_under`/`R_hedge` formulas. This is strong enough to act on and not strong enough to publish as-is. **Replication across seeds is the priority.**

### 3. The real calibration problem is structural, and TCL cannot reach it

On the test set the model asserts mean confidence **0.478** while being right **2.7%** of the time. But its hedge *classification* is 99.9% correct -- it knows these facts are volatile. The failure is that the confidence scalar bolted to `[TEMPORAL_HEDGE]` is 0.45, fixed by fiat, and never validated against realised accuracy.

**No token in the scheme can fix this.** The floor is `[UNKNOWN]` at 0.10, which would still give ECE ~0.073 -- and `[UNKNOWN]` appears **zero times in the training data**, so the model cannot learn to emit it at all. That traces to the unfinished Week-2 task in `docs/STATUS.md` (`[UNKNOWN]` generation via deployment-date offset sampling, assigned to Aarav).

The achievable ECE ceiling is set by the confidence-scalar scheme and a data gap, not by the calibration loss. This is the most consequential finding of the exercise and it reframes what the paper can honestly claim.

### 4. Where the method does work

On stable facts the model scores EM 0.86-0.88 and ECE ~0.15 -- it knows immutable facts and says so. The hedge-token approach is sound where the model has real knowledge; the temporal set is brutal because the answers genuinely aren't there.

### 5. TCL's measurable cost, now quantified

TCL downgraded 3 of 49 stable facts to `[COND_CONFIDENT]` (hedge accuracy 1.000 -> 0.939, ECE 0.152 -> 0.160). Small, real, and in the predicted direction -- the over-hedging price of the reduced overconfidence seen in training. Detectable *only* on the stress set, which is exactly why temporal-delta test alone is insufficient.

## 2026-08-11 -- Generation script built; two structural problems with the evaluation as designed

**Built `src/training/generate_predictions.py`** -- the missing link. Loads an adapter, rebuilds the exact LoRA structure it was trained with, greedy-decodes answer + hedge, and emits records in the canonical shape `adapt_predictions.py` documents, so output drops into Jason's pipeline unmodified. Runs at ~0.24 s/example; fallback rate 0.000, i.e. the model emits hedge tokens in free generation rather than only under teacher forcing.

Sanity check on real generations -- exactly the phenomenon the paper is about:

| question | predicted | gold | hedge |
|---|---|---|---|
| CEO of Nike? | Mark Parker | John Donahoe | `[TEMPORAL_HEDGE]` |
| CEO of Mozilla? | Brendan Eich | Mitchell Baker | `[TEMPORAL_HEDGE]` |

Stale-but-once-correct answers, appropriately hedged.

**Problem 1: the previously reported 93.2%/93.4% hedge accuracies are TRAINING-SET numbers.** `run_tcl_mlx.py` evaluates over the same slice it trains on. The paired SFT-vs-TSCT comparison remains valid (both arms measured identically on identical data) but those are not generalisation figures and must not be reported as such. The generation runs now produce the first held-out numbers this project has ever had.

**Problem 2: `temporal-delta`'s test split cannot detect over-hedging.** All 3,622 test rows are `[TEMPORAL_HEDGE]` (3,287) or `[COND_CONFIDENT]` (335) -- **zero `[CONFIDENT]`, zero `[UNKNOWN]`** -- because the split is time-partitioned to facts that changed in 2023-2024, and stable facts by definition do not change. So:
- always-`[TEMPORAL_HEDGE]` scores ~90.8% hedge accuracy on this set,
- over-hedging is *structurally invisible* here, and reporting temporal-delta test alone systematically flatters an over-hedging model -- which is precisely the direction TCL pushes (see the entry below).

This is why the stress sets are not optional. `stress_stable_facts.jsonl` (49 immutable facts, all gold `[CONFIDENT]`) is the only over-hedging detector available, and has never been run. Both arms are being evaluated on both sources.

**Emerging concern to confirm once numbers land:** early test-set EM is ~4.6%, while `[TEMPORAL_HEDGE]` asserts 0.45 confidence. If that holds, the model is *still* badly overconfident even when hedging correctly -- the calibrated response for a fact it gets right 5% of the time is nearer `[UNKNOWN]` (0.10). And `[UNKNOWN]` appears **zero times in the training data**, so the model cannot learn to emit it. That traces directly to an unfinished Week-2 task (`[UNKNOWN]` generation via deployment-date offset sampling, assigned to Aarav in `docs/STATUS.md`). If confirmed, the ceiling on achievable ECE is set by a data gap, not by TCL.

## 2026-08-11 -- SFT vs TSCT at 7B: identical accuracy, halved overconfidence. The claim needs restating.

**The team's first-ever paired TSCT-vs-SFT result on a working checkpoint.** Same model, data, seed, steps; the only difference is lambda.

| | SFT-only (lambda=0) | TSCT (0.5/0.5/0.3) |
|---|---|---|
| hedge accuracy | 93.23% | 93.42% |
| lift over majority (0.4995) | +0.4328 | +0.4347 |
| total errors | 542 | 527 |
| **over-confident errors** | **466 (86.0%)** | **356 (67.6%)** |
| under-confident errors | 76 (14.0%) | 171 (32.4%) |
| **mean signed confidence error on errors** | **+0.3598** | **+0.1740** |

**On accuracy TCL does nothing:** +0.19pp, 15 examples of 8,008, inside one standard error (~0.29pp).

**On error asymmetry it does a lot:** mean overconfidence on errors is cut by more than half, and the specific dangerous error -- emitting `[CONFIDENT]` on a fact that should have been hedged -- falls **466 -> 351 (-24.7%)**. The errors TCL adds run the other way (76 -> 170 over-hedges). This is exactly what the asymmetric `lambda_over`/`lambda_under` design is meant to produce, observed in behaviour for the first time.

**Consequence for the paper's framing.** "TCL improves calibration" is measurable in two ways that come apart here. Hedge-token *accuracy* is solved by plain CE and TCL adds nothing to it. What TCL changes is the *direction* of the residual errors. That is a sharper and more defensible claim than a bare ECE reduction, and it is mechanistic rather than a summary statistic -- but it is not what the current draft says. **Anyone reporting only accuracy would conclude TCL is worthless; anyone reporting only ECE would miss why it helps.**

**Caveats, all load-bearing:**
1. **n=1 seed per arm.** The 115-example asymmetry shift dwarfs the 15-example accuracy delta, but seed variance for it is unmeasured. Replicate before this goes in the paper.
2. **This is not ECE.** It is stated confidence vs the volatility *label*, not vs whether the model's *answers* are correct. See the blocker below.
3. **The trade has a cost.** Over-hedging more than doubled. The proposal has an accuracy-regression budget and the Slack table already showed MMLU over-hedging. `stress_stable_facts.jsonl` and `stress_mixed_paragraphs.jsonl` exist precisely to catch this and have never been run.

**Blocker identified for real ECE.** `eval_pipeline.py` requires `predicted_answer` (generated text) and `correct` (was the answer right), not just `predicted_hedge`. Our runs are teacher-forced hedge classification and produce neither. Getting ECE therefore requires a **generation/inference script**: load adapter -> generate answer + hedge on a benchmark -> emit the canonical format in `adapt_predictions.py`. That script is the single missing link in the entire chain -- Jason's eval pipeline, stress sets, plot code and results-table generator are all complete and all blocked on exactly this one artifact, which is the same thing Tanvi's 24 broken prediction files were meant to supply.

**Adapters saved** for `sft_only_seed0` and `tsct_seed0_paired`, so this is now unblocked without retraining. Seeds 0-2 predate the save fix and are unrecoverable.

## 2026-08-10 -- First real 7B checkpoint (seed 0), and the CE confound it exposes

**Result (seed 0, Qwen2.5-7B-4bit, 6,006 steps, 8,008 examples, 90 min):**
- **Gradient connectivity holds at 7B:** 1202/1202 probes nonzero, mean 6.82, min 0.18, max 50.38. Never zero.
- **No collapse, and here that finally means something.** Pre-train was the degenerate 100% `[CONFIDENT]` (identical spare rows); post-train spread to 46.1% `[CONFIDENT]` / 6.3% `[COND_CONFIDENT]` / 47.6% `[TEMPORAL_HEDGE]` / 0% `[UNKNOWN]`. Every 0.5B seed pinned to a *single* token; this one did not. `pred_confident_frac` across all steps is centred and spread (0.0/0.25/0.5/0.75/1.0 at 438/1441/2137/1559/431 steps).
- **Calibration terms moved for the first time:** `l_over` -68.1%, `l_under` -66.4%. At 0.5B these were too small and noisy to use as evidence. `r_hedge` rose +82.3%, which reads correctly -- entropy 1.318 -> 1.262 against a 1.386 max, i.e. the model committing to specific hedge tokens rather than sitting at untrained-uniform. Not collapse.
- CE 2.33 -> 0.82.

**The confound, and why it is the important part of this entry.** `tcl_total` is still ~94% CE (0.82 of 0.87); the calibration terms contribute roughly **1%** of the loss after lambda weighting. And the gold hedge token is *part of the supervised sequence*, so plain CE is already training the model to emit the correct hedge token by imitation. **This run therefore cannot distinguish "TCL works" from "CE on the hedge token works."** The hedging behaviour above is real, but attributing it to TCL is unsupported.

This does **not** touch the gradient-connectivity result, which is a claim about the autograd graph (`argmax` severed it, the softmax expectation reconnects it) and is independent of loss weighting. What is unestablished is whether the reconnected signal *changes the outcome*.

**Queued: SFT-only ablation.** Same model/data/seed/steps, `--lambda-over 0 --lambda-under 0 --lambda-hedge 0` -> CE only. This is not a new experiment: `paper/paper_draft.md:156` already specifies "**SFT only**: Base model fine-tuned on TemporalDelta with CE loss only (no TCL). This is the critical ablation isolating the contribution of TCL." Running it gives the team its first real TSCT-vs-SFT data point -- both arms of the comparison the whole paper rests on currently have zero working checkpoints. Writes to `data/prep/tcl_mlx_7b/sft_only_seed0/`.

**Harness note:** the `c_hat_grad_norm` probe deliberately keeps FIXED reference lambdas (0.5/0.5) rather than the run's training lambdas. Under the lambda=0 ablation, weighting the probe by the training lambdas would make it read exactly 0.0 -- the same signature as the `broken` bug, for an entirely unrelated reason. Holding the probe weights fixed keeps a dead gradient path distinguishable from a merely unweighted one.

## 2026-08-10 -- Model-family eligibility audit for the no-resize hedge-token path

**Ask:** Edward asked whether the experiments should cover other open models, since everything so far is small (and floated the Qwen 27B class as a common daily driver).

**Decision: no size sweep for now; audited family breadth instead, which is where the information actually is.** The claim currently under test -- differentiable `c_hat` reconnects the calibration gradient -- is architecture-independent. `argmax` severs autograd in every model and every framework; the result is already exactly 0 vs never 0 at n=4 seeds and holds at 7B. A 27B run cannot falsify it, so it buys no evidence. What *is* model-dependent is the dead-head trap (previous entry), so that got audited directly:

| model | `config.vocab_size` | `len(tokenizer)` | spare rows | no-resize 4-bit path? |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 152064 | 151665 | 399 | yes (current run) |
| Qwen3-8B | 151936 | 151669 | 267 | yes |
| Mistral-7B-Instruct-v0.3 | 32768 | 32768 | **0** | **no -- needs a resize** |
| gemma-3-27b-it | -- | -- | -- | gated, not checked |
| Meta-Llama-3-8B-Instruct | -- | -- | -- | gated, not checked |

**Finding:** the approach is not portable as written. Mistral pads its vocab to exactly the tokenizer length, so there is nowhere to put hedge tokens without `resize_token_embeddings`, which is not well-defined on quantised weights. Any non-Qwen model needs either an unquantised LM head or repurposing existing rare tokens instead of adding new ones. This belongs in the paper's limitations -- it is a constraint on the method, not just on our harness.

**Also note the two gated models include LLaMA-3 8B, which the proposal actually specifies.** We are running Qwen2.5-7B as a stand-in; that substitution needs justifying in the paper regardless. Getting HF access approved for `meta-llama/Meta-Llama-3-8B-Instruct` is the cheapest way to close that gap and is worth starting now since approval is not instant.

**Operational constraint:** disk is at 96% (37GB free and falling during the 7B run). A 27B 4-bit pull is ~16GB. Not impossible, but it should be a deliberate choice, not a casual one.

## 2026-08-10 -- Scaled TCL to 7B locally via MLX; found and fixed a dead-parameterisation trap first

**Before:** open thread said "confirm mlx-lm LoRA works end-to-end on the real 7-8B model locally." `specs/tcl-fix-validation.md` recommended cloud GPU; the 2026-08-09 entry below corrected that toward local MLX but marked it unproven.

**Port:** `mlx_lm.lora`'s CLI trains plain CE only, so it cannot carry TCL. Its Python API (`mlx_lm.tuner.trainer.train`) does take a custom `loss` callable, but that callable only receives `(model, batch, lengths)` -- no channel for per-example `hedge_position` / `c_gold` / `volatile_mask`. Wrote `src/training/mlx_tcl.py` (term-for-term MLX translation of `tcl_loss.py`) plus `src/training/run_tcl_mlx.py` (own loop, same telemetry columns as the 0.5B diagnostic so results are comparable).

**No embedding resize needed.** Qwen2.5 ships `config.vocab_size` 151936/152064 against `len(tokenizer)` 151665, leaving spare rows. The 4 hedge tokens land on ids 151665-151668, which already exist in the quantised embedding matrix -- so the 4-bit path works, where a resize would not be well-defined.

**Found: those spare rows are a trap.** Every unused row in Qwen2.5-7B is the *same padding row*, byte for byte (norm 0.35933 for every id from 151663 to 151700). Under LoRA the LM head is frozen, so all 4 hedge tokens would emit **identical logits for every hidden state** -> softmax exactly uniform -> `c_hat` constant -> argmax pinned to `[CONFIDENT]` forever. That is indistinguishable from the collapse failure mode by looking at outputs, but it is a dead parameterisation, not a training result. An overnight run would have produced confident-looking garbage.
**Fix:** apply LoRA to `lm_head` as well (`LoRALinear.from_base`), giving each hedge row its own trainable low-rank component. Verified: contrast gradient 0.000000 -> 2406.96, and `pred_confident_frac` starts moving off 1.000 within 2 steps.
**Guard:** `assert_hedge_head_is_trainable()` now hard-fails before training. Note the check must use a **linear** contrast (`logit[0] - logit[i]`): the *sum* of the hedge logits has nonzero gradient even with identical frozen rows (all four track the hidden state in lockstep), and any smooth symmetric measure (variance, spread) sits at a stationary point exactly where the logits coincide -- both would pass a dead head. First version of this guard used the sum and did pass it.

**Local vs cloud, resolved: local wins.** 7B/4-bit/LoRA/batch-4 runs at ~1.2 s/step on the M5 Pro, peak RSS ~4.5GB of 48GB. The cloud-GPU recommendation in `specs/tcl-fix-validation.md` was based on scaling the 0.5B fp32 full-fine-tune timing, which badly overestimates the quantised-LoRA path. No cloud provisioning needed for the 8B-class run.

**Launched:** 3 seeds x 3 epochs over an 8,008-example volatility-balanced slice (4000/volatility cap), `mlx-community/Qwen2.5-7B-Instruct-4bit`, ~6,006 steps/seed, writing to `data/prep/tcl_mlx_7b/seed{0,1,2}/`. Given the seed-sensitivity found above, multi-seed is the point -- a single 7B seed would tell us as little as seed 0 alone did at 0.5B.

## 2026-08-09 -- Compute stack correction: MLX changes the local-vs-cloud call

**Before:** Assumed 4-bit QLoRA doesn't work on Apple Silicon (true for the HF/bitsandbytes/PEFT stack, which is CUDA-first), and recommended cloud GPU for the full 8B run.
**Found:** `mlx` 0.31.2 and `mlx-lm` 0.31.3 are already installed locally. `mlx_lm.lora` natively supports LoRA/DoRA/full fine-tuning with quantization built for unified memory, not bolted onto MPS. This is a different tool from the HF stack and doesn't inherit its limitations.
**Status:** Not yet proven at 8B scale -- next step is a short local mlx-lm LoRA run to confirm before committing either way.
**Also found:** no telemetry anywhere in the repo. No wandb/tensorboard hookup, no step-level logging before us. This is likely a contributing reason the calibration-collapse bug went undetected for as long as it did -- nobody had per-term loss curves to look at.

## 2026-08-09 -- TCL gradient-path fix: validated on proxy model

**Before:** All 6 original checkpoints (SFT x3 seeds, TSCT x3 seeds) collapsed to `[CONFIDENT]` on 100% of predictions. ECE ~0.86. Root cause diagnosed by Jason + mentor (documented in `docs/tcl_debugging.md`): `c_hat` computed via non-differentiable argmax + table lookup, severing gradient to the calibration loss terms.
**Fix:** Reimplemented `c_hat` as the differentiable softmax-weighted expectation over hedge-token logits. Built standalone training harness in `src/training/` (tcl_loss.py, hedge_tokens.py, data.py, run_tcl_diagnostic.py) since the original training code lives in Tanvi's separate, unlinked repo and isn't available to us.
**After:** Ran both paths side by side on a 0.5B proxy model (Qwen2.5-0.5B-Instruct), 104 real TemporalDelta examples, 78 steps, same seed.
- `broken` path: gradient norm exactly 0.0 at every step. Hedge predictions collapsed to 99% `[CONFIDENT]` -- reproduces the original bug exactly.
- `fixed` path: nonzero gradient at every step (mean 4.26, max 14.65). Hedge predictions spread across the label distribution (33.7% `[CONFIDENT]` / 66.3% `[TEMPORAL_HEDGE]`) instead of collapsing.
**Caveat:** `L_over`/`L_under`/`R_hedge` formulas are inferred, not recovered from Tanvi's repo (we don't have it) -- documented as assumptions in `tcl_loss.py`. The differentiability fix doesn't depend on getting the exact formulas right, but should be re-validated if/when the real loss code surfaces.
**Where:** committed + pushed to `origin/tcl-fix-validation` on our fork (`edward-lcl/temporal-self-consistency`). Full writeup: `specs/tcl-fix-validation.md`.

## 2026-08-09 -- Repo access resolved via fork

**Before:** Only READ access to `jasontae/temporal-self-consistency` (public, not a fork, can't push).
**Fix:** Forked to `edward-lcl/temporal-self-consistency`. Cloned locally to `/Users/edward/Projects/temporal-self-consistency` (moved here from an initial wrong location inside `algoverse-foundry/` -- corrected same day). `origin` = our fork (push access), `upstream` = Jason's repo (read-only, for pulling their updates). We do not push to upstream.

## 2026-08-09 -- Found: repo's own docs contradict the Google Doc's Results/Discussion

**Found:** The Google Doc's Paper tab already contains a written Discussion analyzing a specific finding (3-seed vs 8-seed ECE flip, p=0.0099 vs p=0.494) as if it were real. The repo's own `paper/paper_draft.md` header says "Results / Discussion / Conclusion pending model results," and `docs/STATUS.md` says checkpoints are broken and plots are placeholder data. These can't both be true.
**Decision (Edward):** Do not ask Jason where the Doc's numbers came from. Re-derive everything independently on our own checkpoints once training works, then compare against Jason's numbers afterward.

## 2026-08-08 -- Joined project, initial landscape review

Added to the team by Kiran (mentor) via Slack, closing out ~2026-08-20. Read the Google Doc (Final Proposal, Paper, Paper Outline, Implementation Instructions, Project Plan/Tasks tabs) and the eval-pipeline repo. Findings: eval pipeline (`src/evaluation/eval_pipeline.py`) is complete and solid; TemporalDelta dataset (14,206 records, contamination-audited, time-partitioned) is real and hosted on HuggingFace (`jasontae/temporal-delta`); two baselines (self-consistency, temperature scaling) already have real numbers just not yet transcribed into the paper; label smoothing and RAG baselines are genuinely unrun; all 6 training checkpoints are broken (see above). Full state captured in `specs/tsct-project-state.md`.

---

## Open threads (carry forward, update as resolved)

- [x] ~~Confirm mlx-lm LoRA works end-to-end on the real 7-8B model locally, at real dataset scale~~ -- **done 2026-08-10.** ~1.2 s/step at 7B/4-bit/batch-4, ~4.5GB peak. Local is viable; no cloud needed. Required porting TCL to MLX (`mlx_lm.lora` CLI is CE-only) and adapting `lm_head` with LoRA.
- [ ] **Amend `specs/tcl-fix-validation.md`** -- its hedge-collapse section and verdict overclaim from a single seed; the collapse result inverts at seed 3 (see 2026-08-10 entry). The gradient-connectivity result is unaffected and still stands.
- [ ] Read out the 3-seed 7B run in `data/prep/tcl_mlx_7b/` once it finishes -- seed 0 is in (gradient holds, no collapse); still need seeds 1-2 to know whether the post-train hedge distribution is stable at 6,006 steps or still seed noise as it was at 78.
- [ ] **Compare SFT-only vs TSCT at 7B seed 0** once the queued ablation lands. If CE-only reproduces the same hedge distribution, TCL is contributing nothing measurable and the paper's central claim needs rethinking before 08-20.
- [ ] Consider whether lambda values need raising -- calibration terms are ~1% of total loss, which may be too weak for TCL to shape behaviour regardless of whether the gradient path is connected.
- [ ] Get access to Tanvi's training repo (or her sign-off that our reimplementation is a fair substitute) -- not yet requested.
- [ ] Wire up real logging (wandb or similar) to any full training run so per-term loss curves exist this time.
- [ ] Label smoothing SFT baseline -- unrun.
- [ ] RAG baseline -- unrun, optional per proposal.
- [ ] Transcribe existing self-consistency + temperature-scaling baseline numbers into the paper's Results table.
- [ ] Reconcile our re-derived numbers against the Doc's existing (unverified) Discussion section, once we have real checkpoints.
- [ ] Framing/positioning against the current frontier (temporal calibration / verbalized confidence literature, last 1-2 years) -- not started, flagged by Edward as a major open task.

## 2026-08-09 -- Telemetry lessons pulled from Bias-Steering ("No Bench") project, and a likely resolution to the Discussion-numbers mystery

**Ask:** Edward asked us to derive lessons from how the Bias-Steering team did their analysis/logging/telemetry (he found it notably better than typical) and apply them to TSCT so we log the right things going in, not after the fact.

**Source:** `/Users/edward/Projects/Algoverse-Bias-Steering`, specifically `docs/VERIFICATION_2026-08-07.md` and `docs/THE_CORRECT_PROBLEM.md`.

**Lessons, generalized:**
1. Keep raw per-record artifacts, not just aggregate CSVs/summaries. Their 2025 headline result only survived scrutiny because someone could recount it from raw response pickles against the summary CSV, row by row. If we only ever save aggregate `eval_results.json`, a bug like the collapse could hide behind a plausible-looking mean.
2. Watch for silently-blended data. Their response logs were cumulative-append (log N = every prior model's records plus the new one), and a naive load blended models together into numbers that matched nothing. Any of our per-checkpoint or per-seed logs need to be self-describing (model/seed/step recorded *inside* the record) rather than inferred from filename or file position.
3. State denominators explicitly. "96% opinionated" was actually 96/96 -- a ceiling effect, not a rate close to 100%. Every rate we report (hedge %, ECE, EM) should carry its n visibly, not just a bare percentage.
4. Log at the granularity that can distinguish *causes*, not just outcomes. Their per-layer vector norms (varying up to 1400x across layers) explained a whole autumn of "coefficient chaos" that had been treated as arbitrary. For us: log per-step gradient norm AND per-loss-term values (L_over, L_under, R_hedge, base LM loss separately) AND the hedge-token prediction distribution -- not just total loss. This is what let us catch the broken/fixed gradient difference on the proxy model; the real run should keep the same resolution.
5. Strip known scaffolding/artifacts before scoring, and check whether it changes the numbers. Their stored responses were full of un-stripped chat-template tokens and a truncated prompt echo, undermining confidence in what the judge actually scored. Worth a similar check on our hedge-token extraction once real generations exist.
6. Run a real extraction-variance check. Two of their "independent" runs turned out to be byte-identical copies -- they had zero estimate of how much a result moves under reseeding. Our 3-seed (and originally proposed 8-seed) design already does this; keep it, don't collapse it to save compute.
7. Apply an ordered, first-match-wins validity screen before treating one aggregate number as a capability/effect claim -- ask "what else could produce this number" before reporting it as a result. Directly relevant to the reconciliation below.

**Action for TSCT training run:** wire up wandb (or equivalent) logging total loss + each loss term + gradient norm + hedge-distribution per step, per seed, self-tagged records (not filename-inferred), and keep raw per-example predictions alongside any aggregate metrics file. This was the concrete gap identified earlier (no telemetry existed on any past run) -- this is what we log instead of print statements once the local mlx-lm run starts.

**Colab notebook:** the link Edward shared requires Google sign-in; plain fetch can't read it (`web_fetch` hit the Google auth wall, no content). Need an exported `.ipynb`/PDF or pasted cells to actually review it.

---

## 2026-08-09 -- Likely source of the Google Doc's Discussion numbers found, and a new contradiction to resolve

**Found:** Edward pasted a Slack message ("pre-benchmark results," posted end of last month) containing a full per-benchmark SFT-vs-TSCT results table plus a Bonferroni significance line: `TSCT vs sft: ECE_reduction=+0.4350, p=0.0099, d=+3.767, sig=False`. The `p=0.0099` matches, exactly, the number already sitting in the Google Doc's Discussion section that we flagged earlier as unexplained (see the 2026-08-09 "repo's own docs contradict" entry above). This is very likely the actual source of the Doc's numbers -- a real analysis someone ran and posted in Slack, not a fabrication. Good news: mystery substantially resolved, and without needing to ask Jason (Edward found it independently via his own Slack search).

**New contradiction this opens:** the Slack table shows TSCT actually hedging (e.g. MMLU: 89.1% hedged vs SFT's 16.9%) and producing lower ECE than SFT on every benchmark (0.42 vs 0.85 on temporal_delta, etc.) -- i.e., a model that is *not* collapsed to 100% `[CONFIDENT]`. That directly conflicts with `docs/STATUS.md`'s claim (checkpoints emit `[CONFIDENT]` on 100% of predictions) and with what we've been treating as ground truth for why training is broken.
**Also notable, independent of the collapse question:** even in this healthier-looking run, TSCT fails the proposal's accuracy-regression criterion on MMLU and FreshQA (-3.75pp and -4.83pp against a 1-2pp budget) and over-hedges on MMLU specifically (should be ~100% CONFIDENT since MMLU is fully stable, actually only 10.9% CONFIDENT). So even the "good" run has a real, separate problem: calibration improved by hedging indiscriminately, not by hedging *correctly*.
**Not yet resolved:** whether this Slack table is from an earlier run that later regressed into the all-CONFIDENT collapse, whether it's a different checkpoint set than the 6 we were told are broken, or whether "checkpoints broken" in STATUS.md was itself describing a different, later state. Applying lesson #7 above rather than guessing -- need to identify which checkpoint(s) produced this table and when, before treating either data point as settled.
**Next action:** ask Edward whether he has more from that Slack thread (timestamps, which checkpoint/run ID, whether Tanvi described what changed after posting it) before we draw conclusions from it.
