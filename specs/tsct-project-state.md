# TSCT Project — State & Handoff Spec

_Written 2026-08-09. Source of truth for any agent picking up this work._

## Project

**Temporal Self-Consistency Training for Reasoning Under Evolving World Knowledge (TSCT)** — NeurIPS-track team project. Trains an LLM (LLaMA-3 8B) to append one of four hedge tokens to every answer, calibrated to fact volatility, so it signals uncertainty about facts that may have changed since training cutoff — without retrieval at inference time.

- `[CONFIDENT]` 0.95 — immutable facts
- `[COND_CONFIDENT]` 0.75 — slow-changing facts
- `[TEMPORAL_HEDGE]` 0.45 — fast-changing facts
- `[UNKNOWN]` 0.10 — beyond reliable knowledge horizon

## Team / context

- Mentor: Kiran N — closing the team out ~2026-08-20, wants NeurIPS submission.
- Jason: eval pipeline + project oversight. Logan: baselines. Tanvi: model training (owns a **separate, unlinked** training repo). Aarav: data/contrastive pairs, MTurk.
- Edward added 2026-08-08 to help with paper + poster.

## Source documents

- Google Doc `1bo_01Jbnilmd0BVxshKMecU12teQWTJJ0RR7uCEfOEk` (account `eluecheelip@gmail.com`) — tabs: Final Proposal, Paper, Paper Outline, Implementation Instructions (`t.sdbavb89uk1r`), Project Plan/Tasks (`t.gj7hsr12m7k6`).
- Eval/data-pipeline repo (upstream, READ-only, public, not a fork): `https://github.com/jasontae/temporal-self-consistency`
- **Our fork** (push access): `https://github.com/edward-lcl/temporal-self-consistency`, cloned to `/Users/edward/Projects/algoverse-foundry/temporal-self-consistency`. `origin`=fork, `upstream`=Jason's repo.
- Dataset (public): HuggingFace `jasontae/temporal-delta`

## What's solid

- Related Work framing (vs RAG, self-consistency, temperature scaling, label smoothing) is coherent.
- TemporalDelta dataset: 14,206 train records, 11 Wikidata SPARQL properties, contamination-audited, time-partitioned (Train 2018-2021 / Val 2022 / Test 2023-2024). Hosted on HF.
- Eval pipeline (`src/evaluation/eval_pipeline.py`): ECE (equal-frequency binning), EM/F1, volatility-split breakdown, Bonferroni correction (α≈0.007 / 7 comparisons), volatility discrimination scoring, temporal generalization gap. Smoke-tested, code complete.
- Two baselines have real numbers already: self-consistency (ECE 0.1654, hedge alignment 0.2308), temperature scaling (ECE 0.2094→0.0170, accuracy 0.979→0.979, T=0.2616).

## What's broken — THE critical blocker

All 6 checkpoints (SFT×3 seeds, TSCT×3 seeds) collapsed to emitting `[CONFIDENT]` on 100% of predictions. ECE ~0.86 (claims 0.95 confidence, ~8% actual accuracy).

**Root cause (already diagnosed by Jason + mentor, documented in `docs/tcl_debugging.md` in the repo):** `c_hat` (confidence fed into the TCL loss) was computed via post-argmax lookup on the selected hedge token — a discrete, non-differentiable operation — instead of as the differentiable softmax expectation over hedge logits. Result: CE loss trains normally (answer tokens unaffected) so SFT looks fine, but the calibration loss terms (`L_over`, `L_under`, `R_hedge`) never receive real gradient — `tcl_total` moved <1% across the entire run.

**Documented fix:**
```python
# correct — differentiable
hedge_probs = F.softmax(hedge_logits, dim=-1)
c_hat = (hedge_probs * conf_scalars).sum(dim=-1)

# broken — gradient cut at argmax
hedge_id = torch.argmax(hedge_logits)
c_hat = HEDGE_CONFIDENCE[hedge_id]
```
Revised hyperparameters from the same doc: epochs 1→3, lr 2e-4→5e-5, lambda_over 1.0→0.5.

**Important gap:** the file this fix lives in, `src/training/tcl_loss.py`, is **not in the eval/pipeline repo** — the README states training/inference are maintained in a separate repo by Tanvi, which we do not have a link to. The diagnostic doc's guidance is complete enough to reimplement independently; we are not blocked on getting Tanvi's repo, just duplicating effort if we don't get it.

## Unresolved discrepancy — do not build on this yet

The Google Doc's Paper tab already contains a written Discussion section citing a specific finding (3-seed vs 8-seed ECE flip, p=0.0099 vs p=0.494) as if it were a real result. This directly contradicts: (a) the repo's own `paper/paper_draft.md`, whose header says "Results / Discussion / Conclusion pending model results," and (b) `docs/STATUS.md`, which says checkpoints are broken and plots are placeholder data. **Decision (Edward, 2026-08-09): do not ask Jason where these numbers came from. Re-derive everything independently, then compare against Jason's numbers after.** Do not treat the Doc's existing Discussion numbers as real until we've reproduced something ourselves.

## Two baselines still fully unimplemented (separate from the checkpoint bug)

- Label smoothing SFT — needs to be a genuinely separately fine-tuned model (CE + ε=0.1), not confidence rescaling of the base model.
- RAG baseline — not started, optional per proposal.

## Local compute available

Apple M5 Pro, 18 cores (6 Super + 12 Performance), 48GB unified memory. `torch` 2.12.0 installed. This is Apple Silicon MPS, not CUDA — most 4-bit QLoRA tooling (bitsandbytes) does not work on MPS. Realistic path is LoRA in bf16/fp16 on the 8B model, slower than the team's likely-original CUDA setup. **Recommended approach: validate the TCL gradient-path fix on a small proxy model (1-3B) locally first, before committing to a full 8B run**, to avoid burning days debugging on slow hardware.

## Immediate next actions (in order)

1. Reimplement TCL loss with the documented fix on a small proxy model; confirm `tcl_total` actually moves and hedge token distribution stops collapsing.
2. Once validated, scale to LLaMA-3 8B / Mistral 7B locally or on cloud compute (TBD which).
3. Run the eval pipeline (already built, in the fork) against our own checkpoints — this reproduces or refutes the Doc's existing Discussion numbers.
4. Implement label smoothing + RAG baselines.
5. Only after (3): write Results/Discussion/Abstract sections for real. Related Work, Methods, Introduction, Future Work, Conclusion can be drafted now, independent of training status.
6. Reconcile our numbers against the Doc's existing (unverified) Discussion section.

## Deadline

Kiran closing the team out ~2026-08-20. Paper + poster both wanted; original ask was same-day (2026-08-09), which is aggressive given the above.
