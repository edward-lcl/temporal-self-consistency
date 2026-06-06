# Week 3+ Status — Jason's Workload

Updated after burning through all unblocked tasks across Weeks 3, 4, and 5.

## ✅ Done (deliverables ready)

### Data prep
- `mmlu_stable_subset.jsonl` — 4,048 time-stable MMLU questions (regression check set)
- `stress_test_extended_horizon.jsonl` — 3,097 facts changing 2024–2025 (18–36 mo post-cutoff)
- `stress_stable_facts.jsonl` — 49 hand-curated stable facts (over-hedging detector)
- `stress_mixed_paragraphs.jsonl` — 60 passages mixing stable + volatile claims (selective hedging test)

### Evaluation infrastructure
- `eval_pipeline.py` — all 6 metrics (ECE, EM/F1, volatility breakdown, Bonferroni, volatility discrimination, temporal generalization gap)
- `full_analysis.py` — runs all 24 Tanvi prediction files through the pipeline
- All 6 publication-quality plots generated via `paper_plots.py` (placeholder data, swap real values when checkpoints work)
- `generate_results_table.py` — outputs markdown, LaTeX, and CSV versions of the 8-condition results table

### Hedge quality framework
- `hedge_quality_rubric.md` — 1-5 Likert scale rubric for MTurk annotators
- `hedge_quality_rubric.py` — automated scoring matrix for supplementing human eval

### Paper writing
- `proposal_updates.md` — all dataset/methodology revisions ready to paste into the Google Doc

---

## ⛔ Blocked (waiting on team)

### From Tanvi
The 24 prediction files she sent are broken — every output is `[CONFIDENT]`. Need her to:
1. Verify TCL λ values are non-zero in the training script
2. Confirm hedge tokens are properly connected to the model's output layer (resized embeddings working)
3. Check decoding strategy isn't greedy on the hedge token position
4. Re-run inference on all 6 checkpoints once fixed

Then deliver predictions on all 6 benchmark sets:
- `temporal_delta_test.jsonl` ✓ in hand
- `freshqa.jsonl` ✓ in hand
- `mmlu_stable_subset.jsonl` ✓ ready to send
- `stress_test_extended_horizon.jsonl` ✓ ready to send
- `stress_stable_facts.jsonl` ✓ ready to send (newly built)
- `stress_mixed_paragraphs.jsonl` ✓ ready to send (newly built)

Plus eventually the 4th condition: SFT + TCL + DPO checkpoint (depends on Aarav's MTurk pairs).

### From Logan
- Label smoothing SFT — needs to be a separately fine-tuned model, not confidence rescaling
- Temperature scaling — needs to use actual model logits, fitted T on val set
- RAG baseline — not yet started, optional per proposal
- Test set contamination audit — completed for val, need confirmation it ran on test too

### From Aarav
- Entity blacklist (Wikipedia pageview API filter on test set) — handed off to him in Week 1
- `[UNKNOWN]` token generation via deployment-date offset sampling — Week 2 task
- MTurk preference pairs for DPO — Week 3 pilot, Week 5 full collection

---

## 🟡 Pending paper writing (assigned across team)

Per task doc, these are weekly deliverables but unassigned among the 4 of us:
- Abstract — last (after results)
- Introduction — can draft now using proposal motivation
- Related Works — can draft now using proposal references
- Methods — can draft now using proposal methods section
- Results — blocked until checkpoints work
- Discussion — blocked until results
- Conclusion — last

I can knock out Introduction, Related Works, and Methods sections without further team input. Want me to?

---

## Files in `week45_deliverables/`

| File | Purpose |
|---|---|
| `prep_stress_stable.py` | Generates `stress_stable_facts.jsonl` |
| `prep_stress_mixed.py` | Generates `stress_mixed_paragraphs.jsonl` |
| `paper_plots.py` | All 6 paper figures (calibration, ECE bars, etc.) |
| `generate_results_table.py` | Final results table in MD/TeX/CSV |
| `hedge_quality_rubric.py` | Rubric + automated scoring |
| `hedge_quality_rubric.md` | Rubric printed (for MTurk task spec) |
| `proposal_updates.md` | All textual revisions for the Google Doc |
| `full_analysis.py` | (Already shared earlier) full eval runner for predictions |

## Next concrete action items for Jason

1. Send proposal_updates.md to mentor for review
2. Send stress test files (stable + mixed paragraphs) to Tanvi for inclusion in next inference batch
3. Send Logan a final reminder on label smoothing + temp scaling fixes
4. Start drafting Methods / Introduction / Related Works sections of the paper
