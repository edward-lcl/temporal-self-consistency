# TSCT Claims Ledger

_Every claim this project might make, with the evidence behind it, what would
falsify it, and what is still missing. The CHANGELOG is chronological -- what
happened, when. This is the same content indexed by **claim**, so it can be
read as "why we believe what we believe."_

Started 2026-08-11. **Update this whenever a claim's evidence changes.**

## How to read a status

Adapted from the adjudication protocol in _What Makes a Terminal-Bench Task
Hard?_ (Frontier-Bench 0.1 audit), which separates a number from what that
number can support. The key discipline borrowed: an **ordered validity screen**
with a first-matching rule, and an explicit `UNCERTIFIED` outcome.

`UNCERTIFIED` is **not** "halfway between true and false." It means the
available evidence is not enough to decide, and it must not be silently
converted into either. That paper's framing: _"An indeterminate case is not
halfway between genuine-hard and fake-hard; it is a case where the available
evidence is not enough."_

| status | meaning |
|---|---|
| `ESTABLISHED` | Replicated, controlled, and survives the checks listed. |
| `SUPPORTED` | Consistent evidence, but not replicated or not controlled. |
| `CONTESTED` | Our evidence conflicts with an existing claim in the project. |
| `REFUTED` | We believed it; evidence says otherwise. Recorded, not deleted. |
| `UNCERTIFIED` | Evidence insufficient to decide. Needs a named, specific test. |

## Frozen provenance

Every quantitative claim below is tied to this snapshot. Re-derive the pins if
any of them move.

| component | pin |
|---|---|
| repo commit | `1484a6887d33ec231edd4400bd4223ea05937f11` |
| dataset `jasontae/temporal-delta` | `8763b5be0f1e49f16758699bbde2079fcd862b99` |
| model `mlx-community/Qwen2.5-7B-Instruct-4bit` | `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed` |
| framework | mlx 0.31.2 / mlx-lm 0.31.3, Apple M5 Pro 48GB |
| LoRA | rank 8, scale 20.0, 16 layers + `lm_head` |

---

# A. Claims about the TCL mechanism

## A1. The original `c_hat` computation severed the calibration gradient
**`ESTABLISHED`**

Evidence: isolated gradient probe on `lambda_over*L_over + lambda_under*L_under`.
`broken` (argmax + table lookup) gives **exactly 0.0** at 78/78 steps in each of
4 seeds at 0.5B, and the `fixed` softmax-expectation path gives nonzero at
78/78 in all 4 (means 3.69-4.47). At 7B, 1202/1202 probes nonzero across 3 seeds.

Why it is strong: this is a property of the autograd graph, not of a model or
dataset. `argmax` is non-differentiable everywhere.

Falsified by: any `broken`-mode run showing nonzero gradient, or any `fixed`-mode
run showing exactly zero.

Missing: nothing. This one is done.

## A2. The fix prevents hedge-token collapse
**`REFUTED` at 0.5B / `UNCERTIFIED` at 7B**

`specs/tcl-fix-validation.md` claims `fixed` "does not collapse to a single
token," from seed 0 alone. Across 4 seeds it does not hold: seed 1 collapses to
100% `[TEMPORAL_HEDGE]`, and **seed 3 collapses to 100% `[CONFIDENT]`** -- the
exact pathology the fix should prevent -- while `broken` at that seed fails to
collapse at all. Both directions invert.

At 7B no arm collapses and the distribution is stable across 3 seeds (spread
<=4.7pp), but we cannot attribute that to the fix without a `broken` arm at 7B,
which was never run.

Missing: a `broken` arm at 7B, or withdrawal of the claim. **`tcl-fix-validation.md`
still asserts the 0.5B version and needs amending.**

## A3. Padded-vocabulary models silently break hedge-token training
**`ESTABLISHED`**

Qwen2.5's spare embedding rows are all the same padding row (norm 0.35933 for
every id 151663-151700). With a frozen LM head the 4 hedge tokens emit identical
logits for every hidden state: softmax exactly uniform, `c_hat` constant, argmax
pinned to `[CONFIDENT]`. Indistinguishable from collapse by inspecting outputs.
Fixed by adapting `lm_head` (or tied `embed_tokens`) with LoRA: contrast gradient
0.000000 -> 2406.96.

Generalisation audit: Qwen2.5-7B 399 spare rows, Qwen3-8B 267, **Mistral-7B-v0.3
zero** (needs a resize, undefined on quantised weights). Gemma-3-27b and
Llama-3-8B gated, unchecked.

Note for anyone re-implementing the guard: it must use a **linear** contrast
(`logit[0] - logit[i]`). The logit *sum* has nonzero gradient even with identical
frozen rows, and any smooth symmetric measure (variance, spread) sits at a
stationary point exactly at the degenerate configuration. Both pass a dead head.
Our first version used the sum and did.

---

# B. Claims about TCL's effect

## B1. TCL reduces ECE relative to SFT
**`REFUTED`** (at this snapshot, n=1 seed)

| arm | temporal-delta test ECE | stress_stable ECE |
|---|---|---|
| SFT-only (lambda=0) | 0.4552 | 0.1520 |
| TSCT | 0.4506 | 0.1602 |

Difference on the headline benchmark: **0.0046**. On stable facts TSCT is
slightly *worse*. See B3 for why the benchmark cannot resolve this anyway.

Missing: seed replication. But see B3 -- replication will not rescue the claim,
because the measurement instrument is saturated at the oracle.

## B2. TCL shifts the direction of residual errors
**`SUPPORTED`** (n=1 seed, training-set measurement)

At equal hedge accuracy (93.23% vs 93.42%), mean signed confidence error on
errors falls **+0.3598 -> +0.1740**, and `[CONFIDENT]`-instead-of-hedge errors
drop **466 -> 351 (-24.7%)**, with the added errors running the safe direction
(76 -> 170 over-hedges). Consistent with the asymmetric `lambda_over`/`lambda_under`
design.

Caveats: measured on the **training slice** (`run_tcl_mlx.py` evaluates over the
data it trains on), n=1 seed. Held-out confirmation exists but is small: TSCT
downgraded 3 of 49 stable facts to `[COND_CONFIDENT]`.

Missing: seed replication, and a held-out measurement on a volatility-balanced
set. This is currently the strongest *positive* claim we have about TCL.

## B3. ECE on temporal-delta test measures base-rate matching, not calibration
**`ESTABLISHED`** — and this supersedes B1

Controls the project had never run, following the TB3 protocol
(reference-solution run + empty-solution controls):

| policy | test ECE | requires any model capability? |
|---|---|---|
| **ORACLE** — always emit the *gold* hedge | **0.4504** | perfect classifier |
| TSCT (actual) | 0.4506 | — |
| SFT (actual) | 0.4552 | — |
| constant `[CONFIDENT]` | 0.9227 | **no** |
| constant `[COND_CONFIDENT]` | 0.7227 | **no** |
| constant `[TEMPORAL_HEDGE]` | 0.4227 | **no** |
| **constant `[UNKNOWN]`** | **0.0727** | **no** |

Two findings, both decisive:

1. **TSCT is already at oracle.** 0.4506 vs 0.4504. A *perfect* volatility
   classifier scores no better. There is nothing left for any method to win on
   this benchmark -- it is saturated at the ceiling of what the token scheme
   permits.
2. **A constant that ignores the input beats the oracle 6x.** Always-`[UNKNOWN]`
   scores 0.0727. In TB3's taxonomy this is **exploit-only-passable**: the metric
   is passable by a route requiring no capability at all.

Mechanism: test EM accuracy is 2.7%, so ECE is minimised by whichever fixed
scalar sits nearest 0.027. `[UNKNOWN]`=0.10 wins. No model behaviour involved.

**Therefore no ECE number on this test set is interpretable without these
controls reported alongside it.** A reviewer who runs this sinks any ECE-based
claim in the paper.

## B4. The Doc's reported +0.4350 ECE reduction is a broken-baseline artifact
**`CONTESTED`** (our evidence conflicts with the Doc's Discussion; n=1 seed)

Slack table: temporal_delta **TSCT 0.42 vs SFT 0.85**, `ECE_reduction=+0.4350,
p=0.0099`.

Our TSCT arm reproduces theirs (0.4506 vs 0.42). Our SFT arm does not
(0.4552 vs 0.85). Read against the control table in B3:

- their SFT 0.85 ~ constant-`[CONFIDENT]` control (0.9227) — the signature of a
  model collapsed to `[CONFIDENT]`, which `docs/STATUS.md` says all 6 checkpoints
  were;
- their TSCT 0.42 ~ constant-`[TEMPORAL_HEDGE]` control (0.4227).

So the reported effect is close to **the distance between two constant policies**,
neither of which requires learning. The most parsimonious reading is that the
comparison was a broken baseline against a working model.

Confounds preventing `ESTABLISHED`: n=1 seed, different base model
(Qwen2.5-7B vs LLaMA-3 8B), our reimplemented loss with inferred `L_over`/
`L_under`/`R_hedge` formulas, unknown training config on their side.

Missing: seed replication; ideally their checkpoint or its predictions.
**This is the most consequential and least replicated claim in the ledger.**

## B5. Both arms have real, equal discriminative skill
**`REFUTED`** — the apparent skill is a between-dataset artifact

Initially recorded as `SUPPORTED` on pooled test+stable AUROC (SFT 0.6700, TSCT
0.6704, against 0.5 for any constant). Stratifying kills it:

| | SFT | TSCT |
|---|---|---|
| AUROC, test+stable pooled | 0.6700 | 0.6704 |
| AUROC, test only | 0.5026 | 0.5300 |
| **AUROC within relation type** (n-weighted, n≈3400) | **0.5000** | **0.4997** |

Per-relation, both arms, every relation with meaningful n:

| relation | n | SFT | TSCT |
|---|---|---|---|
| head_of_gov | 1349 | 0.500 | 0.500 |
| officeholder | 908 | 0.500 | 0.499 |
| chairperson | 708 | 0.500 | 0.500 |
| ceo | 216 | 0.500 | 0.498 |
| parent_org | 125 | 0.500 | 0.500 |

The 0.67 was entirely the model separating the *stable-facts stress set* from the
*temporal-delta set* — a distinction between two corpora, not between examples.
Within a relation type the confidence is essentially constant, so AUROC is 0.5 by
construction (all ties).

**Interpretation: the model learned a lookup table from relation type to hedge
token.** "CEO questions get `[TEMPORAL_HEDGE]`" is learnable from the question
template alone, with zero knowledge of the specific entity. It has no ability to
tell which *particular* facts it will get right.

This subsumes several earlier puzzles. TSCT ≈ SFT because both learn the same
lookup. TCL cannot improve calibration because no per-example signal is being
learned to calibrate. And ECE reduces to base-rate matching (B3) because a
confidence that is constant per relation *is* a constant policy, merely
stratified.

**Consequence for D2:** fitting the confidence scalars to realised accuracy would
minimise ECE by implementing the B3 exploit more precisely. It should be reported
as a **baseline that requires no capability**, never as a method.

## B6. Fine-tuning does not improve factual accuracy; it installs the hedge lookup
**`SUPPORTED`** (n=1 seed per arm, format-matched)

D1 baseline. Untuned `Qwen2.5-7B-Instruct-4bit` vs both fine-tuned arms, on
identical questions:

| arm | test EM | test contains | stable EM | mean words |
|---|---|---|---|---|
| **base** | **0.0392** | 0.0425 | **0.7551** | 3.5 |
| SFT-only | 0.0226 | 0.0229 | 0.8571 | 2.3 |
| TSCT | 0.0273 | 0.0273 | 0.8776 | 2.4 |

Fine-tuning **lowers** volatile-fact accuracy (0.0392 -> 0.0273, **-1.19pp, -30%
relative**) and **raises** stable-fact accuracy (0.7551 -> 0.8776, +12.2pp, but
n=49 so ~2 SE). The volatile regression sits right at the edge of the proposal's
stated 1-2pp accuracy-regression budget.

Combined with B5: fine-tuning's contribution is the relation->hedge lookup table,
bought at roughly a point of volatile-fact accuracy. It did not teach the model
anything factual.

**Measurement note — a confound that nearly produced a false headline.** The
first D1 pass ran the base model unprompted. It answered in prose (mean **45.1
words** vs 2.6 for fine-tuned arms), giving 17x the surface area for a
containment match, and reported base at 10.1% vs 3.9% -- a fake 2.6x advantage.
The obvious correction (require gold to be >15% of generated words) over-corrects
in the opposite direction, since a two-word name cannot be 15% of a 45-word
answer; it put base at 0.3%. Neither number was real. Fixed at the source with a
brevity instruction (`--answer-hint`), bringing base to 3.5 words and making EM
and containment agree for every arm. The corrected gap is 1.4x, not 2.6x.
Recorded because the failure mode -- comparing a format-tuned model against an
untuned one and reporting the format difference as a knowledge difference -- is
exactly the class of error this ledger exists to catch.
`base_test_unprompted_partial.jsonl` is retained as the robustness record.

---

# C. Claims about the data and task

## C1. `[UNKNOWN]` is untrainable — it appears zero times in the data
**`ESTABLISHED`**

Train slice label composition (n=8008): `[TEMPORAL_HEDGE]` 50.0%, `[CONFIDENT]`
43.8%, `[COND_CONFIDENT]` 6.2%, **`[UNKNOWN]` 0.0%**. Validation and test contain
none either.

The model cannot emit the one token that would minimise ECE (B3). Traces to an
unfinished Week-2 task in `docs/STATUS.md` (`[UNKNOWN]` generation via
deployment-date offset sampling, assigned to Aarav).

## C2. temporal-delta's test split cannot detect over-hedging
**`ESTABLISHED`**

All 3,622 test rows are `[TEMPORAL_HEDGE]` (3,287) or `[COND_CONFIDENT]` (335) --
no `[CONFIDENT]`, no `[UNKNOWN]` -- because the split is time-partitioned to facts
that changed in 2023-2024. Always-`[TEMPORAL_HEDGE]` scores ~90.8% hedge accuracy.
Over-hedging is structurally invisible. The stress sets are therefore mandatory,
not optional; TCL's over-hedging cost was detectable only there.

## C3. Most wrong answers are fabrications, not stale facts
**`SUPPORTED`** (bounded, not exact)

Rescoring test predictions against the dataset's own multi-epoch chains
(3,186 of 10,864 entity-property keys have >1 recorded value):

| | SFT | TSCT |
|---|---|---|
| current value (correct) | 2.3% | 2.7% |
| prior value — once true, scored wrong | 11.8% | 11.2% |
| neither — in no recorded epoch | **85.9%** | **86.1%** |

~97% of stale answers appear in the train split. Bound: only 29% of keys have
multi-epoch coverage, so "neither" mixes true fabrications with real former
values the dataset never recorded. **12% is a floor on staleness, 86% a ceiling
on fabrication.**

Measurement gap: "recited a former officeholder" and "invented a person" score
identically under EM, though they are different epistemic states.

## C4. Contamination has not been audited for *our* base model
**`UNCERTIFIED`**

The contamination audit was run for the team's setup, not for Qwen2.5-7B, whose
cutoff may cover 2023-2024. Weak reassurance: 2.3-2.7% current-value accuracy
suggests it does not know the post-change values. Not a substitute for an audit.

---

# D. Open — no evidence either way

| # | question | why it matters | cost |
|---|---|---|---|
| ~~D1~~ | **RESOLVED** -> see B6. Fine-tuning lowers volatile EM 30% relative and raises stable EM; it adds the hedge lookup, not knowledge. | — | done |
| D2 | Do the confidence scalars fit realised accuracy? | 0.95/0.75/0.45/0.10 are assigned by fiat. Realised: 2.7% volatile, 87.8% stable. Fitting them is where ECE actually lives — and it is a **contribution**, not a patch. | ~1 h |
| D3 | Does B4 survive seeds? | Most consequential, least replicated claim here. | ~3.6 h |
| D4 | What is in the 86% "neither" bucket? | Distinguishes "no knowledge" from "corrupted knowledge". Mechanistic. | ~1 h |
| D5 | Does TCL help on a **volatility-balanced** held-out set? | Every held-out measurement so far is on a set where one class is 90.8%. | needs a new split |
| D6 | **Does within-relation discrimination ever exceed 0.5?** (Edward, 2026-08-11) | B5 says our model is a relation→hedge lookup table. The sharp question is whether *any* model does better, or whether this task is only ever solvable by template matching. This is the pattern-matching-vs-reasoning question made measurable, and B5 gives it a clean threshold: within-relation AUROC > 0.5. | see D7 |
| D7 | **Temporal sweep across model generations** (Edward, 2026-08-11) | Hold parameter count fixed (~7-8B) and vary release date: a 2024, an early-2025, a mid-2025, and a mid-2026 model. Two questions at once — (a) does within-relation discrimination (D6) improve with generation, and (b) does the knowledge-cutoff boundary itself move, which changes which facts are "post-cutoff" per model and is a confound for every result in this ledger. Note the cheap version needs **no training**: zero-shot answer accuracy per generation already tests (b), and D1's harness covers it. | high — several model downloads (37GB free) + eval per model; the zero-shot half is much cheaper |

---

# E. Standing methodological commitments

Adopted from the TB3 audit protocol. These are cheap and they are what makes the
numbers defensible.

1. **Report controls with every metric.** Oracle run and constant-policy controls,
   always. B3 exists only because we ran them.
2. **Pin the snapshot.** Dataset sha, model sha, repo commit on every quantitative
   claim.
3. **Separate calibration from discrimination.** ECE and AUROC answer different
   questions; a constant policy scores well on one and 0.5 on the other.
4. **State denominators.** "93% hedge accuracy" was a training-set number on a
   set where one class held 50%. Both facts belong next to it.
5. **Keep raw per-record artifacts.** `raw_generation` and self-describing
   provenance in every prediction row, so any aggregate can be recounted.
6. **Never convert `UNCERTIFIED` into a result.** Insufficient evidence is its own
   outcome.
