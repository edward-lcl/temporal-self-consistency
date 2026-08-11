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

## Bottom line as of 2026-08-11

**TCL's only surviving claim is that its gradient path is now connected. Every
behavioural claim tested has failed replication — but B8 turns the diagnosis
constructive: the signal the method needs is already in the model, at AUROC 0.85,
and the discrete hedge-token parameterisation is what discards it.**

| claim | status |
|---|---|
| A1 — the `argmax` bug severed the calibration gradient; the fix reconnects it | `ESTABLISHED` (4 seeds @0.5B, 3 @7B) |
| A3 — padded-vocab models silently break hedge training | `ESTABLISHED` |
| A2 — the fix prevents hedge collapse | `REFUTED` (inverts at seed 3) |
| B1 — TCL reduces ECE | `REFUTED` (0.0046) |
| B2 — TCL shifts error direction | `REFUTED` (holds seed 0, gone seed 1) |
| B5 — the hedge output discriminates per example | `REFUTED` (within-relation AUROC 0.500) |
| B4 — the Doc's +0.4350 is a broken-baseline artifact | `CONTESTED` (n=1) |
| B6 / B7 — fine-tuning adds no knowledge, and causes entity interference | `SUPPORTED` |
| B3, C1, C2, C5, C6 — the benchmark cannot measure the claim | `ESTABLISHED` |
| **C7 — fine-tuning explicitly teaches the wrong answer for 29% of test questions** | **`ESTABLISHED`** (mechanism behind B6/B7) |
| C8 — `exact_match` is a mild floor; the obvious relaxation enriches for name collisions | `ESTABLISHED` |
| **B8 — the model's own logprobs predict its correctness at 0.84 where the hedge scheme is at chance** | **`SUPPORTED`, qualified** (z=13.6 but AP only 3.3x baseline; 35% of positives are train-seen) |

Two independent reasons this setup could not have shown a TCL effect even if one
exists, so the refutations above should not be read as "TCL cannot work":

1. **The instrument is saturated** (B3). A perfect volatility classifier scores
   ECE 0.4504; TSCT scores 0.4506. There is no headroom, and a capability-free
   constant beats both 6x.
2. **There is no per-example signal to calibrate** (B5, B7). The model learned a
   relation-level hedge template and an entity-blind answer pool. A calibration
   loss operating on a quantity that does not vary within a relation has nothing
   to act on.

The honest summary for the paper: the gradient bug was real and is fixed; the
benchmark cannot detect whether fixing it helps; and on this data the model is
doing template matching on both output channels. Demonstrating a TCL effect needs
a different instrument, not more seeds.

**And the paper does not have to end there.** B8 shows the per-example signal
exists in the same model on the same examples — the discrete four-token scheme is
what destroys it. Scaling the power axis (model size, seeds, lambda) does not move
the result; changing the parameterisation does. That converts "this path is a dead
end" into "here is the specific design choice that caused it."

But B8 is a **qualified** positive, and the qualification must travel with it: the
effect is statistically overwhelming (z=13.6) and practically weak (average
precision 0.0913 against a 0.0273 base rate, 3.3x). It establishes that the
architecture discards real information. It does **not** establish that recovering
that information would produce a usable system. Claiming the latter would repeat
exactly the error this ledger exists to catch.

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
**`REFUTED`** — held at seed 0, vanished at seed 1

Recorded as `SUPPORTED` on seed 0 and flagged then as "the strongest positive
claim we have about TCL." The seed-1 pair (D3) removes it.

| | hedge acc | errors | over-confident | mean signed conf error |
|---|---|---|---|---|
| SFT seed 0 | 0.9323 | 542 | 86.0% | +0.3598 |
| TSCT seed 0 | 0.9342 | 527 | 67.6% | **+0.1740** |
| SFT seed 1 | 0.9344 | 525 | 100.0% | +0.5003 |
| TSCT seed 1 | 0.9324 | 541 | 100.0% | **+0.4988** |

| | seed 0 | seed 1 |
|---|---|---|
| mean signed conf error, SFT -> TSCT | **-0.1858** | **-0.0015** |
| over-confident errors, SFT -> TSCT | 466 -> 356 (**-23.6%**) | 525 -> 537 (**+3.0%**) |

At seed 1 the effect is absent and slightly reversed. The confusion matrices show
why: at seed 1 both arms make essentially a single error type,
`[TEMPORAL_HEDGE]` -> `[CONFIDENT]` (SFT 524, TSCT 537), and TSCT produced
**one** over-hedge error versus 170 at seed 0.

Same failure mode as A2: a single-seed behavioural result that does not survive
replication. Between-seed variance in TSCT's error profile dwarfs the
between-arm difference.

Missing: n=2 is enough to withdraw the claim, not enough to bound the variance.
A third seed would characterise it, but nothing here supports the effect.

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

## B7. Fine-tuning taught the answer *vocabulary* without the entity→answer mapping
**`SUPPORTED`** (n=1 seed per arm) — D4

Classifying every test prediction by where the answer string exists in the corpus:

| category | base | SFT-only | TSCT |
|---|---|---|---|
| not any value in the dataset | 86.3% | 47.8% | 44.1% |
| **a real value, but for a DIFFERENT entity** | **7.9%** | **38.1%** | **42.1%** |
| a value for *this* entity (current or prior epoch) | 5.9% | 14.1% | 13.8% |
| the current value per Wikidata | 0.2% | 0.2% | 0.2% |

Fine-tuning raised the share of answers drawn from the training corpus from
**13.8% to ~56%** — and the growth is overwhelmingly in the *wrong-entity*
category, up roughly **5x** from 7.9%.

**Attribution: this is fine-tuning, not TCL.** SFT-only (lambda=0) already shows
38.1% wrong-entity against base's 7.9%; TCL adds a further 4pp. Whatever is
causing the interference is the fine-tuning itself, and turning the calibration
loss off does not avoid it.

**Interpretation: the model memorised the pool of plausible answers without
learning which answer belongs to which entity.** It now retrieves a real
officeholder name from the training distribution and attaches it to the wrong
organisation. That is textbook interference, not knowledge acquisition.

Read with B5 the picture is consistent on both axes:

| output | what fine-tuning learned |
|---|---|
| hedge token | relation-level template (`ceo` -> `[TEMPORAL_HEDGE]`) — B5 |
| answer | corpus-level answer pool, entity-blind — B7 |

Both are distributional. Neither is fact-specific. This is the direct,
mechanistic answer to "pattern matching or reasoning?" for this setup, now with
two independent lines of evidence.

Caveat on reading the base column: base's 86.1% "not in dataset" is **not** a
fabrication rate. The base model answers with real-world knowledge the dataset's
value vocabulary simply does not contain (e.g. Neal Mohan, John Donahoe). Absence
from the corpus is a statement about the corpus, not about correctness — which is
why the Wikidata row is reported separately.

Missing: seed replication; and the same breakdown for SFT-only, to check whether
the interference is caused by TCL or by fine-tuning per se (D3 will supply the
adapters for this).

## B8. The signal the method needs is already in the model — the hedge scheme discards it
**`SUPPORTED`** (n=1 seed, single model) — **this is the constructive finding**

Same model, same 3,622 test examples, same correctness labels. Two confidence
signals compared as predictors of *whether the model's own answer is right*:

| confidence signal | overall AUROC | **within-relation AUROC** |
|---|---|---|
| hedge-token confidence (what TSCT trains) | 0.5300 | **0.4997** |
| mean answer log-probability (free, already there) | 0.8406 | **0.8465** |
| min answer log-probability | 0.8415 | **0.8485** |

**And the untuned model's signal is better still.** Same questions, base model
vs fine-tuned:

| arm | correct | logprob AUROC |
|---|---|---|
| **base (no fine-tuning)** | **142** | **0.8814** |
| TSCT | 99 | 0.8406 |

Fine-tuning made the model less accurate (142 -> 99 correct) *and* degraded the
self-knowledge signal it already had (0.8814 -> 0.8406) *and* added a hedge
output that is at chance (0.4997). Every axis we can measure moved the wrong way.

The sharpest statement of the finding is therefore: **the untuned model already
carries the best calibration signal in the system, and every subsequent stage of
the pipeline degrades or discards it.**

The trained four-token scheme is **at chance** within a relation (B5). The
model's own token probabilities, which cost nothing and require no training,
separate correct from incorrect answers at **~0.85**.

**Not a length artifact.** Stratifying by exact answer-token count and pooling:

| answer tokens | n | correct | AUROC |
|---|---|---|---|
| 3 | 139 | 18 | 0.8101 |
| 4 | 288 | 21 | 0.6909 |
| 5 | 443 | 18 | 0.7931 |
| 6 | 641 | 16 | 0.8159 |
| 7 | 677 | 14 | 0.9265 |
| 8 | 680 | 3 | 0.9542 |
| 9 | 272 | 5 | 0.8899 |

Length-stratified pooled AUROC **0.8612** (n=3,140). High in every stratum.

### The argument structure to use (from _The Recall Debt_, COLM 2026 submission)

That paper argues a *structural* failure: flat memory schemas cannot reach
multi-hop terminals, and it proves this is architectural rather than a tuning
problem by **scaling the power axis and showing a plateau** — BM25 0%, dense 0%,
late-interaction 2.5%, frontier API embedder 22.6% — against **61.6% once the
bridge is made explicit**. Their framing: _"An absolute floor, not a
degradation... A better embedding model moves documents around in the same
similarity space; it does not add the missing relation."_

Our evidence has exactly that shape, and this is a far stronger way to present
B8 than "logprobs score higher":

| axis | intervention | result |
|---|---|---|
| **power** | 0.5B -> 7B | no effect on the calibration claim |
| **power** | 4 seeds @0.5B, 2-3 @7B | no effect (A2, B2 refuted) |
| **power** | lambda 0 -> 0.5/0.5/0.3 | 0.0046 ECE (B1) |
| **structure** | discrete hedge token -> model's own logprob | **0.4997 -> 0.8465** |

Scaling the power axis does not move it. Changing the parameterisation does.
That is the same claim shape, and it is what makes this a design finding rather
than a tuning observation.

### Why this reframes the whole project

The model **already knows what it does not know**, per example, at 0.85. The
architecture then throws that away and replaces it with a 4-way discrete token
whose value is a near-deterministic function of the relation type. Every negative
result in section B follows from that one design choice:

- nothing per-example to calibrate -> the loss cannot bite (B1, B2)
- confidence constant within relation -> chance discrimination (B5)
- confidence constant within relation -> ECE reduces to base-rate matching (B3)

**The diagnosis is not "temporal calibration is impossible." It is "this
parameterisation destroys the available signal."**

### The constructive proposal that follows

1. Replace the discrete hedge vocabulary with a **continuous confidence** derived
   from (or trained against) the model's own answer distribution.
2. Retarget the objective. Predicting *fact volatility* is a relation-level
   template with no per-example variance. Predicting **"does this model know this
   fact"** has real per-example variance and is what calibration means.
3. Keep the hedge tokens only as a **presentation layer** over a calibrated
   scalar, if discrete output is wanted for the interface — not as the thing the
   loss operates on.

### Is it statistically real? Yes. Is it practically useful? No.

Both, and they must be reported together — Edward pushed on this and the answer
splits.

| | value |
|---|---|
| AUROC | 0.8406 |
| Hanley-McNeil SE | 0.0250 |
| analytic 95% CI | [0.792, 0.890] |
| bootstrap 95% CI (200x) | [0.813, 0.873] |
| **z vs 0.5** | **13.6** |
| **average precision** | **0.0913** (baseline 0.0273 = **3.3x**) |
| precision @ 50% recall | **0.113** |

**Statistically it is not marginal at all** — z=13.6. **Practically it is weak.**
At a 2.7% base rate, ranking by logprob and taking enough to recover half the
correct answers still leaves you wrong ~89% of the time. AUROC is
imbalance-insensitive and flatters rare-positive problems; average precision is
the honest lens and it says 3.3x baseline, not a solved problem.

Some of that weakness is inherited: the 2.7% base rate is itself produced by the
broken benchmark (C5), so AP would look different on a set where the model gets a
meaningful fraction right. That is an explanation, not an excuse.

### Validity gate on the positive class

Applying the discipline from _The Recall Debt_ Section 5 (a plausible filter can
*enrich* for artifacts, and a weak judge will wave them through) to our own 99
positives:

| check | count | share |
|---|---|---|
| answer value also appears in the TRAIN split | 52/99 | 52.5% |
| **this exact question appears in the TRAIN split** | **35/99** | **35.4%** |
| "correct" means reproducing a value that is now expired (C5) | 75/99 | 75.8% |

So a third of the positive class is train-seen, and three quarters of "correct"
means *successfully reciting a stale fact*. The signal is partly memorisation
confidence rather than knowledge confidence.

**It survives exclusion**: dropping every train-seen positive gives AUROC
**0.8176** on npos=64, against 0.8406 on all 99. Real signal remains, but B8 is a
qualified positive, not a clean one.

Caveats: n=1 seed, one model, one dataset. Logprob confidence is established in
the literature; the contribution is the **controlled same-model, same-examples
comparison** showing the proposed method at chance where the free signal is not.

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

## C5. The test split's gold answers are expired values
**`ESTABLISHED`** — rate measured by D8. My initial reading of its *consequence* was an overclaim; see the correction below.

Structural fact, from the dataset alone:
- Every test row carries `t_end` of **2023 (1,582)** or **2024 (2,040)**. Not one has
  an open-ended or current value.
- **91.3% of test rows (3,306/3,622) are orphans**: the gold is the last value the
  dataset records for that entity+property, and the dataset itself says that value
  *ended*, with no successor recorded.

So the question is asked in the present tense ("Who is the CEO of X?") while the
gold answer is one the dataset marks as no longer true. Spot checks against known
reality:

| question | gold | base model | actual |
|---|---|---|---|
| CEO of YouTube LLC | Susan Wojcicki | Neal Mohan | Mohan since Feb 2023 — **model correct, scored wrong** |
| Chair of News Corporation | Rupert Murdoch | Lachlan Murdoch | Lachlan since Nov 2023 — **model correct, scored wrong** |

### D8 result (live Wikidata, queried 2026-08-11T10:02Z)

The rate is now measured, and it **confirms the defect while refuting the
consequence I initially drew from it.**

| verdict | pairs | share |
|---|---|---|
| `gold_is_stale` | 2,575 | 80.3% |
| `gold_is_current` | 35 | **1.1%** |
| `indeterminate` (no unambiguous current value) | 595 | 18.6% |

**Stale-gold rate among decided pairs: 2,575/2,610 = 98.7%.** Only 35 of 3,205
test golds are the value that holds today. Note the direction of C6's bias:
Wikidata lag makes the reference look *more* like the older gold, so it inflates
`gold_is_current`. 98.7% is therefore conservative.

### Correction: the mechanism I claimed is real but rare

Rescoring predictions against live Wikidata instead of the dataset gold
(n=2,949 with a resolvable current value):

| arm | EM vs dataset gold | EM vs Wikidata current | penalised for being right |
|---|---|---|---|
| base | 0.0380 | 0.0058 | 16 (0.5%) |
| SFT | 0.0197 | 0.0061 | 17 (0.6%) |
| TSCT | 0.0254 | 0.0051 | 15 (0.5%) |

The YouTube and News Corp spot checks were **exceptions, not the rule**. Only
~0.5% of predictions (15-17 rows) are cases where the model gave the current
value and was marked wrong. I claimed this "inverts the paper's premise" and
"explains the 2.7% accuracy" — **both were overclaims from two examples, and the
rate refutes them.**

What is actually true: the models do not know *either* value. 3.8% against the
expired gold, 0.6% against current truth. Most wrong answers are wrong both ways,
consistent with C3's 86% "neither" bucket.

### What survives, restated precisely

1. **The premise holds.** Models score 0.6% against current truth, so they
   genuinely do not know these post-cutoff facts. That is what TSCT assumes.
2. **The benchmark is still invalid, for a different reason than I said.** It
   asks present-tense questions and scores agreement with values that expired in
   2023-24. "Accuracy" on it means *reproduces the old snapshot*, not *knows the
   fact* — and it overstates knowledge by roughly 6x (3.8% vs 0.6%).
3. **Contamination of other claims is minor, not fatal.** At ~0.5% affected, B6/D1
   and the C3 rescore stand; the relative ordering between arms is unchanged. I
   previously wrote "do not report any accuracy or ECE number from this split" —
   that was too strong. They can be reported **with the label-as-of date stated**
   and the 6x overstatement noted.

## C7. Fine-tuning explicitly teaches the wrong answer for 29% of the test set
**`ESTABLISHED`** — and it is the mechanism behind B6 and B7

The split is partitioned by **time**, not by question. The same question
("Who is the CEO of X?") therefore appears in multiple splits with different
answers:

| | count |
|---|---|
| unique test questions | 3,177 |
| **also present in TRAIN** | **927 (29.2%)** |
| ...of those, TRAIN teaches a *different* answer | **895 (96.5%)** |

So for roughly 900 test items, fine-tuning shows the model the question with the
**2018-2021** value, and the test then asks the same question expecting the
**2023-2024** value. The training procedure actively teaches the answer the
evaluation marks wrong.

**This is the mechanism behind B6 and B7.** Fine-tuning lowers volatile-fact
accuracy (0.0392 -> 0.0273) and floods predictions with wrong-entity values
(7.9% -> 42.1%) because that is precisely what it was trained to do on a third of
these questions. Those were previously recorded as observations without a cause;
this is the cause.

Not a labelling bug so much as an unavoidable consequence of time-partitioning a
fact table — but it makes "train on TemporalDelta, test on TemporalDelta" a
self-defeating protocol for anything except measuring memorisation of the older
snapshot. Any future design should partition by **entity**, not by time, or
exclude train-seen questions from the test set and report that subset separately.

## C8. `exact_match` is a mild capability floor, but the obvious relaxation is worse
**`ESTABLISHED`**

Checking whether EM undercounts correct answers (base arm): 158/3,622 (4.4%) are
scored wrong with nonzero token overlap, 85 at f1>=0.5. Counting those as correct
would move accuracy 3.92% -> 6.27%.

**It should not be done.** Inspecting that bucket, it is dominated by
name-collision artifacts rather than genuine variants:

| predicted | gold | verdict |
|---|---|---|
| Lachlan Murdoch | Rupert Murdoch | different people, shared surname |
| Michel Boyer | Michel Paulin | different people, shared forename |
| Petr Chmelík | Petr Witowski | different people, shared forename |
| **Dave Calhoun** | **David Calhoun** | **genuine — same person** |
| **Richard Gonzalez** | **Richard A. Gonzalez** | **genuine — same person** |

Genuine undercount is a handful of items, not 85. This is the same failure mode
_The Recall Debt_ documents in its Section 5: a plausible relaxation
(there, embedding-disjointness; here, f1>=0.5) **enriches for collisions rather
than purifying**, and a weak gate waves them through. Our own proposed fix would
have inflated accuracy by ~60% relative, almost entirely with wrong people who
share a name.

Conclusion: keep EM, note it is a slight floor, and do **not** substitute a fuzzy
threshold without a capability-grade check on what it admits.

## C6. The verification reference has its own lag — "current" is not ground truth
**`ESTABLISHED`** (as a constraint on method, before any D8 number is read)

Raised by Edward 2026-08-11, pointing at a Google/DeepMind leadership reorg
reported 2026-08-05 — six days before our audit query.

D8 compares the dataset's gold against *live Wikidata*. That is a better
reference than the frozen snapshot, but it is **not ground truth**. Wikidata is
volunteer-edited and lags reality by an unknown, entity-dependent interval. For a
change six days old, it may be absent, partial, or mid-edit. So D8's verdicts can
be wrong in both directions:

- `gold_is_stale` may be **wrong** if Wikidata has moved ahead of a change that
  was later reverted or mis-entered;
- `gold_is_current` may be **wrong** if Wikidata simply has not caught up, making
  a genuinely stale gold look fine.

This is C5 recursing one level up. The dataset's labels are a dated snapshot of
Wikidata; our audit is a *later* dated snapshot of the same source. Neither is
truth, and the honest framing of D8 is:

> **"dataset gold vs Wikidata as of `queried_at_utc`"**, not "gold vs reality".

Practical consequences, all adopted:
1. `queried_at_utc` is recorded in the audit output — already done.
2. A follow-up pass records each entity's Wikidata `modified` timestamp, so the
   freshness of the reference is reportable rather than assumed. An entity last
   edited in 2023 is much weaker evidence than one edited last week.
3. Any headline rate from D8 must carry both dates: the dataset snapshot and the
   query date.

**Deeper point worth putting in the paper.** This is the paper's own thesis
turned on the paper: *any* temporal factual benchmark is a dated snapshot whose
labels decay, and validating one requires another dated snapshot that is also
decaying. There is no fixed reference. That argues for benchmarks that report a
label-as-of date and are re-validated on a schedule, rather than published once
as if timeless — a concrete, defensible methodological recommendation that comes
directly out of this audit.

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
| ~~D4~~ | **RESOLVED** -> see B7. Fine-tuning shifts answers into the training corpus (13.7% -> 55.8%), almost all attached to the WRONG entity (7.8% -> 42.0%). Memorised vocabulary, no entity mapping. | — | done |
| D5 | Does TCL help on a **volatility-balanced** held-out set? | Every held-out measurement so far is on a set where one class is 90.8%. | needs a new split |
| ~~D8~~ | **RESOLVED** -> see C5. Stale-gold rate 98.7% (2,575/2,610). But only ~0.5% of predictions are penalised for being current, so the contamination is minor and my "inverts the premise" reading was wrong. | — | done |
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
