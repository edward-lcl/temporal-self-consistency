# Temporal Self-Consistency Training for Reasoning Under Evolving World Knowledge

*Draft. Results / Discussion / Conclusion written 2026-08-11 from the independent reimplementation; see `specs/CLAIMS-LEDGER.md` for per-claim evidence, controls and provenance pins. **Sections 1–3 still describe the originally proposed framing and now under-state the negative findings — they need revision to match Sections 4–6.***

---

## Abstract

Language models answer questions about a changing world from frozen weights, and a natural remedy is to train them to emit calibrated hedge tokens indicating how volatile a fact is. We implement this approach — a four-token hedge vocabulary trained with an asymmetric temporal calibration loss (TCL) — and report that it does not work, together with an account of why that outcome was structurally guaranteed.

We first identify and correct a real defect: the calibration loss computed its confidence estimate through a non-differentiable `argmax`, so its gradient was exactly zero at every training step. Correcting it to a softmax expectation restores gradient at every step, across four seeds and at 7B scale. With the gradient restored, however, TCL changes expected calibration error by less than 0.005 against a matched cross-entropy baseline, and the sign of the difference flips between seeds.

Validity controls explain the null. A *constant* confidence output, ignoring the input entirely, achieves ECE 0.0727 where a *perfect* volatility classifier achieves 0.4504 — the capability-free policy beats the oracle sixfold. The learned hedge output is a relation-level template with chance-level per-example discrimination (within-relation AUROC 0.4997). We generalise this into a condition on when ECE is meaningful at all: a hedging scheme's ECE rewards it only insofar as its asserted confidences sit near the realised accuracies of the classes they label. On our stable-fact set, where the asserted 0.95 matches a realised 0.88, the oracle wins as expected; on the volatile set, where 0.45 is asserted against a realised 0.026, the metric inverts.

We further audit the evaluation apparatus and find 98.7% of gold answers expired relative to live Wikidata, 29.2% of test questions also present in training with a different answer, and one of the four hedge tokens absent from the data entirely.

Finally, we show the signal the method sought was already present: the model's own answer log-probabilities predict its correctness at AUROC 0.85 on the same examples where the trained hedge output is at chance, and the *untuned* model's signal is stronger than the fine-tuned one's. The discrete hedge parameterisation discards information the model already had. We report this as evidence that the representation, not the optimiser, sets the ceiling — while noting that the recovered signal is statistically overwhelming yet practically weak (average precision 3.3× baseline), and so does not by itself constitute a working method.

---

## 1. Introduction

Language models trained on static text corpora must answer questions about a constantly changing world using frozen weights. While prior work has established that LLMs hallucinate confidently when their factual knowledge has become outdated, no existing approach teaches a model to reliably distinguish between time-stable facts (the speed of light, the year World War 2 ended) and time-volatile facts (the current CEO of a company, the present holder of a political office). As a result, deployed models routinely emit overconfident assertions on facts that have changed since their training cutoff — a failure mode that contributes to misinformation and undermines user trust.

Existing approaches address this problem only partially. Retrieval-Augmented Generation (RAG) sidesteps the issue by injecting up-to-date context at inference time, but requires a curated, continually-refreshed corpus and fails silently when retrieval is incomplete. Self-consistency decoding improves reasoning reliability through ensemble voting but cannot help when the model's underlying knowledge is uniformly wrong. Temperature scaling and label smoothing produce globally better-calibrated confidence scores but cannot distinguish per-fact volatility — they apply the same correction to a physical constant as to a political appointment.

We argue that temporal calibration must be a *native model capability*, not an external patch. A well-calibrated model should:
1. Express full confidence on time-stable facts ("The speed of light is 299,792,458 m/s.")
2. Hedge appropriately on slow-changing facts ("The capital of France is Paris, as of my knowledge.")
3. Flag time-volatile facts as potentially outdated ("The CEO of X is Y, but this may have changed.")
4. Decline confidently when a fact is genuinely beyond reliable knowledge ("I cannot reliably answer who currently holds this position.")

We introduce **Temporal Self-Consistency Training (TSCT)**, a fine-tuning methodology that teaches a language model to natively distinguish time-volatility classes and emit appropriately calibrated hedges, without requiring retrieval at inference time. Our contributions are:

- **TemporalDelta**: a contrastive temporal dataset constructed from 11 Wikidata properties spanning fast- and slow-changing fact categories, plus integration of PAT-Questions for additional contrastive supervision. The dataset is labeled with validity intervals and a four-token hedge taxonomy.
- **Temporal Calibration Loss (TCL)**: a novel loss function that augments standard cross-entropy with three additional terms — overconfidence penalty, underconfidence penalty, and hedge quality reward — applied selectively to time-sensitive examples.
- **Hedge token vocabulary**: four discrete hedge tokens (`[CONFIDENT]`, `[COND_CONFIDENT]`, `[TEMPORAL_HEDGE]`, `[UNKNOWN]`) appended to the model's output, enabling structured uncertainty expression and direct ECE measurement.
- **An empirical evaluation framework** comparing TSCT to seven baselines (including the proposed hypothesis that TCL outperforms label smoothing, temperature scaling, RAG, and SFT-only alternatives) using post-cutoff benchmarks (FreshQA, PAT-Questions 2024, TLQA, TDBench) and a contamination-audited Wikidata test set.

Our research question is: *Can a language model be trained to reliably distinguish time-stable from time-volatile factual claims and express calibrated uncertainty about the latter, without access to external retrieval at inference time?*

---

## 2. Related Work

### 2.1 Factual reasoning benchmarks for LLMs

The MMLU benchmark established a standard for measuring factual and reasoning capabilities across 57 subjects. However, MMLU is a static snapshot: questions about current events, recent scientific discoveries, or recently changed facts become stale or misleading as time passes. Models trained to maximize MMLU may learn to confidently assert facts that are contextually time-bound, with no mechanism for expressing temporal uncertainty. We use a curated MMLU stable subset (24 immutable subjects) as a regression check rather than a primary benchmark.

### 2.2 Time-sensitive question answering

TempLAMA introduced a dataset of time-sensitive facts ("Who is the CEO of X?") with versions spanning multiple years, exposing how LLMs fail to appropriately hedge answers when the correct answer depends on when the question is asked. A key limitation of TempLAMA is that it diagnoses temporal inconsistency without offering a training method to fix it. We extend this line of work by treating temporal calibration as a trainable objective. We do not use TempLAMA as a primary evaluation benchmark because its questions are drawn from time ranges that predate our base model's training cutoff, conflating memorization with calibration.

More recent benchmarks address this temporal gap directly. **FreshQA** distinguishes never-changing, slow-changing, and fast-changing questions and is updated to include post-cutoff content. **PAT-Questions** provides multi-snapshot answers (Dec 2021, Dec 2023, March 2024) explicitly designed for measuring temporal answer drift, plus a self-updating script that refreshes ground truth via SPARQL. **TLQA** focuses on list-based answers aligned to validity intervals. **TDBench** systematically constructs time-sensitive QA from temporal databases and introduces a "time accuracy" metric evaluating not only whether an answer is right but whether the model's stated temporal references are valid. We adopt these four as our primary evaluation benchmarks.

### 2.3 Self-consistency and ensemble methods

Self-consistency decoding samples multiple reasoning chains and selects the most consistent answer, dramatically improving reasoning reliability on tasks where intermediate reasoning errors are the dominant failure mode. We argue that self-consistency addresses only one of two failure modes for time-sensitive QA: a model whose weights confidently encode a stale fact will sample multiple mutually consistent but uniformly incorrect chains, and majority voting will amplify the wrong answer rather than expose the uncertainty. We include self-consistency as a baseline and predict that TSCT will outperform it on volatile-fact subsets while incurring lower inference cost (no 10× sampling overhead).

### 2.4 Calibration and post-hoc methods

Temperature scaling and label smoothing are well-established techniques for reducing overconfidence in classification models. Both apply globally uniform corrections — temperature scaling divides all logits by a learned scalar, label smoothing distributes target probability mass uniformly across non-target tokens. Neither can distinguish between facts whose correct confidence should be high (physical constants) and facts whose correct confidence should be low (current leadership positions). We use both as baselines and hypothesize that TCL's per-volatility-class supervision produces meaningfully better calibration on time-sensitive subsets while preserving accuracy on stable subsets.

### 2.5 Retrieval-Augmented Generation

RAG augments LLM generation with dynamically retrieved documents to supply up-to-date factual grounding. RAG is effective but architectural rather than fundamental: the model itself still lacks a principled representation of its own temporal uncertainty. RAG fails silently when retrieval returns stale or irrelevant documents, introduces inference-time latency, and is unavailable in retrieval-free deployments (offline, embedded, edge). We compare against RAG to demonstrate that TSCT can produce competitive calibration without retrieval infrastructure.

---

## 3. Methods

### 3.1 Hedge token vocabulary

We extend the base LLaMA-3 8B tokenizer with four special hedge tokens that the model is trained to emit at the end of every factual response:

| Token | Confidence scalar (eval) | Use case |
|---|---|---|
| `[CONFIDENT]` | 0.95 | Time-stable facts (physical constants, immutable historical events) |
| `[COND_CONFIDENT]` | 0.75 | Slow-changing facts (capitals, organizational structures) |
| `[TEMPORAL_HEDGE]` | 0.45 | Fast-changing facts likely affected by knowledge cutoff |
| `[UNKNOWN]` | 0.10 | Facts genuinely beyond the model's reliable knowledge horizon |

At training time, the softmax probability of the emitted hedge token serves as the differentiable confidence signal (preserving gradient flow into the calibration loss). At evaluation time, hedge tokens are deterministically mapped to fixed scalar confidence values for ECE computation. This dual representation is critical: training requires soft probabilities to enable gradient updates, but evaluation requires fixed mappings to ensure reproducibility across model checkpoints.

### 3.2 Temporal Volatility Taxonomy

Each Wikidata property is mapped to one of three volatility classes:

- **Immutable**: physical constants, mathematical facts, historical events that have already occurred. Source: a curated set of facts plus filters on inception dates and historical event types.
- **Slow-changing**: facts that change on the order of years to decades. Examples: capital cities (P36), member-of relationships (P463), parent organizations (P749), ownership (P127), board memberships (P3320).
- **Fast-changing**: facts that change on the order of months to a few years. Examples: CEO (P169), head of state (P35), head of government (P6), chairperson (P488), officeholder (P1308), director/manager (P1037).

The taxonomy is constructed at the property level rather than the fact level, ensuring consistent labeling across instances of the same relation type. The resulting `relation_volatility_map.json` is shared across all team members and used by every component of the training and evaluation pipeline.

### 3.3 TemporalDelta dataset construction

We construct TemporalDelta from 11 Wikidata properties extracted via the public SPARQL endpoint (`query.wikidata.org`). For each property, we issue a query collecting all statements with their associated P580 (start time) and P582 (end time) qualifiers. Large properties exceeding the 60-second query timeout are partitioned by continent (for geographic entities) or by entity type (for organizational entities) and merged with deduplication.

Each raw triple is converted to a natural-language QA pair using property-specific templates (e.g., "Who is the CEO of {entity}?"). Each QA pair is annotated with:
- The corresponding Wikidata QID for entity and value
- The validity interval `[t_start, t_end]`
- Its volatility class
- A ground-truth hedge token assignment based on volatility class and whether `t_end` is present

The resulting 207,064 raw triples are split into train/validation/test by *change year* (the year the fact last changed, taken from `t_end`):
- **Train**: changes in 2018–2021
- **Val**: changes in 2022
- **Test**: changes in 2023–2024

Only records with a `t_end` field can be assigned to a split, reducing the usable dataset to ~14,000 records — a notable coverage limitation reflecting the sparsity of Wikidata's temporal qualifiers.

### 3.4 PAT-Questions integration

We supplement TemporalDelta with PAT-Questions (singlehop and multihop), comparing the Dec 2021 and Dec 2023 snapshots of each question. Questions with unchanged answers across snapshots become `[CONFIDENT]` training examples; questions with changed answers become `[TEMPORAL_HEDGE]` examples. This integration addresses a critical imbalance in the Wikidata-only training data, where `[CONFIDENT]` examples were severely underrepresented. After integration, the training set contains 14,206 records distributed as: `[CONFIDENT]` 35%, `[COND_CONFIDENT]` 5%, `[TEMPORAL_HEDGE]` 60%, `[UNKNOWN]` <1%.

The remaining `[UNKNOWN]` underrepresentation is addressed via deployment-date offset sampling: at training time, fast-changing facts are paired with simulated query dates drawn from `[cutoff, cutoff + 24 months]`, and examples where the simulated date exceeds the validity interval are labeled `[UNKNOWN]`.

### 3.5 Temporal Calibration Loss (TCL)

We define TCL as a sum of standard cross-entropy plus three calibration-shaping terms:

$$\text{TCL}(x, y, v, \hat{c}) = \text{CE}(x, y) + \lambda_1 \cdot L_{\text{over}}(\hat{c}, y, v) + \lambda_2 \cdot L_{\text{under}}(\hat{c}, y, v) - \lambda_3 \cdot R_{\text{hedge}}(\hat{c}, v)$$

where $x$ is the question, $y$ is the gold answer, $v$ is the volatility class, $\hat{c}$ is the model's emitted confidence (softmax probability of the chosen hedge token), and $\lambda_1, \lambda_2, \lambda_3$ are tunable weights.

- $L_{\text{over}}$ penalizes high confidence paired with incorrect answers on volatile facts (the overconfidence failure mode).
- $L_{\text{under}}$ penalizes low confidence paired with correct answers on stable facts (the over-hedging failure mode).
- $R_{\text{hedge}}$ rewards appropriate hedge selection based on the fact's volatility class.

TCL is applied selectively to time-sensitive examples; immutable facts use standard CE. This selective application avoids penalizing the model for being confident about physical constants while shaping calibration where temporal volatility is real.

### 3.6 Training procedure

Starting from a pretrained LLaMA-3 8B base model, we apply supervised fine-tuning with TCL on TemporalDelta + PAT-Questions for three random seeds (42, 123, 456) to enable statistical comparison. Calibration metrics on the validation set are monitored via Expected Calibration Error throughout training.

An optional fourth training condition applies Direct Preference Optimization (DPO) on top of SFT+TCL, using human-annotated preference pairs collected via Amazon Mechanical Turk that contrast appropriate vs inappropriate hedge selections.

### 3.7 Inference and decoding

At inference, the model generates the answer text followed by a hedge token. To avoid degenerate decoding behavior (e.g., greedy decoding collapsing to a single most-frequent hedge token), we sample the hedge token position with temperature 1.0 over only the four hedge token IDs, and emit the argmax. This decouples hedge-token selection from the otherwise-greedy answer-text decoding.

### 3.8 Evaluation framework

We compute the following metrics for each model condition:

1. **Expected Calibration Error (ECE)** with equal-frequency binning: $\sum_{b \in B} \frac{|b|}{n} |\text{acc}(b) - \text{conf}(b)|$ where each bin contains the same number of examples.
2. **Exact Match (EM)** and **token-level F1** for factual answers, normalized to lowercase and stripped of punctuation.
3. **Volatility-split breakdown**: ECE and accuracy computed separately for fast/slow/immutable subsets.
4. **Bonferroni-corrected significance tests**: TSCT compared against 7 baselines (Base LLM, Self-consistency, Temperature scaling, Label smoothing SFT, RAG, SFT-only, Oracle) with corrected $\alpha = 0.05/7 \approx 0.007$. We report Cohen's $d$ alongside $p$-values and flag results that are statistically significant but not practically meaningful (ECE reduction < 0.01).
5. **Volatility discrimination**: confusion matrix comparing model-predicted volatility class against ground-truth labels, with per-class precision/recall/F1.
6. **Temporal generalization gap**: ECE and accuracy bucketed into 12-month windows by distance from training cutoff, measuring how performance degrades with horizon.

We also conduct three stress tests: (i) extended horizon evaluation on facts 18–36 months post-cutoff; (ii) adversarial stable-facts evaluation measuring over-hedging on 49 curated immutable facts; (iii) mixed-paragraph evaluation on 60 passages containing both stable and volatile claims, measuring selective rather than blanket hedging.

### 3.9 Baselines

We compare TSCT against:

- **Base LLM**: LLaMA-3 8B Instruct with structured JSON prompting, no fine-tuning.
- **Self-consistency**: Base LLM sampled 10 times per question, majority-voted.
- **Temperature scaling**: Base LLM logits divided by a learned temperature parameter $T$ fit on the validation set to minimize NLL.
- **Label smoothing SFT**: Base model fine-tuned on TemporalDelta with CE loss + label smoothing $\epsilon = 0.1$.
- **RAG**: Base LLM augmented with dense retrieval over a Wikipedia snapshot using a standard bi-encoder.
- **SFT only**: Base model fine-tuned on TemporalDelta with CE loss only (no TCL). This is the critical ablation isolating the contribution of TCL.
- **Oracle**: A hypothetical perfect model that always emits the correct hedge label for every example. Used as a ceiling.

All trainable baselines use three random seeds matching TSCT for fair comparison.

---

## 4. Results

### 4.1 Experimental setup and deviations from the proposed protocol

All results below come from an independent reimplementation, and three deviations from Section 3 must be stated before any number is read.

**Base model.** We report `Qwen2.5-7B-Instruct` (4-bit, MLX) rather than LLaMA-3 8B. LLaMA-3 is access-gated; Qwen2.5-7B is ungated, of the same parameter class, and — as Section 4.7 shows — its vocabulary structure turns out to matter for the method in ways that generalise.

**Loss implementation.** The original TCL implementation was unavailable to us. We reimplemented it from the documented specification. The differentiability property under test (Section 4.2) is formula-agnostic, but the concrete forms of `L_over`, `L_under` and `R_hedge` are inferred rather than recovered, and any claim about their relative magnitudes inherits that assumption.

**Provenance.** Every quantitative result is pinned to dataset revision `8763b5be`, model revision `c26a38f6`, and is reproducible from the released harness. Raw per-example predictions are released alongside aggregates.

### 4.2 The calibration loss received no gradient, and the fix restores it

The reported failure — all six original checkpoints emitting `[CONFIDENT]` on 100% of predictions — is attributable to a single defect. `c_hat` was computed by `argmax`-selecting a hedge token and indexing a fixed confidence table. `argmax` is non-differentiable, so `L_over` and `L_under` never received gradient while cross-entropy trained normally.

We measure this directly rather than inferring it from loss curves, isolating `λ_over·L_over + λ_under·L_under` and backpropagating it alone:

| condition | gradient norm |
|---|---|
| original (`argmax` + table lookup) | **exactly 0.0** at 78/78 steps, all 4 seeds |
| corrected (softmax expectation) | nonzero at 78/78 steps, all 4 seeds (mean 3.69–4.47) |

The result holds at 7B: 1,202/1,202 probes nonzero across three seeds. This is a property of the computation graph, not of a model or dataset, and it is the one claim in this paper that replicates without exception.

We note one negative result about our own diagnostic. A companion single-seed analysis reported that the fix also prevents hedge-token collapse. Across four seeds it does not: at seed 1 the corrected path collapses to 100% `[TEMPORAL_HEDGE]`, and at seed 3 it collapses to 100% `[CONFIDENT]` — the exact pathology the fix is meant to prevent — while the broken path fails to collapse at all. At 78 steps the post-training distribution is seed noise. **The gradient result stands; the collapse result does not.**

### 4.3 With the gradient restored, TCL has no measurable effect

We train paired arms differing only in λ: SFT-only (λ=0, cross-entropy on hedge labels) and TSCT (λ_over=λ_under=0.5, λ_hedge=0.3), identical data, seed and step count.

| seed | SFT ECE | TSCT ECE | Δ |
|---|---|---|---|
| 0 | 0.4552 | 0.4506 | −0.0045 |
| 1 | 0.4551 | 0.4572 | +0.0021 |

The difference is within ±0.005 and **changes sign between seeds**. Hedge-token accuracy is likewise indistinguishable (93.2% vs 93.4%).

This does not reproduce the ~0.43 ECE reduction previously reported for this comparison. The discrepancy is not in the TSCT arm, which lands close to the earlier figure (0.451 vs 0.42); it is in the baseline. Our SFT baseline scores 0.4552 and 0.4551 across two seeds, against 0.85 previously reported. Section 4.4 supplies the likely explanation.

### 4.4 Validity controls: a capability-free constant beats a perfect classifier

Before interpreting any ECE value we run the controls the protocol omitted — a reference (oracle) run and empty-solution (constant-policy) controls:

| policy | test ECE | requires model capability? |
|---|---|---|
| **oracle** — emit the *gold* hedge every time | **0.4504** | perfect classifier |
| TSCT | 0.4506 | — |
| SFT-only | 0.4552 | — |
| constant `[CONFIDENT]` | 0.9227 | **no** |
| constant `[COND_CONFIDENT]` | 0.7227 | **no** |
| constant `[TEMPORAL_HEDGE]` | 0.4227 | **no** |
| **constant `[UNKNOWN]`** | **0.0727** | **no** |

Two observations. First, **TSCT is already at the oracle** (0.4506 vs 0.4504): a perfect volatility classifier scores no better, so no method can improve on this benchmark. Second, **a constant that ignores the input entirely beats the oracle by 6×.**

These controls also locate the earlier result. The previously reported pair sits almost exactly on two constant policies — SFT 0.85 against constant-`[CONFIDENT]` at 0.9227, TSCT 0.42 against constant-`[TEMPORAL_HEDGE]` at 0.4227. A baseline collapsed to `[CONFIDENT]`, which the original checkpoints were reported to be, would score ≈0.87. The most parsimonious reading is that the reported effect is the distance between a broken baseline and a working model rather than the contribution of a calibration loss. We state this as a discrepancy rather than a correction: we cannot rule out differences arising from the base model or our reimplementation.

### 4.5 The hedge output is a relation-level template, not a per-example judgement

Pooled across our two evaluation sets, hedge confidence appears to predict answer correctness at AUROC 0.670. Stratification removes the effect entirely:

| | SFT | TSCT |
|---|---|---|
| AUROC, both sets pooled | 0.6700 | 0.6704 |
| AUROC, temporal test only | 0.5026 | 0.5300 |
| **AUROC within relation type** | **0.5000** | **0.4997** |

Per relation, with n≥100: `head_of_gov` 0.500, `officeholder` 0.500/0.499, `chairperson` 0.500, `ceo` 0.500/0.498, `parent_org` 0.500. The pooled 0.67 was the model separating two *corpora*, not two examples.

Within a relation the emitted confidence is essentially constant, so AUROC is 0.5 by construction. **The model learned a lookup from relation type to hedge token** — inferable from the question template with no knowledge of the entity. This explains 4.3 directly: a calibration loss operating on a quantity with no within-relation variance has nothing to act on.

### 4.6 Fine-tuning degrades accuracy, knowledge attribution, and the model's own confidence signal

Against the untrained base model on identical questions, with output length matched:

| | base | SFT-only | TSCT |
|---|---|---|---|
| test EM (volatile) | **0.0392** | 0.0226 | 0.0273 |
| stress-stable EM | 0.7551 | 0.8571 | **0.8776** |
| answer is a real value for a *different* entity | **7.9%** | 38.1% | 42.1% |
| answer appears nowhere in the dataset | 86.3% | 47.8% | 44.1% |

Fine-tuning lowers volatile-fact accuracy by 30% relative while raising stable-fact accuracy, and it increases wrong-entity answers roughly fivefold. The model acquires the *pool* of plausible officeholder names without the entity→answer mapping. Because SFT-only already shows 38.1%, this is attributable to fine-tuning itself, not to TCL.

### 4.7 Dataset and instrument integrity

Four properties of the evaluation apparatus materially affect what any result here can mean.

**The test split cannot detect over-hedging.** All 3,622 test rows are `[TEMPORAL_HEDGE]` (3,287) or `[COND_CONFIDENT]` (335); there is not one `[CONFIDENT]` or `[UNKNOWN]` row, because the split is time-partitioned to facts that changed. A model always emitting `[TEMPORAL_HEDGE]` scores ~90.8% hedge accuracy, and over-hedging is structurally invisible. TCL's over-hedging cost was detectable only on the stable-fact stress set (3 of 49 downgraded).

**`[UNKNOWN]` is untrainable.** It appears zero times in train, validation or test. The model cannot learn to emit the one token that would minimise ECE on this benchmark.

**Gold labels are overwhelmingly expired.** Audited against live Wikidata (2026-08-11), of 2,610 pairs with an unambiguous current value, **2,575 (98.7%) of dataset golds are no longer current**; only 35 are. Wikidata's own lag biases this toward agreement, so it is conservative. The practical impact on scoring is small — only ~0.5% of predictions are cases where the model gave the current value and was marked wrong — because the models know neither value (0.6% accuracy against current truth). The benchmark therefore measures *agreement with a 2023–24 snapshot*, and overstates knowledge roughly sixfold relative to current truth.

**Training teaches the answer the test marks wrong.** Because the partition is by time rather than by question, **927 of 3,177 unique test questions (29.2%) also appear in train, and 895 of those (96.5%) with a different answer.** For roughly a third of the test set, fine-tuning shows the model the 2018–2021 value and evaluation then demands the 2023–24 one. This is the mechanism behind 4.6.

---

## 5. Discussion

### 5.1 The null result was structurally guaranteed

Section 4.3 reports no effect. Sections 4.4 and 4.5 explain why no effect was obtainable, and the two reasons are independent.

The instrument is saturated: a perfect volatility classifier scores 0.4504 where TSCT scores 0.4506, leaving no headroom for any method. And there is no signal to calibrate: the hedge output is constant within a relation, so a loss defined over it has nothing to grip. Neither is a fact about TCL. Both would apply to any method optimising a discrete volatility label against ECE on this benchmark.

This matters for how the null should be read. It is not evidence that temporal calibration is unachievable, and it is not primarily evidence that TCL is a poor loss. It is evidence that **the experiment as designed could not have distinguished a working method from a broken one.**

### 5.2 When is ECE a valid measure of a hedging scheme?

Our two evaluation sets give the same metric opposite verdicts, and the contrast is diagnostic:

| set | class | n | realised accuracy | asserted | ratio | verdict |
|---|---|---|---|---|---|---|
| volatile | `[TEMPORAL_HEDGE]` | 3,287 | 0.0256 | 0.45 | **17.6×** | constant beats oracle 6× |
| volatile | `[COND_CONFIDENT]` | 335 | 0.0448 | 0.75 | **16.7×** | " |
| stable | `[CONFIDENT]` | 49 | 0.8776 | 0.95 | 1.08× | **oracle wins** |

We state the principle generally:

> A hedging scheme's ECE rewards it only insofar as its asserted confidences sit near the realised accuracies of the classes they label. Where that gap is large, ECE is minimised by whichever fixed scalar lies nearest the base rate, and a capability-free constant outperforms a perfect classifier. ECE then measures scalar-to-base-rate mismatch, not calibration ability.

The consequence extends well past this paper. **Any** calibration result reported as ECE on a benchmark where the model is rarely correct is exposed to the same inversion, and cannot support a capability claim unless constant-policy controls are reported beside it. We recommend such controls become standard, in the same way that empty-solution controls have become standard in agentic benchmark auditing.

We also note the taxonomy is wrong in *scale*, not in *kind*: `[COND_CONFIDENT]` items genuinely are answered correctly more often than `[TEMPORAL_HEDGE]` items (0.0448 vs 0.0256). The ordering is real. The four scalars were assigned a priori and never fitted, and they are off by more than an order of magnitude.

### 5.3 The signal the architecture discards

The information the method needs is already present in the model. On the same examples, using the same model, we compare confidence signals as predictors of the model's own correctness:

| signal | within-relation AUROC | overall AUROC |
|---|---|---|
| hedge-token confidence (trained) | 0.4997 | 0.5300 |
| mean answer log-probability (free) | **0.8465** | 0.8406 |
| min answer log-probability (free) | 0.8485 | 0.8415 |
| **base model, before fine-tuning** | — | **0.8814** |

The untuned model carries the strongest calibration signal in the system, and every subsequent stage degrades or discards it: fine-tuning reduces both accuracy (142→99 correct) and the logprob signal (0.8814→0.8406), then the hedge vocabulary replaces it with an output at chance.

Two caveats travel with this. The effect is **statistically overwhelming and practically weak**: z = 13.6 against AUROC 0.5, but average precision is 0.0913 against a 0.0273 base rate — only 3.3× — so ranking by logprob to recover half the correct answers still leaves ~89% false positives. AUROC flatters rare-positive regimes and should not be reported alone. And the positive class is partly contaminated: 35.4% of correct answers are questions seen in training, though the signal survives their exclusion (AUROC 0.8176, n=64).

We therefore claim only that **the architecture discards real information**, not that recovering it yields a usable system.

The structure of this argument follows recent work on structural failure in retrieval, where scaling the retriever leaves a plateau that changing the schema removes. Our axes behave the same way: model scale (0.5B→7B), seeds, and λ move the result not at all, while changing the parameterisation moves within-relation AUROC from 0.4997 to 0.8465. The ceiling is set by the representation, not by the optimiser.

### 5.4 Recommendations

1. **Report oracle and constant-policy controls with every ECE number.** Without them, ECE cannot distinguish calibration from base-rate matching.
2. **Report discrimination separately from calibration**, using average precision alongside AUROC in rare-positive regimes.
3. **Fit confidence scalars to realised accuracy — and report the result as a capability-free baseline**, never as a method.
4. **Prefer continuous confidence derived from the model's own output distribution** to a discrete token vocabulary; retain tokens as a presentation layer if a structured interface is wanted.
5. **Partition temporal datasets by entity, not by time.** Time partitioning trains the model on the answer the evaluation marks wrong.
6. **Carry a label-as-of date and re-validate on a schedule.** Gold decays; so does any reference used to validate it.
7. **Do not relax exact match without checking what the relaxation admits.** Ours would have been ~60% name collisions.
8. **Guard against dead parameterisations** when adding tokens to padded vocabularies (Section 4.7 note below).

### 5.5 Threats to validity

**Single model, few seeds.** One base model, two seeds for the paired comparison, four for the gradient diagnostic. Sufficient to withdraw claims, insufficient to bound variance.

**Reimplemented loss.** `L_over`, `L_under` and `R_hedge` are inferred. Section 4.2 is formula-agnostic; 4.3 is not.

**Base-model substitution.** Qwen2.5-7B, not LLaMA-3 8B. The comparison to previously reported numbers is therefore indicative, not decisive.

**The reference decays too.** Our staleness audit compares gold against Wikidata at a fixed timestamp. Wikidata lags reality by an unknown, entity-dependent interval — a leadership change six days before our query may be absent. The audit measures *gold versus Wikidata as of a date*, never *gold versus truth*. This is the paper's own thesis applied to the paper: validating a temporal benchmark requires another dated snapshot, which is also decaying. There is no fixed reference, which is precisely the argument for scheduled re-validation.

**A practical note on padded vocabularies.** Qwen2.5 ships `vocab_size` larger than its tokenizer, and every unused embedding row is the *same* padding row. With a frozen output head, four added hedge tokens therefore emit identical logits for every hidden state: the softmax is exactly uniform, `c_hat` is constant, and `argmax` returns `[CONFIDENT]` forever. This is indistinguishable from training collapse by inspection of outputs, but it is a dead parameterisation. Adapting the output projection resolves it (contrast gradient 0.000000 → 2406.96). A guard must use a **linear** contrast between hedge logits: their sum has nonzero gradient even when all rows are identical, and any symmetric measure (variance, spread) sits at a stationary point exactly at the degenerate configuration — both pass a dead head. Model families differ: Qwen2.5-7B has 399 spare rows, Qwen3-8B 267, and Mistral-7B-v0.3 **zero**, so the no-resize approach is not portable.

---

## 6. Conclusion

We set out to test whether a language model can be trained to distinguish time-stable from time-volatile facts and express calibrated uncertainty about the latter. We found a real defect in the proposed loss, fixed it, and then found that fixing it changes nothing measurable — because the benchmark it would be measured on cannot detect the difference.

The concrete findings are that the calibration gradient was severed by a non-differentiable `argmax` and is restored by a softmax expectation (replicated across four seeds and at 7B); that with the gradient restored, TCL changes ECE by less than 0.005 and the sign is seed-dependent; that a constant confidence output beats a perfect volatility classifier by 6× on this benchmark, so ECE here measures base-rate matching rather than calibration; that the learned hedge output is a relation-level template with chance-level per-example discrimination; and that the evaluation apparatus has defects large enough to matter on their own — 98.7% of gold answers are expired, 29.2% of test questions are taught a different answer during training, and `[UNKNOWN]` never appears in the data at all.

The constructive finding is that the model's own answer log-probabilities predict its correctness at AUROC 0.85 on exactly the examples where the trained hedge output is at chance, and that the untuned model's signal is stronger still. The information the method sought was present before training and was discarded by the representation chosen to express it.

We therefore report a negative result about a method, a set of defects in a benchmark, and one methodological recommendation we believe generalises: **ECE without constant-policy controls cannot support a claim about calibration ability**, and in the low-accuracy regime where temporal calibration is interesting, it will systematically favour methods that assert a well-chosen constant over methods that actually know something.

---

## Limitations

- **Wikidata coverage**: TemporalDelta is derived from Wikidata, which has systematic biases toward Western, English-language, and high-profile entities. The model may learn temporal metacognition for a narrow slice of world knowledge.
- **Wikidata qualifier sparsity**: Only 24% of extracted triples had end-date qualifiers (P582), the field on which our time-partitioned split depends. This reduced our usable training set substantially.
- **Base model cutoff mismatch**: LLaMA-3 8B's March 2023 cutoff overlaps with our val set (2022 changes) and ~44% of our test set (2023 changes). We address this empirically via a contamination audit removing fact categories where base model accuracy exceeds 90%, rather than re-partitioning the splits. *Note: results in Sections 4–6 use Qwen2.5-7B, for which this audit was not re-run. Its 0.6% accuracy against current values is weak evidence against contamination but is not a substitute for the audit.*
- **Train/test question overlap**: the time-based partition places 29.2% of unique test questions in the training split with a different answer. Reported accuracy on this benchmark is therefore partly a measure of resistance to the training signal. Partitioning by entity would remove this.
- **Gold-label decay**: 98.7% of test gold answers are no longer current as of 2026-08-11. Accuracy on this split measures agreement with a 2023–24 snapshot rather than with the world, and overstates knowledge of current facts roughly sixfold.
- **No fixed reference for validation**: the audit above compares gold against Wikidata at a pinned timestamp, and Wikidata itself lags reality by an unknown, entity-dependent interval. Any validation of a temporal benchmark is one dated snapshot checked against another, both decaying.
- **Metric validity**: ECE on the volatile test split is minimised by a constant confidence output and is therefore not a valid measure of calibration ability in this regime. Results are reported with oracle and constant-policy controls throughout; ECE values without those controls should not be compared across papers.
- **Statistical power**: the paired arm comparison uses two seeds; the correctness-prediction analysis has 99 positive examples out of 3,622. Sufficient to withdraw claims, insufficient to bound effect sizes.
- **Hedge verbosity**: Temporal hedging adds tokens to outputs and may annoy users seeking concise answers; calibration must be balanced against usability.
- **Volatility taxonomy rigidity**: Facts rarely fall neatly into immutable/slow/fast categories. Edge cases (e.g., scientific consensus that changes suddenly) may confuse the model.
- **Generalization to long-context**: TSCT is validated on single-fact QA. Generalization to multi-fact documents or long-form generation is an open question — partially addressed by our mixed-paragraph stress test.
- **Ethical considerations**: A model that expresses uncertainty may be gamed by adversaries who prefer confident (if wrong) outputs. Appropriate uncertainty can also be misused to dismiss valid information. Deployment requires careful consideration of these tradeoffs.
