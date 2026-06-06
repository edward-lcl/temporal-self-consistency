# Temporal Self-Consistency Training for Reasoning Under Evolving World Knowledge

*Draft sections — Introduction, Related Works, Methods. Results / Discussion / Conclusion pending model results.*

---

## Abstract

*Placeholder — write last after results are finalized. Will follow standard structure: problem → method → key result → significance.*

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

*Pending corrected model checkpoints. Will populate from `results.json` once Tanvi's TSCT training is fixed (currently all checkpoints emit `[CONFIDENT]` exclusively).*

---

## 5. Discussion

*Pending results.*

---

## 6. Conclusion

*Pending results.*

---

## Limitations

- **Wikidata coverage**: TemporalDelta is derived from Wikidata, which has systematic biases toward Western, English-language, and high-profile entities. The model may learn temporal metacognition for a narrow slice of world knowledge.
- **Wikidata qualifier sparsity**: Only 24% of extracted triples had end-date qualifiers (P582), the field on which our time-partitioned split depends. This reduced our usable training set substantially.
- **Base model cutoff mismatch**: LLaMA-3 8B's March 2023 cutoff overlaps with our val set (2022 changes) and ~44% of our test set (2023 changes). We address this empirically via a contamination audit removing fact categories where base model accuracy exceeds 90%, rather than re-partitioning the splits.
- **Hedge verbosity**: Temporal hedging adds tokens to outputs and may annoy users seeking concise answers; calibration must be balanced against usability.
- **Volatility taxonomy rigidity**: Facts rarely fall neatly into immutable/slow/fast categories. Edge cases (e.g., scientific consensus that changes suddenly) may confuse the model.
- **Generalization to long-context**: TSCT is validated on single-fact QA. Generalization to multi-fact documents or long-form generation is an open question — partially addressed by our mixed-paragraph stress test.
- **Ethical considerations**: A model that expresses uncertainty may be gamed by adversaries who prefer confident (if wrong) outputs. Appropriate uncertainty can also be misused to dismiss valid information. Deployment requires careful consideration of these tradeoffs.
