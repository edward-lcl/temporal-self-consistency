# Proposal Updates — Spring 2026 (Post-Mentor-Review)

Updates to "Temporal Self-Consistency Training for Reasoning Under Evolving World Knowledge" based on mentor recommendation and Week 1 data work.

---

## Section: Datasets and Evaluation — REPLACEMENT

The following section replaces the existing "Datasets (evaluation)" and "Training data" sections in the proposal.

### Datasets (evaluation)

The original proposal listed TempLAMA and StreamingQA as primary evaluation benchmarks. Both have been **dropped** because their questions are drawn from time periods that fall before our base model's knowledge cutoff (March 2023 for LLaMA-3 8B), which means accuracy on these benchmarks measures memorization rather than post-cutoff temporal calibration. We replace them with three more recent benchmarks recommended by our mentor:

- **FreshQA** (https://arxiv.org/abs/2310.03214): Benchmark explicitly designed for fast-changing world facts; distinguishes 'never-changing', 'slow-changing', and 'fast-changing' question types — directly aligned with our Temporal Volatility Taxonomy.
  - GitHub: https://github.com/freshllms/freshqa (CSV format)
  - Restricted to post-cutoff questions in our evaluation.

- **PAT-Questions** (2024, https://jannatmeem95.github.io/PAT-Questions-Web/): 2,882 single-hop and 3,290 multi-hop present-anchored temporal questions with multiple date snapshots (Dec 2021, Dec 2023, March 2024). Includes a self-updating script that refreshes answers via SPARQL queries against current Wikidata.
  - Use as primary contrastive-pair source for training data construction (changed vs unchanged answers across snapshots).
  - Also used as evaluation benchmark on the held-out 2024 snapshot.

- **TLQA** (2025, https://github.com/elixir-research-group/TLQA): List-based temporal QA benchmark with answers aligned to time periods, derived from Wikipedia articles and infoboxes. Includes train/test splits with golden evidence annotations.
  - Used as eval-only benchmark — list-format answers are not well-suited to single-answer hedge-token output but provide a strong evaluation signal.

- **TDBench** (2026, https://github.com/ssoy0701/tdbench, https://arxiv.org/abs/2508.02045): Time-sensitive QA benchmark constructed using temporal databases, temporal SQL, and temporal functional dependencies. Introduces a "time accuracy" metric evaluating the validity of time references in model explanations alongside answer accuracy.
  - Used as supplementary eval. Time accuracy metric is adopted as part of our hedge quality evaluation.

### Training data

- **TemporalDelta (ours)**: Multi-relation Wikidata snapshot triples converted to QA pairs, annotated with validity intervals and volatility class. Collected via SPARQL queries on 11 Wikidata properties spanning fast-changing (CEO, head of state, head of government, chairperson, officeholder, director) and slow-changing (capital, member of, owned by, parent organization, board member) categories. Split 70/15/15 train/val/test by change year to avoid temporal leakage.

- **PAT-Questions training subset**: Singlehop and multihop questions from PAT-Questions with Dec 2021 vs Dec 2023 snapshot comparison used to generate `[CONFIDENT]` (unchanged) vs `[TEMPORAL_HEDGE]` (changed) labels. This addresses an initial hedge token imbalance in our Wikidata-only training data.

- **Curated immutable seed examples**: A small curated set of physical constants, mathematical facts, historical dates, and authorship facts annotated with `[CONFIDENT]`. Used to ensure the model sees all four hedge tokens during training.

- **Preference pairs for DPO**: Human-annotated pairs from Amazon Mechanical Turk rating temporal hedge appropriateness (per Week 5 task).

### Final training dataset statistics

| Split | Records | `[CONFIDENT]` | `[COND_CONFIDENT]` | `[TEMPORAL_HEDGE]` | `[UNKNOWN]` |
|---|---|---|---|---|---|
| Train | 14,206 | 5,005 (35%) | 705 (5%) | 8,494 (60%) | 2 (<1%) |
| Val   | 1,678  | 0           | 290 (17%)   | 1,388 (83%)        | 0           |
| Test  | 3,622  | 0           | 335 (9%)    | 3,287 (91%)        | 0           |

The `[UNKNOWN]` token underrepresentation in training will be addressed in Week 2 via Aarav's deployment-date offset sampling, which simulates queries asked from future deployment dates where the model genuinely cannot know the answer.

---

## Section: Benchmarks/Evaluation Sets — REPLACEMENT

Replaces the existing "Benchmarks/Evaluation Sets" subsection.

- **Primary benchmarks**: FreshQA, TLQA, TDBench, plus our TemporalDelta test set
- **Secondary benchmark**: MMLU stable subset (24 time-stable subjects: math, physics, chemistry, anatomy, etc.) — used as accuracy regression check; TSCT must not degrade performance here
- **Stress-test sets**:
  - Extended horizon: facts 18–36 months post-cutoff (3,097 records from our Wikidata pipeline)
  - Adversarial stable: 49 hand-curated stable facts where over-hedging is penalized
  - Mixed-paragraph: 60 passages containing both stable and volatile claims requiring selective hedging
- **Baselines**: base LLM, RAG-augmented LLM, self-consistency decoding, SFT without TCL, temperature scaling, label smoothing SFT, oracle (always emits correct hedge label)

---

## Section: Potential Limitations — ADDITIONS

Add the following to the existing limitations list.

- **Base model cutoff mismatch with split design**: The original proposal's split (train=[2018–2021], val=2022, test=[2023–2024]) assumed a 2021-cutoff model. Our base model (LLaMA-3 8B) has a March 2023 cutoff, meaning val and ~44% of test are technically within the training window. We address this empirically rather than re-partitioning: the contamination audit (Logan's Week 1 task) runs the base model on all val and test questions and removes any fact category where base model accuracy exceeds 90%. This filters out memorized facts regardless of date boundaries.

- **Wikidata qualifier sparsity**: Of the 207,064 raw triples extracted, only 24% had end-date qualifiers (P582) — the field on which our time-partitioned split relies. This reduces the usable dataset substantially but reflects systematic Wikidata gaps rather than a pipeline flaw. Documented as a coverage limitation in the dataset section.

- **TDBench data availability**: TDBench code/data was released after our Week 1 deadline. We integrate it as a Week 3+ evaluation benchmark contingent on data availability.

---

## Section: Tasks Document — REPLACEMENT TEAM ASSIGNMENT TABLE

Replaces the Week 1 and Week 2 sections of the tasks document.

### Week 1: Setup and Data Collection — UPDATED

**Jason and Aarav**

- Collect Wikidata triples via SPARQL across 11 target relations: P169 (CEO), P35 (head of state), P6 (head of government), P36 (capital), P463 (member of), P488 (chairperson), P1308 (officeholder), P127 (owned by), P749 (parent organization), P1037 (director/manager), P3320 (board member)
- For large relations (P6, P36, P463, P749), use continent filters (for geographic entities) or entity-type filters (for organizational entities) to work around the 1-minute SPARQL timeout
- Extract validity intervals from P580/P582 qualifiers
- Classify into volatility classes per relation type
- Convert to QA format with hedge token assignments
- Apply time-partitioned split: Train=[2018–2021], Val=2022, Test=[2023–2024]
- Integrate PAT-Questions (singlehop + multihop, Dec 2021 vs Dec 2023 snapshots) into training data
- Add curated immutable seed examples to ensure all four hedge tokens appear in training
- Deliverables:
  - `temporal_delta_train.jsonl` (14,206 records)
  - `temporal_delta_val.jsonl` (1,678 records)
  - `temporal_delta_test.jsonl` (3,622 records)
  - `all_triples.jsonl` (207,064 raw records)
  - `relation_volatility_map.json`

**Tanvi**

- Set up fine-tuning environment (HuggingFace + trl)
- Load base model: **LLaMA-3 8B** (decision finalized over Mistral 7B for stronger instruction-following on hedge token output format)
- Add 4 hedge tokens to vocabulary, resize embeddings
- Lock and share output format spec: `Answer: <text> / Hedge: [TOKEN]`

**Logan**

- Run base LLaMA-3 8B Instruct on all val and test questions (expanded from test-only after cutoff mismatch was identified), record accuracy and ECE
- Flag fact categories where base model exceeds 90% accuracy, report to Jason for removal
- Download FreshQA, TLQA, TDBench; restrict all eval sets to post-cutoff time ranges
- Implement oracle baseline (always emits correct hedge label, used as ceiling in results table)

### Week 2: Dataset Finalization and Training Prep — UPDATED

**Aarav**

- Review all three `.jsonl` splits for temporal leakage; verify strict time boundaries
- Build contrastive pairs: (Q at T1, A1) paired with (Q at T2, A2 ≠ A1)
- Sample simulated deployment date offsets (0–24 months post-cutoff) to generate additional `[UNKNOWN]` training examples (addresses Week 1 distribution gap)
- Build entity blacklist (Wikipedia >10K page views) and apply to test set
- Hand finalized `train.jsonl` to Tanvi

**Jason**

- Set up ECE pipeline: equal-frequency binning, softmax of hedge token as confidence source
- Implement accuracy metrics: exact match and F1
- Implement volatility-split breakdowns: Fast vs Slow vs Immutable
- Set up Bonferroni correction framework (α ≈ 0.007 across 7 comparisons)
- Implement volatility discrimination scoring pipeline
- Implement temporal generalization gap metric (12-month splits from training cutoff)

**Tanvi**

- Implement TCL loss: all three terms (L_over, L_under, R_hedge)
- Write unit tests confirming each penalty and reward term fires correctly on toy inputs
- Begin SFT-only training run on mini dataset from Aarav's Week 1 sample
- Configure multiple random seeds (minimum 3) for all training conditions

**Logan**

- Implement self-consistency baseline: sample 10 outputs per question, majority vote
- Implement temperature scaling baseline (post-hoc calibration on val set using model logits)
- Implement label smoothing SFT baseline: a separately fine-tuned model on TemporalDelta with CE loss + ε=0.1 smoothing (not a confidence rescaling of the base model)

---

## Notes on what changed vs original proposal

1. **Eval benchmarks**: TempLAMA and StreamingQA removed (pre-cutoff). FreshQA retained. TLQA, PAT-Questions, TDBench added.
2. **Training data**: PAT-Questions added as primary contrastive-pair source. Wikidata expanded from initial 5 properties to 11.
3. **Base model**: LLaMA-3 8B decision documented (over Mistral 7B).
4. **Cutoff mismatch**: Explicitly addressed via contamination audit rather than re-partitioning splits.
5. **Stress test sets**: Three concrete stress sets defined and built (extended horizon, adversarial stable, mixed paragraph).
6. **Logan's Week 2 baselines**: Clarified that label smoothing must be a separately trained model, temperature scaling must use real logits — not confidence rescaling tricks.
