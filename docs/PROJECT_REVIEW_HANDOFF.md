# TSCT Independent Review, Recovery, and Research Handoff

**Snapshot date:** 2026-08-12

**Audience:** Team ALTJ, project mentors, and the next researcher taking over the work

**Purpose:** A control document for the project state, Edward's role, the evidence chain, and the remaining decisions. This is not the paper draft.

## Executive handoff

Edward joined after a Slack escalation asked for implementation help with unfinished technical work and assistance completing the paper. In practice, the assignment expanded substantially. The original model results were not ready to support the paper, so the intervention became an independent research recovery:

1. establish a writable fork and reconstruct the actual project state;
2. independently reimplement and validate the missing training path;
3. produce working checkpoints and the missing inference bridge;
4. rerun the proposed comparison and its baselines from raw outputs;
5. replicate single-seed findings before treating them as results;
6. audit the dataset, metric, and reference labels;
7. preserve checkpoints, logs, per-example predictions, and a claims ledger; and
8. rewrite the paper around what the evidence supports rather than the result originally expected.

The role was therefore not only “paper/post support” or “implementation assistance.” It became **Edward-led experimental recovery, independent verification, benchmark audit, evidence management, and research drafting**, with an AI coding/research agent executing substantial code, experiments, analysis, and prose under Edward's direction. The paper's [Generative AI Disclosure](../paper/paper_draft.md#generative-ai-disclosure) records that division explicitly.

The present scientific bottom line is narrower than the original proposal and stronger as an audit result:

> The calibration gradient defect was real and was repaired, but this evaluation cannot establish a TCL benefit. On the volatile test regime, a capability-free constant beats a perfect hedge classifier on ECE; the learned hedge output has no within-relation discrimination; most reference answers have expired; train/test construction teaches outdated answers for many repeated questions; and repairing the two repairable data defects makes the ECE pathology worse. The best surviving direction is selective answering or retrieval routing from the model's internal confidence signal, not a four-token calibration vocabulary.

That summary is **snapshot-bounded**. It does not establish that temporal calibration in general is impossible, nor that the team's unavailable original training implementation is equivalent to the independent reimplementation.

## 1. Assignment basis and role boundary

The private Slack message supplied for this review said the team was moving slowly, the first paper draft and poster were overdue, and asked for Edward or another implementor to help Jason finish technical tasks and assist with the paper. This public-repository artifact intentionally paraphrases that assignment rather than reproducing the private workspace URL, personnel tagging, or verbatim internal message.

Before the intervention, the public upstream repository already contained meaningful work:

- Jason's evaluation and data-pipeline infrastructure, status documentation, stress sets, plotting code, and draft scaffolding;
- a public TemporalDelta dataset and a defined four-token hedge taxonomy;
- reported self-consistency and temperature-scaling baseline numbers; and
- a diagnosis of the non-differentiable `argmax` path in `docs/tcl_debugging.md`.

The public repository did **not** contain the training implementation or working inference outputs. Its own [status file](STATUS.md) reported that the supplied prediction files emitted `[CONFIDENT]` throughout, with several team-owned baselines and data tasks still unfinished. The original training repository and exact loss formulas remained unavailable.

Edward's intervention should therefore be credited as an independent recovery and audit built on the team's proposal, dataset, and evaluation infrastructure—not as the origin of the entire project, and not as a routine editing pass over finished experiments.

## 2. Repository authority and branch topology

Remote state was fetched and checked on 2026-08-12.

| authority | ref | state | meaning |
|---|---|---|---|
| upstream project | `jasontae/temporal-self-consistency`, `main` | `594dd20` | Team's public evaluation/data repository before Edward's fork work |
| Edward's fork, main | `edward-lcl/temporal-self-consistency`, `main` | `c962715` | Two operational commits beyond upstream: minimal CI and one starter test |
| Edward's fork, research checkpoint | `edward-lcl/temporal-self-consistency`, `tcl-fix-validation` | `2e7ca4d` | 33 pushed research commits beyond upstream; authoritative reviewed experiment snapshot |
| this handoff | `codex/project-review-handoff` | based on `2e7ca4d` | Documentation-only branch created so the handoff does not absorb local experiment changes |

Two topology details matter:

- Fork `main` and `tcl-fix-validation` diverged directly after upstream `594dd20`. The research branch does **not** contain the two CI/test commits on fork `main`.
- The active checkout is `/Users/edward/Projects/temporal-self-consistency`. The older path in [the initial state spec](../specs/tsct-project-state.md) points to `/Users/edward/Projects/algoverse-foundry/temporal-self-consistency` and is stale.

The working tree on `tcl-fix-validation` also contains a later local model-control run. Those changes were deliberately excluded from this handoff commit and are described in Section 8.

## 3. What changed after Edward joined

### 3.1 Project recovery and reproducible checkpoints

- Resolved fork/write access and recorded the initial state, ownership, blockers, and source documents.
- Reimplemented the missing TCL training components independently under `src/training/`, documenting inferred loss formulas rather than presenting them as recovered team code.
- Added per-term loss logging and a direct isolated gradient probe so the claimed fix was tested at the mechanism level.
- Preserved run metadata, per-step loss logs, raw generations, model identifiers, and per-example outputs rather than only summary tables.

### 3.2 Experiments completed

- Proxy-model broken-versus-fixed diagnostic across four seeds.
- Three 7B TSCT training seeds and paired SFT-only/TSCT runs at seeds 0 and 1.
- A generation/inference bridge that turns saved adapters and base models into the canonical format consumed by Jason's evaluation pipeline.
- Held-out TemporalDelta evaluation, stable-fact stress evaluation, oracle controls, and all four constant-policy controls.
- Untuned-base comparisons and answer-source/interference analysis.
- Seed replication of behavioural claims that initially appeared positive.
- Live-Wikidata currency audit with cached raw API responses and a pinned query timestamp.
- Cross-vintage base-model evaluations and selective-prediction operating points.

### 3.3 Research interpretation and writing

- Created a claim-indexed evidence ledger with `ESTABLISHED`, `SUPPORTED`, `CONTESTED`, `REFUTED`, and `UNCERTIFIED` states.
- Preserved corrections instead of deleting superseded findings. The clearest example is the single-seed “fix prevents collapse” claim, later refuted at seed 3.
- Reframed the paper from a claimed calibration improvement to a measurement-validity and benchmark-audit result.
- Wrote/revised the abstract, introduction contribution list, Results, Discussion, Conclusion, Limitations, related-work additions, and the AI disclosure from the completed experiments.

## 4. Chronological checkpoint map

| date | checkpoint | outcome | primary evidence |
|---|---|---|---|
| 2026-08-08 | onboarding and landscape review | Paper/post request found to depend on unfinished and contradictory results | [initial state spec](../specs/tsct-project-state.md), [working log](../specs/CHANGELOG.md) |
| 2026-08-09 | fork and proxy diagnostic | Writable fork established; independent broken/fixed harness built; gradient path shown as 0 versus nonzero | `6cb043f`, [validation record](../specs/tcl-fix-validation.md) |
| 2026-08-10/11 | MLX 7B scale-up | Local 7B training made viable; padded-vocabulary dead head found and repaired; working checkpoints produced | `da53a8c`, `a6b97d5`, `data/prep/tcl_mlx_7b/` |
| 2026-08-11 | first held-out comparison | Missing inference script completed; SFT/TSCT ECE gap measured at only 0.0046 at seed 0 | `1484a68`, `data/prep/predictions_7b/` |
| 2026-08-11 | validity controls | Oracle and constant policies showed the headline ECE was dominated by base-rate/scalar mismatch | `e33f926`, Claims B3/F1 |
| 2026-08-11 | discrimination and base-model checks | Within-relation hedge AUROC fell to chance; fine-tuning reduced volatile accuracy and degraded the model's own signal | `4ad9fa2` through `929da92` |
| 2026-08-11 | benchmark audit | 98.7% stale gold among decided pairs; train/test question overlap and name-collision risks quantified | `78f2001`, `e3b6100`, `77e9676` |
| 2026-08-11 | replication | Three apparent single-seed TCL behaviours failed replication; SFT ECE stayed near 0.455 across two seeds | `023b474`, `03c232a`, `79a3624`, `a3c152f` |
| 2026-08-11 | paper recovery | Draft rewritten around the benchmark/instrument diagnosis and selective-answering direction | `af89bc9` through `f806a14` |
| 2026-08-11 | model-vintage ladder | Three pushed base-model points suggested absolute selective-answering precision rises with model currency | `6ee053a`, `2e7ca4d` |
| 2026-08-12 | handoff audit | Remote topology rechecked; headline raw records recounted; newer uncommitted Gemma controls discovered and quarantined from the pushed snapshot | this document |

## 5. Evidence map: what to trust for what

| artifact | use | evidentiary status |
|---|---|---|
| [Claims Ledger](../specs/CLAIMS-LEDGER.md) | Claim-by-claim evidence, falsifiers, caveats, and corrections | Best interpretive index; later sections outgrew its frozen repo pin and need a provenance refresh |
| [Working Log](../specs/CHANGELOG.md) | Chronology, decisions, false starts, and handoff context | Historical narrative; some open-thread items are stale because the log was not fully normalized after later commits |
| [Initial State Spec](../specs/tsct-project-state.md) | What was known and blocked at entry | Historical baseline only; paths and next actions have since changed |
| [TCL Fix Validation](../specs/tcl-fix-validation.md) | Original proxy diagnostic | Read the correction notice first; gradient finding holds, collapse finding was refuted |
| `data/prep/tcl_diagnostic*/loss_log.csv` | Direct broken/fixed gradient measurements | Raw, replicated mechanism evidence |
| `data/prep/tcl_mlx_7b/*/run_meta.json` and `loss_log.csv` | 7B run configuration, distributions, gradients, and training steps | Raw run evidence; adapter weight files themselves are not committed |
| `data/prep/predictions_7b/*.jsonl` | Per-example model outputs and scores | Raw behavioural evidence; pushed files define the reviewed snapshot |
| [Gold currency audit](../data/prep/gold_currency_audit.json) plus `wikidata_cache/` | Gold-versus-Wikidata-as-of-date audit | Recountable dated reference, not timeless ground truth |
| [Paper draft](../paper/paper_draft.md) | Current long-form scientific narrative | Draft, not submission-ready; see Section 9 |
| `src/training/` | Independent training and generation implementation | Reimplementation with explicit formula/architecture caveats, not recovered original team code |

## 6. Raw-record verification performed for this handoff

The following numbers were recomputed on 2026-08-12 from the checked-out raw artifacts, not copied only from the prose ledger.

### 6.1 Gradient-path mechanism

Across the four proxy seeds, every broken-mode probe was exactly zero and every fixed-mode probe was nonzero:

| proxy seed | broken nonzero probes | fixed nonzero probes | fixed mean gradient norm |
|---|---:|---:|---:|
| original seed 0 | 0/78 | 78/78 | 4.2565 |
| seed 0 reproduction | 0/78 | 78/78 | 4.2564 |
| seed 1 | 0/78 | 78/78 | 3.6946 |
| seed 2 | 0/78 | 78/78 | 4.4739 |
| seed 3 | 0/78 | 78/78 | 4.1158 |

This verifies the narrow mechanism claim: `argmax` severed the path and the softmax expectation restored it. It does not verify a behavioural benefit.

### 6.2 Held-out ECE and replication

| run | n | ECE | exact-match accuracy |
|---|---:|---:|---:|
| SFT-only seed 0 | 3,622 | 0.455163 | 0.02264 |
| TSCT seed 0 | 3,622 | 0.450635 | 0.02733 |
| SFT-only seed 1 | 3,622 | 0.455052 | 0.02402 |
| TSCT seed 1 | 3,622 | 0.457178 | 0.02098 |

The apparent seed-0 benefit is 0.004528 and reverses at seed 1. That is evidence against a stable TCL ECE effect in this setup.

### 6.3 Oracle and capability-free controls

On the same 3,622 TSCT records:

| policy | ECE |
|---|---:|
| perfect gold-hedge oracle | 0.450414 |
| constant `[CONFIDENT]` | 0.922667 |
| constant `[COND_CONFIDENT]` | 0.722667 |
| constant `[TEMPORAL_HEDGE]` | 0.422667 |
| constant `[UNKNOWN]` | **0.072667** |

The input-independent `[UNKNOWN]` policy beats the perfect hedge oracle by about 6.2 times on ECE. This is the central validity failure.

### 6.4 Stable-fact counter-regime

The 49-row stable set is also valid JSON and reproduces the contrasting high-accuracy regime:

| arm | exact-match accuracy | ECE | predicted hedge distribution |
|---|---:|---:|---|
| untuned base | 0.7551 | 0.2296 | 49 `[CONFIDENT]` |
| SFT-only | 0.8571 | 0.1520 | 49 `[CONFIDENT]` |
| TSCT | 0.8776 | 0.1602 | 46 `[CONFIDENT]`, 3 `[COND_CONFIDENT]` |

This is why the paper should argue that calibration behaviour depends on the accuracy regime, rather than claiming ECE is universally unusable.

### 6.5 Dated gold audit

The committed audit reports a Wikidata query time of `2026-08-11T10:02:20.385788+00:00` and 3,205 unique entity/property pairs:

- 2,575 `gold_is_stale`;
- 35 `gold_is_current`; and
- 595 `indeterminate`.

The stale rate among the 2,610 decided pairs is **98.659%**. This means “gold versus Wikidata at the query time,” not “gold versus final truth.”

### 6.6 Pushed three-model selective-answering ladder

Using exact match against the audit's resolvable current labels, the pushed per-example outputs reproduce:

| model | scored n | current-answer accuracy | precision in top 1% by mean log-probability | lift over base rate |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 2,951 | 0.58% | 6.90% | 11.97x |
| gemma-3-4B-it | 2,951 | 1.56% | 24.14% | 15.49x |
| gemma-4-26B-A4B-it | 2,951 | 5.32% | 65.52% | 12.31x |

This supports a promising **snapshot-level** routing result. Because each model is a single run and vintage, family, size, and training recipe are confounded, it should not be promoted into a universal scaling law.

### 6.7 Artifact integrity

The principal pushed prediction files were parsed record by record with zero JSON failures:

- five full 3,622-row base/SFT/TSCT test outputs across the reviewed arms/seeds;
- two full 3,622-row pushed comparison-model outputs;
- three 49-row stable-set outputs; and
- the 60-row mixed-paragraph stress set.

This confirms file completeness and parseability. It is not a semantic audit of every generated answer.

## 7. Evidence-qualified scientific state

### Findings safe to carry forward within this snapshot

- The original discrete `argmax` confidence path has zero calibration gradient; the differentiable expectation restores it.
- The fixed path does not, by itself, prevent output collapse. The earlier single-seed claim was refuted.
- A matched SFT/TSCT ECE benefit is not present across the two completed paired seeds; the sign changes.
- TemporalDelta test ECE is dominated by the mismatch between fixed hedge scalars and very low answer accuracy. Constant controls expose this directly.
- The pushed hedge outputs are nearly constant within relation type and therefore do not provide meaningful per-example discrimination.
- The model's answer log-probability contains much more per-example ranking information than the trained hedge output on these records.
- The reference snapshot is badly stale relative to the dated Wikidata audit, and time partitioning repeats questions across splits with outdated answers.
- Repairing current golds and removing train-seen questions does not rescue the ECE instrument in the extremely low-accuracy volatile regime.
- Selective answering/routing is the strongest constructive direction currently supported by the artifacts.

### Claims that must remain bounded or open

- The independently inferred `L_over`, `L_under`, and `R_hedge` formulas have not been signed off against the unavailable original training code.
- Qwen2.5-7B substitutes for the proposal's LLaMA-3 8B setup; a disagreement with the original Slack/Doc numbers is therefore `CONTESTED`, not proof of misconduct or fabrication.
- The original team's reported checkpoints, prediction provenance, and exact configs remain unavailable for direct comparison.
- One or two behavioural seeds are enough to refute a claimed stable effect, but not enough to estimate its variance precisely.
- The Wikidata audit is a dated secondary reference and may itself lag reality.
- The pushed cross-model ladder is confounded and single-run; the clean size-matched/post-training-matched controls remain undone.
- The mixed stable/volatile paragraph set exists but has not been run. It is the most direct test of whether a router can distinguish regimes inside one input.
- Label-smoothing SFT and RAG baselines remain unimplemented in this fork.

## 8. Local-only work after the pushed checkpoint

The protected `tcl-fix-validation` worktree contains changes that are **not** part of `origin/tcl-fix-validation@2e7ca4d`:

- a `generate_predictions.py` loader fallback through `mlx_vlm` for multimodal/KV-shared Gemma-4 checkpoints;
- completion of the previously partial Gemma-4 31B output to 3,622 rows;
- a new complete 3,622-row Gemma-4 E4B output; and
- associated driver/generation logs.

Both files parsed successfully. A provisional recount against the same 2,951 resolvable current labels gives:

| local-only model | current-answer accuracy | precision at top 1% | lift |
|---|---:|---:|---:|
| gemma-4-E4B-it | 0.81% | 13.79% | 16.96x |
| gemma-4-31B-it | 6.98% | 48.28% | 6.92x |

These provisional controls matter because they **weaken the pushed claim that lift remains constant at 12–15x**. They also supersede the ledger statement that E4B is unloadable. They must be checkpointed, reviewed, and integrated—or explicitly excluded—before the claims ledger and paper can be called current. This handoff does not commit them because they belong to a separate in-progress experiment and include source changes not yet reviewed here.

## 9. Paper state

[The current draft](../paper/paper_draft.md) is approximately **9,394 words** before bibliography conversion and is intentionally a long-form evidence draft, not a submission-ready short paper.

Completed:

- full narrative through Abstract, Introduction, Results, Discussion, Conclusion, and Limitations;
- mechanism, validity-control, benchmark-audit, and selective-answering results incorporated;
- explicit threats to validity; and
- measured Generative AI Disclosure.

Not complete:

- venue-length compression;
- conversion of newer related-work notes into full venue-format citations;
- reconciliation with the local-only E4B/31B results;
- refreshed per-claim provenance pins after `1484a68` (the ledger's frozen table predates many later claims);
- final separation between proposed protocol and protocol actually executed;
- original training-lead sign-off on the independent loss reimplementation;
- poster/post finalization; and
- a final author/team review of scope, contribution wording, and disclosure.

Venue and deadline statements in the ledger were imported from an earlier prior-art scan and were not live-reverified for this handoff. Treat them as planning notes until refreshed from the official call.

## 10. Recommended takeover order

### P0 — freeze a coherent scientific snapshot

1. Review and checkpoint the local Gemma-4 loader fix and both completed outputs in a separate experiment commit.
2. Recompute the model-control table, update or withdraw the “constant lift” wording, and update the paper and claims ledger together.
3. Refresh provenance so every quantitative table names the exact repo commit, dataset revision, model revision, prediction file hash, and query timestamp used.
4. Decide whether to merge/cherry-pick fork `main`'s CI/test commits into the research line; do not imply CI covers a branch where it is absent.

### P1 — close the highest-value validity gaps

1. Run `stress_mixed_paragraphs.jsonl` with a scoring protocol capable of evaluating claim-level routing inside mixed inputs.
2. Obtain the original training code/formulas or a written training-lead confirmation that the reimplementation is a fair substitute.
3. Run the clean controlled model pair(s) identified in the ledger, with exact model revisions and a common generation harness.
4. Either implement the label-smoothing and RAG baselines or explicitly narrow the paper so it does not promise them.

### P2 — produce the submission package

1. Cut the paper only after P0 fixes the result set. Preserve the long draft as an audit appendix/source document.
2. Reduce the main paper to one contribution spine: instrument validity and the routing alternative. Move the training bug and engineering details to an appendix.
3. Refresh citations and official venue constraints.
4. Build the poster/post from the final claim table, not from the original proposal or superseded Slack result.

## 11. Complete pushed-fork commit inventory

### Research branch: 33 commits beyond upstream

| commit | checkpoint |
|---|---|
| `6cb043f` | Add standalone TCL fix validation harness + results |
| `da53a8c` | Scale TCL validation to 7B; add first paired SFT-vs-TSCT result |
| `1484a68` | Add generation script; first held-out ECE for the project |
| `e33f926` | Add claims ledger; run oracle and constant-policy controls on ECE |
| `4ad9fa2` | Refute B5: discrimination is a between-dataset artifact, not per-example skill |
| `aabf212` | Add D1 base-model baseline: fine-tuning installs the lookup, not knowledge |
| `78f2001` | Add C5: test-split gold answers are expired values |
| `e3b6100` | Resolve D8: gold is 98.7% stale, but my consequence claim was an overclaim |
| `7696cba` | Resolve D4 as B7; append correction notice to tcl-fix-validation.md |
| `023b474` | Refute B2 on seed-1 replication; complete B7; add bottom-line summary |
| `8940140` | Add B8: the model's own logprobs predict correctness where the hedge scheme is at chance |
| `929da92` | Qualify B8: statistically overwhelming, practically weak, partly memorisation |
| `77e9676` | Add C7 and C8; record that the untuned model has the best calibration signal |
| `03c232a` | Replicate B4 at n=2: SFT baseline is stable at 0.455, not 0.85 |
| `07fa779` | Add section F: what the paper should actually claim |
| `af89bc9` | Write Results, Discussion, Conclusion, Abstract and Limitations |
| `bd27f13` | Revise Introduction and add forward-references in Methods |
| `3070dfc` | Add Generative AI Disclosure with measured Pangram scores |
| `a6b97d5` | Fix multi-EOS handling; support models with zero spare embedding rows |
| `795144e` | Add Related Work 2.6–2.9 from a literature pass; deflate three novelty claims |
| `50f9d89` | Update draft header to reflect the Related Work pass and outstanding citation work |
| `c8c5bbf` | Add B9: internal confidence discriminates at every horizon but its level runs backwards |
| `ba05b1a` | Add F5 and F6 from the Algoverse-Bias-Steering prior-art and model-set docs |
| `357fcba` | Add verbatim voice candidates mapped to the findings they produced |
| `0177c38` | Add C9: the benchmark inverts model quality |
| `fbe663e` | Add F0: the paper's organising frame is two TB3 failure modes at once |
| `990e505` | Add F-1: repairing the benchmark makes it worse, which is the real spine |
| `f99d69e` | Add F-2: the positive direction is abstention, not calibration |
| `f806a14` | Rewrite Results, Discussion and Conclusion around the new spine |
| `79a3624` | Replicate B9 at seed 1: the level is arbitrary, the ranking is not |
| `6ee053a` | Add F-2b: the abstention operating point scales with model quality |
| `a3c152f` | Add B10 and B11: the replication rate, and abstention as regime detection |
| `2e7ca4d` | Add F-2c: three-point vintage ladder, and a smaller newer model beats a bigger older one |

### Fork main: two separate operational commits

| commit | checkpoint |
|---|---|
| `a26446f` | Add a minimal CI workflow |
| `c962715` | Add a starter test for an untested module |

## 12. Handoff rule

Do not use the proposal, paper prose, a Slack table, or a summary statistic as the sole source of truth. Start from the exact branch and commit, follow the claim to its raw per-record artifact and run metadata, reproduce the aggregate, and carry the stated caveats forward. If the raw artifact and prose disagree, preserve the disagreement and update the claim status; do not silently choose the more convenient number.
