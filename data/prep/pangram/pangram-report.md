# Pangram voice-fidelity check — TSCT paper draft

Run 2026-08-11 against `paper/paper_draft.md`, using the `pangram-voice` skill
from `~/Projects/tbench3-archive`. Prose only: tables, headers, code and
italic meta-notes stripped, per that skill's guidance that data-enumeration
windows dilute the signal.

| section | words | verdict | ai_assisted | human | worst window |
|---|---|---|---|---|---|
| abstract | 391 | AI | 1.00 | 0.00 | 0.97 |
| intro (revised portion) | 741 | AI | 1.00 | 0.00 | 0.99 |
| results | 1017 | **Mixed** | 0.74 | 0.26 | 0.99 |
| discussion | 934 | AI | 1.00 | 0.00 | 0.99 |
| conclusion | 297 | AI | 1.00 | 0.00 | 0.99 |

## Reading

Exactly the documented pattern: interpretive/thesis prose pegs at ~0.99,
data-enumeration reads human and dilutes (Results is the only Mixed section,
and it is the one carrying the tables and numbers).

Per the skill's governing rule, **these scores cannot be honestly lowered by
rewriting** — that corpus showed agent narrative prose is robust to rework,
plateauing ~0.65 across four rounds and four techniques. The only honest dial
is verbatim-density: real human material scores Human (0.02-0.19), assembled
verbatim+connective scores Mixed (~0.59), fully agent-drafted scores AI (0.99).

## What would actually move these

1. **Author-written or dictated interpretive prose** for §1, §5, §6 — the
   framing paragraphs, not the numbers.
2. **Verbatim source material.** Several findings in this paper originated in
   the author's own framing rather than the agent's, and using those words
   directly is both more accurate as attribution and the legitimate way to
   raise human-density. Candidates: the pattern-matching-vs-reasoning question
   that produced B5, the observation about a reference snapshot decaying that
   produced C6, and the pushback on a 0.005 delta that produced section F.
3. **Disclosure**, which is already added to the paper and is the end-state
   this skill endorses.

Nothing here should be rewritten to chase a number.
