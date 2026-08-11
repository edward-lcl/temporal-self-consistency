# Verbatim voice candidates — Edward

Extracted 2026-08-11 from the Claude Code session transcripts for this project
(`~/.claude/projects/-Users-edward-Projects-temporal-self-consistency--claude-worktrees-.../`).
24 messages, 1,705 words. Full text: `edward_verbatim.txt`.

## Why this file exists, and its limit

The `pangram-voice` skill's rule is that the only honest way to raise
human-density is to bring in **genuine human input** — verbatim speech, the
author's own typing — never by paraphrasing agent prose. This is that source
material.

**The limit, which the author identified himself:** these are messages
*directing the work*, not *stating the paper's claims*. They can supply
motivation and framing — the "why we looked here" sentences — but not the
technical assertions, which the author would have to write or say properly.
**Nothing here has been spliced into the paper.** These are candidates.

## The four passages that became findings

Each of these is a case where the author's framing produced a result the agent
was not pursuing. Using his words for those framings is more accurate as
attribution, not only better for voice.

### → B5 / §5.3 — the pattern-matching test
> "but even if discrimination improves as models get bigger and they improve,
> how would we test if its pattern matching vs reasoning."

This question produced the within-relation stratification that collapsed the
apparent AUROC 0.67 to 0.4997 and established that the hedge output is a
relation-level template. The agent had reported the pooled figure without
stratifying.

### → §5.5 / C6 — the reference decays too
> "this is a genuinely more difficult question since google reorg was this week"
> *(with a link to a leadership change six days before our Wikidata query)*

This produced C6: our staleness audit compares gold against Wikidata at a pinned
timestamp, and Wikidata has its own lag, so the audit measures gold-versus-a-dated-
snapshot rather than gold-versus-truth. The agent had been treating the live
source as ground truth.

### → F1 / §5.2 — the reframing
> "But these things, they're changing my zero point zero zero five. Like, that's
> so small. Like, that doesn't seem like significantly relevant at all... that's
> why we need, like, the best possible framing or even look for, like, the
> alternative methods to the approach rather than what we set out to do."

This forced the move from reporting a null as the result to asking why the null
was structurally guaranteed — which became the ECE validity condition, the
paper's headline.

### → §5.1 / B9 — the unknown unknown
> "the most important question is always the one that's always the unknown, the
> unknown of the unknown. Like, we don't know what the right question to be
> asking is."

And on process, the framing that drove the whole audit:
> "how I see it is, like, the OODA loop. So, like, we would observe new things as
> we run more tests, and then we would orientate differently, decide differently,
> and act."

## Suggested use

1. **§5 Discussion opening** — the OODA passage, cleaned to written register by
   the author, is a legitimate account of how the investigation actually
   proceeded, and workshop reviewers weighing reproducibility respond to that.
2. **§5.3** — the pattern-matching question, in his words, as the motivation for
   stratifying.
3. **§5.5** — the reorg observation as the motivation for C6.
4. **Acknowledgements or the AI disclosure** — recording which framings
   originated with the author rather than the agent is the honest version of the
   authorship statement, and it is materially true here.

These are spoken-register transcripts. They need the author to rewrite them for
a paper, which is the point — that is what makes the resulting prose his.
