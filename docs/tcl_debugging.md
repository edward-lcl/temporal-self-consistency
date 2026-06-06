# TCL Debugging Notes

A record of the calibration-loss failure encountered in the first full TSCT
training run, and the diagnostic path that identified it. Kept so the team
doesn't re-discover the same problem.

## Symptom

All six checkpoints (SFT x3 seeds, TSCT x3 seeds) emitted `[CONFIDENT]` for
100% of test predictions. ECE was catastrophic (~0.86): the model claimed
0.95 confidence while being correct ~8% of the time. TSCT and SFT produced
near-identical outputs.

## What the loss curves showed

- **train/loss**: SFT runs converged normally (to <0.5). TSCT runs dropped from
  ~15 to ~10 then plateaued.
- **eval/loss**: TSCT flat at ~2.4 the entire run (no generalization). SFT
  reached ~0.85.
- **tcl_total**: moved from ~1.636 to ~1.627 over the whole run — under 1% of
  its value — then oscillated inside that band. A loss that never leaves its
  init band means gradients are near zero or not connected to params.

## Root cause (identified with mentor)

The SFT runs converging only proves the **CE path and data loading work**. It
says nothing about whether the **TCL gradient path is connected**.

Prime suspect: how `c_hat` (the confidence fed into TCL) is computed in the
forward pass. The design requires `c_hat` to be the **differentiable softmax
probability** over the hedge tokens. If the hedge token is selected via
argmax/sampling and the scalar confidence is looked up *after* that discrete
step, the gradient is cut at the non-differentiable operation. CE still trains
(answer tokens are unaffected), which is exactly why SFT looked normal while
the calibration terms backprop nothing.

## Diagnostic checklist (do this BEFORE rewriting/retraining)

1. **Log each term separately**: `L_over`, `L_under`, `R_hedge`, `CE` — not just
   `tcl_total`. If only CE moves, the calibration path is dead.
2. **Inspect `c_hat`**: confirm it is the differentiable softmax prob, not a
   post-argmax / post-sampling scalar lookup.
3. **Check batch composition**: TCL only fires on time-sensitive (fast/slow)
   examples. If filtering leaves ~0 volatile examples per batch, gradients are
   sparse to zero. Log volatile-count-per-batch.

A 50-100 step diagnostic run with this logging pinpoints the cause in under an
hour — no need to change hyperparameters blindly first.

## The correct pattern

See `src/training/tcl_loss.py`. The key function `compute_c_hat` keeps the
gradient intact by taking an expectation over the differentiable hedge-token
softmax rather than a discrete lookup:

```python
hedge_probs = F.softmax(hedge_logits, dim=-1)        # differentiable
c_hat = (hedge_probs * conf_scalars).sum(dim=-1)     # still differentiable
```

NOT:

```python
hedge_id = torch.argmax(hedge_logits)                # discrete -> gradient cut
c_hat = HEDGE_CONFIDENCE[hedge_id]                    # lookup -> no gradient
```

## Hyperparameters (revised after debugging)

| Param | Original | Revised | Reason |
|---|---|---|---|
| epochs | 1 | 3 | one pass insufficient to learn hedge distribution |
| learning_rate | 2e-4 | 5e-5 | 2e-4 too aggressive for a single epoch |
| lambda_over | 1.0 | 0.5 | all-1.0 was an untuned default |
| lambda_under | 1.0 | 0.5 | calibration should be auxiliary to CE |
| lambda_hedge | 1.0 | 0.3 | hedge reward dominating led to collapse |
