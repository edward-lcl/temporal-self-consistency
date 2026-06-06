"""
Temporal Calibration Loss (TCL) — Reference Implementation
==========================================================
TCL = CE + lambda_over * L_over + lambda_under * L_under - lambda_hedge * R_hedge

CRITICAL: the confidence value c_hat MUST be the differentiable softmax
probability over the hedge tokens. If you argmax/sample the hedge token and
THEN look up a scalar confidence, the gradient is cut at the discrete step
and the calibration terms backprop nothing — CE still trains (so SFT looks
fine) but the model never learns to hedge. This was the bug in the first
TSCT training run where every output collapsed to [CONFIDENT].

This file is a reference for the training team. It is not wired into the
eval pipeline (eval consumes generated predictions, not model internals).
"""
import torch
import torch.nn.functional as F

HEDGE_TOKENS = ["[CONFIDENT]", "[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[UNKNOWN]"]

# Eval-time scalar mapping (used for ECE, NOT for training gradient)
HEDGE_CONFIDENCE = {
    "[CONFIDENT]":      0.95,
    "[COND_CONFIDENT]": 0.75,
    "[TEMPORAL_HEDGE]": 0.45,
    "[UNKNOWN]":        0.10,
}


def compute_c_hat(logits, hedge_position, hedge_token_ids):
    """
    Differentiable confidence: softmax probability mass on the
    highest-confidence hedge token the model assigns.

    logits          : (batch, seq, vocab)
    hedge_position  : (batch,) index of the hedge token slot per example
    hedge_token_ids : list[int] the 4 hedge token vocab IDs

    Returns c_hat: (batch,) differentiable scalar in [0, 1].
    """
    batch = logits.size(0)
    # Gather logits at the hedge position for each example
    pos_logits = logits[torch.arange(batch), hedge_position]   # (batch, vocab)
    hedge_logits = pos_logits[:, hedge_token_ids]              # (batch, 4)
    hedge_probs = F.softmax(hedge_logits, dim=-1)             # (batch, 4) DIFFERENTIABLE

    # Confidence scalars as a tensor, weighted by predicted hedge probability.
    # This keeps c_hat differentiable: it's an expectation over hedge probs,
    # not a discrete lookup.
    conf_scalars = torch.tensor(
        [HEDGE_CONFIDENCE[h] for h in HEDGE_TOKENS],
        device=logits.device, dtype=hedge_probs.dtype,
    )
    c_hat = (hedge_probs * conf_scalars).sum(dim=-1)           # (batch,)
    return c_hat, hedge_probs


def temporal_calibration_loss(
    logits, labels, hedge_position, hedge_token_ids,
    is_correct, volatility_mask, gold_hedge_idx,
    lambda_over=0.5, lambda_under=0.5, lambda_hedge=0.3,
):
    """
    logits          : (batch, seq, vocab)
    labels          : (batch, seq) for CE
    hedge_position  : (batch,) hedge slot index
    hedge_token_ids : list[int]
    is_correct      : (batch,) bool — was the factual answer correct
    volatility_mask : (batch,) bool — True for time-sensitive (fast/slow) examples
    gold_hedge_idx  : (batch,) int in [0,3] — the correct hedge token index
    """
    # --- Standard CE on answer tokens (always on) ---
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1),
        ignore_index=-100,
    )

    c_hat, hedge_probs = compute_c_hat(logits, hedge_position, hedge_token_ids)

    # --- Calibration terms only fire on time-sensitive examples ---
    vmask = volatility_mask.float()
    n_volatile = vmask.sum().clamp(min=1.0)

    # Overconfidence penalty: high c_hat AND wrong  -> penalize
    wrong = (~is_correct).float()
    l_over = (vmask * wrong * c_hat).sum() / n_volatile

    # Underconfidence penalty: low c_hat AND correct -> penalize
    correct = is_correct.float()
    l_under = (vmask * correct * (1.0 - c_hat)).sum() / n_volatile

    # Hedge quality reward: probability mass on the correct hedge token
    correct_hedge_prob = hedge_probs[torch.arange(hedge_probs.size(0)), gold_hedge_idx]
    r_hedge = (vmask * correct_hedge_prob).sum() / n_volatile

    total = ce + lambda_over * l_over + lambda_under * l_under - lambda_hedge * r_hedge

    # Return components separately so they can be logged individually
    # (logging only tcl_total hid the gradient bug last time).
    return total, {
        "ce":        ce.detach().item(),
        "l_over":    l_over.detach().item(),
        "l_under":   l_under.detach().item(),
        "r_hedge":   r_hedge.detach().item(),
        "tcl_total": total.detach().item(),
        "n_volatile_in_batch": int(volatility_mask.sum().item()),
    }


# Recommended starting hyperparameters (from project debugging):
#   learning_rate = 5e-5   (2e-4 was too aggressive for 1 epoch)
#   epochs        = 3      (1 epoch was insufficient)
#   lambda_over   = 0.5
#   lambda_under  = 0.5
#   lambda_hedge  = 0.3    (all-1.0 let the hedge term dominate / collapse)
#
# Always log the per-component dict above, not just tcl_total.
