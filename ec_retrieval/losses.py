"""Losses for positive-set retrieval."""

from __future__ import annotations

import math

import torch


def _directional_multi_positive_loss_vectors(
    scores: torch.Tensor, positive_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must be aligned 2-D tensors")
    positive_mask = positive_mask.bool()
    valid_rows = positive_mask.any(dim=1)
    if not bool(valid_rows.any()):
        raise ValueError("No row contains an observed positive")
    # Log-sum-exp in FP32 avoids precision loss under BF16/FP16 autocast while
    # preserving gradients to the original score tensor.
    selected_scores = scores[valid_rows].float()
    selected_mask = positive_mask[valid_rows]
    positive_scores = selected_scores.masked_fill(~selected_mask, -torch.inf)
    denominator = torch.logsumexp(selected_scores, dim=1)
    any_positive = denominator - torch.logsumexp(positive_scores, dim=1)
    positive_count = selected_mask.sum(dim=1).clamp_min(1)
    mean_positive_score = (
        selected_scores.masked_fill(~selected_mask, 0.0).sum(dim=1)
        / positive_count
    )
    all_positive = denominator - mean_positive_score
    return any_positive, all_positive


def _directional_multi_positive_loss(
    scores: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    any_positive, _ = _directional_multi_positive_loss_vectors(
        scores, positive_mask
    )
    return any_positive.mean()


def multi_positive_infonce(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    symmetric: bool = True,
) -> torch.Tensor:
    """Multi-positive contrastive loss over observed EC-protein associations.

    Unobserved pairs only enter the softmax denominator. They are not persisted
    or re-labelled as experimentally verified negatives.
    """
    forward = _directional_multi_positive_loss(scores, positive_mask)
    if not symmetric:
        return forward
    backward = _directional_multi_positive_loss(scores.T, positive_mask.T)
    return 0.5 * (forward + backward)


def multi_positive_tail_risk_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    all_positive_weight: float = 0.15,
    tail_risk_weight: float = 0.10,
    tail_fraction: float = 0.20,
    symmetric: bool = False,
    tail_risk_score: str = "raw",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Blend easy-positive, all-positive, and hard-query retrieval risks.

    The ordinary multi-positive InfoNCE numerator is a log-sum-exp over known
    positives, so a single easy positive can nearly satisfy a query.  The
    all-positive term instead averages the log-softmax loss of every observed
    positive in the sampled pool.  The tail term averages the largest
    all-positive query losses in the batch, directing a bounded share of the
    gradient toward hard queries without labelling unobserved pairs as verified
    negatives.

    Both weights are convex mixture coefficients in ``[0, 1]``.  Setting them
    to zero exactly recovers directional ``multi_positive_infonce`` (up to the
    intentional FP32 reduction used for numerical stability).

    HiER explicitly uses ``tail_risk_weight=0.0``: every valid query has equal weight
    and tail selection is skipped. In that mode the zero ``tail_positive``
    diagnostic denotes an inactive term, not a measured tail. The nonzero
    function default remains available for historical callers; the current
    training entry explicitly supplies zero.

    ``positive_count_corrected`` subtracts the irreducible ``log(m)``
    all-positive entropy before selecting the CVaR queries. Here ``m`` is the
    number of observed positives in the current candidate pool. This changes
    which queries receive tail emphasis, not the within-query derivatives.
    The default ``raw`` preserves historical experiments.
    """

    if not 0.0 <= all_positive_weight <= 1.0:
        raise ValueError("all_positive_weight must lie in [0, 1]")
    if not 0.0 <= tail_risk_weight <= 1.0:
        raise ValueError("tail_risk_weight must lie in [0, 1]")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if tail_risk_score not in {"raw", "positive_count_corrected"}:
        raise ValueError("Unknown tail_risk_score")

    directions = [(scores, positive_mask)]
    if symmetric:
        directions.append((scores.T, positive_mask.T))

    totals: list[torch.Tensor] = []
    any_means: list[torch.Tensor] = []
    all_means: list[torch.Tensor] = []
    tail_means: list[torch.Tensor] = []
    for directional_scores, directional_mask in directions:
        any_losses, all_losses = _directional_multi_positive_loss_vectors(
            directional_scores, directional_mask
        )
        any_mean = any_losses.mean()
        all_mean = all_losses.mean()
        if tail_risk_weight == 0.0:
            totals.append((1.0 - all_positive_weight) * any_mean
                          + all_positive_weight * all_mean)
            any_means.append(any_mean)
            all_means.append(all_mean)
            tail_means.append(all_mean.new_zeros(()))
            continue
        tail_count = max(1, math.ceil(tail_fraction * all_losses.numel()))
        if tail_risk_score == "positive_count_corrected":
            positive_count = directional_mask.bool().sum(dim=1)
            positive_count = positive_count[positive_count > 0].float()
            tail_scores = all_losses - positive_count.log()
        else:
            tail_scores = all_losses
        tail_mean = torch.topk(tail_scores, k=tail_count, largest=True).values.mean()
        base = (
            (1.0 - all_positive_weight) * any_mean
            + all_positive_weight * all_mean
        )
        total = (1.0 - tail_risk_weight) * base + tail_risk_weight * tail_mean
        totals.append(total)
        any_means.append(any_mean)
        all_means.append(all_mean)
        tail_means.append(tail_mean)

    diagnostics = {
        "any_positive": torch.stack(any_means).mean(),
        "all_positive": torch.stack(all_means).mean(),
        "tail_positive": torch.stack(tail_means).mean(),
    }
    return torch.stack(totals).mean(), diagnostics
