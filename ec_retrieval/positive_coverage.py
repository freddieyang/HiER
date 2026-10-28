"""Positive-set coverage without competition between known positive enzymes.

For scores s [B,C], positives P and unobserved comparison items U, define
R = log(1 + mean_p sum_u exp(s_u - s_p)). The exponential sum upper-bounds
the mean number of unobserved items outranking a positive. R therefore bounds
log(1 + sampled excess rank), not Mean Rank itself. Factorization into two
log-sum-exp reductions avoids a [B,P,C] tensor. Unknown pairs remain unknown.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .losses import _directional_multi_positive_loss_vectors
from .rank_selected_tail import rank_selected_tail_risk_loss, sampled_excess_rank


def positive_coverage_vectors(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Return one FP32 coverage risk per query containing a known positive."""
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("Aligned [B,C] scores and positive mask required")
    mask = positive_mask.bool()
    valid = mask.any(1)
    if not bool(valid.any()):
        raise ValueError("No row contains an observed positive")
    x, mask = scores[valid].float(), mask[valid]  # [B_valid,C]
    has_comparisons = (~mask).any(1)
    unobserved = x.masked_fill(mask, -torch.inf)
    # Prevent undefined logsumexp backward on an all--inf, all-positive row.
    unobserved = unobserved.masked_fill(~has_comparisons[:, None], 0.)
    log_comparison_mass = torch.logsumexp(unobserved, dim=1)
    log_inverse_positive_mean = (torch.logsumexp((-x).masked_fill(~mask, -torch.inf), dim=1)
                                 - mask.sum(1).float().log())
    risks = F.softplus(log_comparison_mass + log_inverse_positive_mean)
    return torch.where(has_comparisons, risks, x.sum(1) * 0.)


def positive_coverage_loss(scores: torch.Tensor, positive_mask: torch.Tensor, *,
        mode: str = "legacy", all_positive_weight: float = .2,
        tail_risk_weight: float = .1, tail_fraction: float = .2):
    """Keep the any-positive branch and weights, replace only coverage risk.

    The legacy branch calls the unchanged loss exactly. The factorized branch
    uses the new query risk both for the all-positive term and its upper tail.
    It adds no temperature, margin, trainable parameter or inference operation.
    """
    if mode == "legacy":
        return rank_selected_tail_risk_loss(scores, positive_mask,
            all_positive_weight=all_positive_weight, tail_risk_weight=tail_risk_weight,
            tail_fraction=tail_fraction, tail_selector="raw")
    if mode != "factorized":
        raise ValueError("Unknown positive coverage mode")
    if not 0 <= all_positive_weight <= 1 or not 0 <= tail_risk_weight <= 1:
        raise ValueError("Loss weights must lie in [0,1]")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0,1]")
    any_losses, old_all = _directional_multi_positive_loss_vectors(scores, positive_mask)
    coverage = positive_coverage_vectors(scores, positive_mask)
    count = max(1, math.ceil(tail_fraction * coverage.numel()))
    indices = torch.topk(coverage.detach(), count).indices
    any_mean, coverage_mean, tail_mean = any_losses.mean(), coverage.mean(), coverage[indices].mean()
    base = (1 - all_positive_weight) * any_mean + all_positive_weight * coverage_mean
    total = (1 - tail_risk_weight) * base + tail_risk_weight * tail_mean
    diagnostics = dict(any_positive=any_mean, all_positive=old_all.mean(),
        positive_coverage=coverage_mean, tail_positive=tail_mean)
    with torch.no_grad():
        excess = sampled_excess_rank(scores, positive_mask)
        sizes = positive_mask.bool().sum(1)
        sizes = sizes[sizes > 0].float()
        diagnostics.update(batch_excess_rank=excess.mean(), batch_positive_count=sizes.mean(),
            tail_selected_excess_rank=excess[indices].mean(), tail_selected_positive_count=sizes[indices].mean(),
            rank_error_zero_fraction=(excess == 0).float().mean())
    return total, diagnostics
