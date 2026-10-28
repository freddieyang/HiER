"""Choose hard queries by sampled rank errors, then optimize existing CE losses.

No gradient is propagated through discrete ranking. The original all-positive
cross-entropy supplies the gradient, including for badly outranked positives.
Unobserved candidates are comparison items, not experimentally verified negatives.
"""
from __future__ import annotations

import math
import torch

from .losses import _directional_multi_positive_loss_vectors, multi_positive_tail_risk_loss


@torch.no_grad()
def sampled_excess_rank(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Average unobserved items preceding each known positive, per valid row.

    Scores/masks: [B,C]. Output: [B_valid]. The identity is mean positive rank
    minus (m+1)/2, with stable descending ordering as in the benchmark. Compute
    integer inversion counts before dividing by m to avoid cancellation.
    """
    if scores.ndim != 2 or scores.shape != positive_mask.shape:
        raise ValueError('Aligned [B,C] scores and positive mask required')
    mask = positive_mask.bool()
    valid = mask.any(1)
    if not bool(valid.any()):
        raise ValueError('No row contains an observed positive')
    mask = mask[valid]
    order = torch.argsort(scores.detach()[valid].float(), dim=1, descending=True, stable=True)
    sorted_positive = mask.gather(1, order)  # [B_valid,C]
    unobserved_before = (~sorted_positive).long().cumsum(1)
    inversion_count = (unobserved_before * sorted_positive).sum(1)
    return inversion_count.float() / mask.sum(1)


@torch.no_grad()
def select_tail_indices(all_losses: torch.Tensor, excess_rank: torch.Tensor,
                        tail_fraction: float, selector: str) -> torch.Tensor:
    """Rank errors take priority; CE only breaks exact rank-error ties."""
    if not 0 < tail_fraction <= 1:
        raise ValueError('tail_fraction must be in (0,1]')
    if all_losses.ndim != 1 or all_losses.shape != excess_rank.shape or not all_losses.numel():
        raise ValueError('Nonempty aligned query vectors required')
    count = max(1, math.ceil(tail_fraction * all_losses.numel()))
    if selector == 'raw':
        return torch.topk(all_losses.detach(), k=count, largest=True).indices
    if selector != 'excess_rank':
        raise ValueError('Unknown tail selector')
    secondary = torch.argsort(all_losses.detach(), descending=True, stable=True)
    primary = torch.argsort(excess_rank[secondary], descending=True, stable=True)
    return secondary[primary[:count]]


def rank_selected_tail_risk_loss(scores: torch.Tensor, positive_mask: torch.Tensor, *,
        all_positive_weight: float = .2, tail_risk_weight: float = .1,
        tail_fraction: float = .2, tail_selector: str = 'raw'):
    """Preserve the base losses/weights and vary only the tail query selector.

    `raw` calls the unchanged original implementation. `excess_rank` selects
    queries by discrete, detached rank errors and averages their original CE.
    This is rank-guided hard-query training, not differentiable MR or CVaR of MR.
    """
    if not 0 <= all_positive_weight <= 1 or not 0 <= tail_risk_weight <= 1:
        raise ValueError('Loss weights must be in [0,1]')
    if tail_risk_weight == 0:
        # No difficult-query selection or rank diagnostics in the adopted
        # objective. Keep all-positive supervision and FP32 reductions.
        if tail_selector not in ('raw', 'excess_rank'):
            raise ValueError('Unknown tail selector')
        return multi_positive_tail_risk_loss(
            scores, positive_mask, all_positive_weight=all_positive_weight,
            tail_risk_weight=0., tail_fraction=tail_fraction)
    any_losses, all_losses = _directional_multi_positive_loss_vectors(scores, positive_mask)
    excess = sampled_excess_rank(scores, positive_mask)
    indices = select_tail_indices(all_losses, excess, tail_fraction, tail_selector)
    if tail_selector == 'raw':
        total, diagnostics = multi_positive_tail_risk_loss(scores, positive_mask,
            all_positive_weight=all_positive_weight, tail_risk_weight=tail_risk_weight,
            tail_fraction=tail_fraction, tail_risk_score='raw')
    else:
        any_mean, all_mean = any_losses.mean(), all_losses.mean()
        tail_mean = all_losses[indices].mean()
        base = (1 - all_positive_weight) * any_mean + all_positive_weight * all_mean
        total = (1 - tail_risk_weight) * base + tail_risk_weight * tail_mean
        diagnostics = dict(any_positive=any_mean, all_positive=all_mean, tail_positive=tail_mean)
    with torch.no_grad():
        count = positive_mask.bool().sum(1)
        count = count[count > 0].float()
        diagnostics.update(tail_selected_excess_rank=excess[indices].mean(),
            batch_excess_rank=excess.mean(), tail_selected_positive_count=count[indices].mean(),
            batch_positive_count=count.mean(), rank_error_zero_fraction=(excess == 0).float().mean())
    return total, diagnostics
