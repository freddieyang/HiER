"""Two directions of the existing multi-positive HiER retrieval objective.

Reuse scores [reaction_batch, enzyme_pool] and the complete TRAIN positive mask.
The reverse direction skips enzyme anchors with no observed positive in the
current reaction batch. Unobserved pairs remain comparison items, not verified
negatives. No encoder, sampling, temperature or inference operation is added.
"""
from __future__ import annotations
import math
import torch
from .positive_coverage import positive_coverage_loss


def bidirectional_retrieval_loss(scores: torch.Tensor, positive_mask: torch.Tensor, *,
        reverse_weight: float = 0., mode: str = 'legacy', all_positive_weight: float = .2,
        tail_risk_weight: float = .1, tail_fraction: float = .2):
    """Average directional objectives with a fixed, explicitly configured weight.

    reverse_weight=0 reproduces the old objective and diagnostics exactly.
    reverse_weight=.5 gives equal mean loss to the two directions, regardless
    of their different anchor counts. Each direction retains its own positive
    coverage and hard-query tail, including the old FP32 logsumexp safeguards.
    """
    if not math.isfinite(reverse_weight) or not 0 <= reverse_weight <= 1:
        raise ValueError('reverse_weight must be finite and in [0,1]')
    settings=dict(mode=mode, all_positive_weight=all_positive_weight,
                  tail_risk_weight=tail_risk_weight, tail_fraction=tail_fraction)
    forward, rdiag=positive_coverage_loss(scores, positive_mask, **settings)
    if reverse_weight == 0:
        return forward, rdiag
    # [B,C] -> [C,B]. Existing directional code masks anchors without positives
    # BEFORE taking the mean or choosing the hard-query tail.
    reverse, ediag=positive_coverage_loss(scores.T, positive_mask.T, **settings)
    total=(1-reverse_weight)*forward + reverse_weight*reverse
    diagnostic={key: (1-reverse_weight)*rdiag[key]+reverse_weight*ediag[key]
                for key in ('any_positive', 'all_positive', 'tail_positive')}
    diagnostic.update({f'r2e_{key}':value for key,value in rdiag.items()})
    diagnostic.update({f'e2r_{key}':value for key,value in ediag.items()})
    with torch.no_grad():
        mask=positive_mask.bool()
        diagnostic.update(r2e_retrieval=forward.detach(), e2r_retrieval=reverse.detach(),
            r2e_valid_anchor_count=mask.any(1).sum().float(),
            e2r_valid_anchor_count=mask.any(0).sum().float(),
            e2r_skipped_anchor_count=(~mask.any(0)).sum().float(),
            observed_positive_count=mask.sum().float())
    return total, diagnostic
