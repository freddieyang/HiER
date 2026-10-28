"""Average E->R supervision within TRAIN reaction groups before across groups.

The scores, positive sets, CE vectors and mixture weights are unchanged. This
changes only the unit used to reduce the reverse objective: a sampled reaction
with many observed enzymes receives the same total base weight as one with few.
No identity or observed EC annotation is supplied to the inference encoder.
"""
from __future__ import annotations

import math
import torch

from .losses import _directional_multi_positive_loss_vectors
from .rank_selected_tail import sampled_excess_rank


def reaction_macro_reverse_loss(scores, positive_mask, *, all_positive_weight=.2,
                                tail_risk_weight=.1, tail_fraction=.2):
    """Reduce [E,R] reverse scores through observed TRAIN groups, vectorized.

    An enzyme with several observed positive reactions participates in each of
    those groups. Reactions without a sampled positive are absent, not negative
    groups. The upper-tail term selects groups using mean all-positive CE.
    """
    if not 0<=all_positive_weight<=1 or not 0<=tail_risk_weight<=1 or not 0<tail_fraction<=1:
        raise ValueError('Invalid existing loss mixture weights')
    any_loss,all_loss=_directional_multi_positive_loss_vectors(scores,positive_mask)
    mask=positive_mask.bool()
    valid=mask.any(1)
    membership=mask[valid].T  # [R,E_valid]
    present=membership.any(1)
    membership=membership[present].float()  # [R_present,E_valid]
    degree=membership.sum(1)  # [R_present]
    # Avoid BF16 matrix reductions changing the intended per-group weights.
    with torch.amp.autocast(device_type=scores.device.type,enabled=False):
        grouped=membership@torch.stack((any_loss,all_loss),dim=1)/degree[:,None]
    any_mean,all_mean=grouped.mean(0).unbind()
    tail_count=max(1,math.ceil(tail_fraction*len(grouped)))
    selected=grouped[:,1].detach().topk(tail_count).indices
    tail_mean=grouped[selected,1].mean()
    total=(1-tail_risk_weight)*((1-all_positive_weight)*any_mean+all_positive_weight*all_mean)+tail_risk_weight*tail_mean
    diagnostics=dict(any_positive=any_mean,all_positive=all_mean,tail_positive=tail_mean)
    with torch.no_grad(),torch.amp.autocast(device_type=scores.device.type,enabled=False):
        excess=sampled_excess_rank(scores,mask)
        positives=mask[valid].sum(1).float()
        group_excess=membership@excess/degree
        group_positive=membership@positives/degree
        diagnostics.update(batch_excess_rank=excess.mean(),batch_positive_count=positives.mean(),
            rank_error_zero_fraction=(excess==0).float().mean(),
            tail_selected_excess_rank=group_excess[selected].mean(),
            tail_selected_positive_count=group_positive[selected].mean(),
            reaction_macro_excess_rank=group_excess.mean(),
            reaction_group_count=scores.new_tensor(len(grouped),dtype=torch.float32),
            maximum_group_enzyme_count=degree.max(),
            reaction_macro_reduction=scores.new_tensor(1.,dtype=torch.float32))
    return total,diagnostics
