"""Learn a contextual residual without altering the reference-route gradient.

Two disjoint parameter groups learn simultaneously from the same TRAIN batch.
This is not distillation from a separately trained checkpoint or an ensemble.
The contextual loss reuses the original retrieval/EC objective and all masks.
"""
import math
import torch

from .reverse_pool_extension import expanded_bidirectional_loss, isolated_extra_rng
from .reaction_ec_bridge import hierarchical_multi_positive_loss, hierarchy_consistency_loss, cross_modal_ec_alignment_loss


def isolated_context_objective(model, reaction, protein, extra, reaction_features,
                               extra_features, mask, extended_mask, qt, pt, *,
                               warmup, reverse_weight, all_positive_weight,
                               tail_risk_weight, tail_fraction, ec_weight,
                               consistency_weight, alignment_weight, dropout_seed):
    """Return the original objective on the residual route; shared grads stop.

    Reaction anchors [Q,D] and extra reverse candidates [C-Q,D] use their
    already computed (detached) states. Protein candidates are detached [P,I].
    The same complete TRAIN positive masks [Q,P] / [C,P] enter both routes.
    """
    if extra is None or extended_mask is None:
        raise ValueError('Isolated context comparison fixes the reverse512 pool')
    # Shared EC heads retain their TRAIN dropout. Isolate the additional draws
    # from the reference stream, just as for the existing extra reaction pool.
    with isolated_extra_rng(reaction['state'].device, dropout_seed):
        refined = model.detached_context_reaction(reaction, reaction_features)
        refined_extra = model.detached_context_reaction(extra, extra_features)
    p = protein['embedding'].detach()
    scale = model.logit_scale.float().detach().clamp(max=math.log(100.)).exp()
    scores = scale * (refined['embedding'] @ p.T)
    extra_scores = scale * (refined_extra['embedding'] @ p.T)
    retrieval, _ = expanded_bidirectional_loss(scores, mask, extra_scores, extended_mask,
        reverse_weight=reverse_weight, mode='legacy', all_positive_weight=all_positive_weight,
        tail_risk_weight=tail_risk_weight, tail_fraction=tail_fraction, reverse_reduction='enzyme_mean')
    protein_logits = [x.detach() for x in protein['logits']]
    ec = .5 * (hierarchical_multi_positive_loss(refined['logits'], qt)
               + hierarchical_multi_positive_loss(protein_logits, pt))
    parents = model.ec_router.parent_maps()
    consistency = .5 * (hierarchy_consistency_loss(refined['logits'], parents)
                        + hierarchy_consistency_loss(protein_logits, parents))
    edges = mask.nonzero()
    alignment = cross_modal_ec_alignment_loss(refined['logits'],
        [x.index_select(0, edges[:, 1]) for x in protein_logits], edges[:, 0])
    loss = retrieval + warmup * (ec_weight*ec + consistency_weight*consistency
                                 + alignment_weight*alignment)
    return loss, dict(context_objective=loss, context_retrieval=retrieval, context_ec=ec,
                      context_alignment=alignment, context_consistency=consistency)
