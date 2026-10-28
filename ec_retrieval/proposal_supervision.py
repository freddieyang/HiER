"""Direct supervision of existing EC parent proposals, without a new encoder.

The coherent posterior is still obtained from the same leaf probabilities.
This TRAIN-only companion loss anchors intermediate L1-L3 predictions using
the unchanged masks; missing annotations contribute no target mass.
General deep-supervision idea: Lee et al., AISTATS 2015,
https://proceedings.mlr.press/v38/lee15a.html . This is a task-specific
application, not a claim that auxiliary classification losses are novel.
"""
import math

from .reaction_ec_bridge import hierarchical_multi_positive_loss


def add_proposal_ec_supervision(loss, reaction, protein, reaction_targets,
                               protein_targets, *, weight=0., warmup=1.):
    """Keep zero-weight execution exact; active loss reuses one forward graph."""
    if not math.isfinite(weight) or weight < 0 or not 0 <= warmup <= 1:
        raise ValueError('Finite nonnegative weight and warmup in [0,1] required')
    if weight == 0:
        return loss, {}
    q, p = reaction['proposal_logits'], protein['proposal_logits']
    if len(q) != 3 or len(p) != 3 or len(reaction_targets) != 4 or len(protein_targets) != 4:
        raise ValueError('The controlled auxiliary uses L1-L3 of a four-level EC hierarchy')
    qloss = hierarchical_multi_positive_loss(q, reaction_targets[:3])
    ploss = hierarchical_multi_positive_loss(p, protein_targets[:3])
    auxiliary = .5 * (qloss + ploss)
    weighted = (weight * warmup) * auxiliary
    return loss + weighted, dict(proposal_ec=auxiliary, proposal_ec_reaction=qloss,
                                proposal_ec_protein=ploss, proposal_ec_weighted=weighted)
