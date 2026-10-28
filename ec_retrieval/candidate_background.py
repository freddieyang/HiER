"""Feature-dependent candidate bias within the existing single retrieval index.

One TRAIN-learned latent reference b gives the candidate term -<b,z_p>.
It is not an enzyme-ID lookup, a measured biological specificity, or a CSLS
density estimate. EC posteriors and their agreement are left intact.
"""
import math

import torch
from torch import nn

from .residue_readout import ResidueReadoutRetriever


class CandidateBackgroundRetriever(ResidueReadoutRetriever):
    """s = scale * [(1-a)<z_q-b,z_p> + a/L sum_l BC(q_l,p_l)].

    Zero initialization exactly retains the existing score without consuming
    random draws. The bounded reference is jointly learned with the unchanged
    retrieval objective. Both sides remain independently pre-indexable.
    """

    def __init__(self, *, candidate_background='learned', background_max_norm=1., **kwargs):
        super().__init__(**kwargs)
        if candidate_background != 'learned' or not math.isfinite(background_max_norm) or background_max_norm <= 0:
            raise ValueError('Learned candidate background requires a positive explicit norm bound')
        self.candidate_background = candidate_background
        self.background_max_norm = float(background_max_norm)
        self.background_raw = nn.Parameter(torch.zeros(self.embedding_dim))

    def background_vector(self):
        raw = self.background_raw.float()
        # Norm strictly below the configured bound; finite derivative at zero.
        return self.background_max_norm * raw / (1. + raw.norm())

    def encode_reaction(self, features):
        result = super().encode_reaction(features)
        packed = result['embedding']  # [B,D + sum_l K_l]
        correction = (1. - self.ec_weight()).sqrt() * self.background_vector()
        result['embedding'] = torch.cat((packed[:, :self.embedding_dim] - correction,
                                         packed[:, self.embedding_dim:]), dim=-1)
        return result

    def explain_scores(self, reaction, protein):
        terms = super().explain_scores(reaction, protein)
        scale = self.logit_scale.float().clamp(max=math.log(100.)).exp()
        bias = -scale * (1. - self.ec_weight()) * (self.background_vector() @ protein['latent_embedding'].T)
        terms['candidate_background'] = bias.unsqueeze(0).expand_as(terms['total'])
        terms['total'] = terms['total'] + terms['candidate_background']
        return terms


def make_residue_retriever(kwargs):
    """Load old checkpoints unchanged; dispatch only explicitly tagged models."""
    if kwargs.get('matched_ablation', 'none') != 'none':
        from .matched_ablation import MatchedAblationRetriever
        return MatchedAblationRetriever(**kwargs)
    if kwargs.get('atomic_readout', 'none') != 'none':
        if any(kwargs.get(k, 'none') != 'none' for k in ('reaction_context', 'candidate_background')):
            raise ValueError('Do not combine atomic readout with other experimental branches')
        from .atomic_readout import AtomicReadoutRetriever
        return AtomicReadoutRetriever(**kwargs)
    if 'atomic_readout' in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop('atomic_readout')
    if kwargs.get('reaction_context', 'none') != 'none':
        if kwargs.get('candidate_background', 'none') != 'none':
            raise ValueError('Do not combine context and candidate-background experiments')
        from .molecule_context import MoleculeContextRetriever
        return MoleculeContextRetriever(**kwargs)
    if 'reaction_context' in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop('reaction_context')
    if kwargs.get('candidate_background') == 'learned':
        return CandidateBackgroundRetriever(**kwargs)
    return ResidueReadoutRetriever(**kwargs)


def validation_selection_key(metrics, policy):
    """Keys are minimized; no TEST values participate in either policy."""
    if policy == 'mean_rank':
        return (float(metrics['official_reactzyme_mean_rank']),)
    if policy == 'hit1_mrr':
        return (-float(metrics['hit@1']), -float(metrics['official_reactzyme_mrr']),
                float(metrics['official_reactzyme_mean_rank']))
    if policy == 'bidirectional_hit1_mrr':
        from .bidirectional_validation import joint_head_values
        joint = joint_head_values(metrics)
        return (-joint['joint_hit1_geomean'], -joint['joint_mrr_geomean'])
    raise ValueError('Unknown checkpoint selection policy')
