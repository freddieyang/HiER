"""Registered single-component controls of the adopted no-stereo/no-aux HiER.

Direct EC prediction still uses all four consistent marginals for matching.
These models never accept true EC annotations at inference.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .fine_ec_retriever import FineECFunctionalRetriever, LeafMarginalRouter, grouped_log_marginal
from .residue_readout import ResidueReadoutRetriever
from . import hierarchical_multi_positive_loss, hierarchy_consistency_loss, cross_modal_ec_alignment_loss

ARMS = ('no_ec_branch', 'no_parent_conditioning', 'mean_residue_pooling', 'no_ec_alignment',
        'no_latent_similarity', 'no_ec_agreement')


class NoECRouter(nn.Module):
    """Parameter-free bypass: no EC logits, labels, or score blocks."""
    depth = 0
    vocab_sizes = ()

    def parent_maps(self):
        return []

    def forward(self, state):
        return dict(logits=[], probabilities=[], confidence=state.new_empty((len(state), 0)))


class DirectECRouter(LeafMarginalRouter):
    """One shared direct L4 MLP, with the unchanged L1--L3 marginalization.

The parent embeddings, parent probability prior, and recursive proposals are
removed together. The MLP keeps the baseline hidden width and activation.
This control has fewer parameters; it is not a parameter-matched comparison.
"""
    def __init__(self, full, hidden_dim, dropout):
        super().__init__(full)
        del self.heads, self.prefix_embeddings
        # New-head initialization must not shift encoder/sampler RNG streams.
        with torch.random.fork_rng(devices=[]):
            self.direct_head = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, self.vocab_sizes[-1]))

    def forward(self, state):
        logits = [None] * self.depth
        logits[-1] = self.direct_head(self.state_norm(state)).float().log_softmax(-1)
        for level in range(self.depth - 1, 0, -1):
            logits[level - 1] = grouped_log_marginal(
                logits[level], getattr(self, f'parent_index_{level}'),
                getattr(self, f'child_order_{level}'), getattr(self, f'group_pointer_{level}'))
        probabilities = [value.exp() for value in logits]
        confidence = []
        for size, prob in zip(self.vocab_sizes, probabilities):
            certainty = (torch.ones_like(prob[:, 0]) if size == 1 else
                         1 + (prob * prob.clamp_min(1e-8).log()).sum(-1) / math.log(size))
            confidence.append(certainty.clamp(0, 1))
        return dict(logits=logits, probabilities=probabilities,
                    confidence=torch.stack(confidence, 1), proposal_logits=[])


class MatchedAblationRetriever(ResidueReadoutRetriever):
    """Isolated controls; construct the full base first for paired seeds.

    Figure 1 score controls retain the shared predictor and both EC objectives.
    Their single remaining component has unit weight and the original learned
    temperature. No unused score-mixture parameter is optimized.
    """
    def __init__(self, *, matched_ablation, **kwargs):
        if matched_ablation not in ARMS:
            raise ValueError('Unknown registered matched ablation')
        if any(kwargs.get(k, 'none') != 'none' for k in
               ('stereo_residual', 'atomic_readout', 'reaction_context', 'candidate_background')):
            raise ValueError('Matched controls exclude other experimental branches')
        for key in ('stereo_residual', 'atomic_readout', 'reaction_context', 'candidate_background'):
            kwargs.pop(key, None)
        if kwargs.get('ec_score_depth') != 4 or kwargs.get('protein_readout') != 'residue_attention':
            raise ValueError('Start from the adopted four-level residue-readout model')
        super().__init__(**kwargs)
        self.matched_ablation = matched_ablation
        if matched_ablation == 'no_ec_branch':
            self.ec_router = NoECRouter()
            del self.ec_weight_logit
            self.ec_score_depth = 0
            self.index_dimension = self.embedding_dim
        elif matched_ablation == 'no_parent_conditioning':
            self.ec_router = DirectECRouter(self.ec_router, self.embedding_dim, kwargs.get('dropout', .1))
        elif matched_ablation == 'mean_residue_pooling':
            del self.residue_weight
        elif matched_ablation in ('no_latent_similarity', 'no_ec_agreement'):
            del self.ec_weight_logit
            self.index_dimension = (sum(self.ec_router.vocab_sizes[:self.ec_score_depth])
                                    if matched_ablation == 'no_latent_similarity'
                                    else self.embedding_dim)

    def ec_weight(self):
        if self.matched_ablation in ('no_ec_branch', 'no_ec_agreement'):
            return self.logit_scale.new_zeros((), dtype=torch.float32)
        if self.matched_ablation == 'no_latent_similarity':
            return self.logit_scale.new_ones((), dtype=torch.float32)
        return super().ec_weight()

    def _pack(self, out):
        if self.matched_ablation == 'no_ec_branch':
            out['latent_embedding'] = out['embedding']  # [B,256]; already unit-normalized.
            out['ec_sqrt_probabilities'] = []
            return out
        packed = super()._pack(out)
        if self.matched_ablation == 'no_latent_similarity':
            # [B,sum_l V_l]: mean of four EC affinities, no latent score block.
            packed['embedding'] = packed['embedding'][:, self.embedding_dim:]
        elif self.matched_ablation == 'no_ec_agreement':
            # [B,D]: EC predictions remain supervised but never enter retrieval.
            packed['embedding'] = packed['latent_embedding']
        return packed

    def encode_protein(self, features):
        if self.matched_ablation != 'mean_residue_pooling':
            return super().encode_protein(features)
        if set(features) != {'mean', 'values', 'offsets_cpu'}:
            raise ValueError('Only the same cached sequence features are accepted')
        # Keep the exact cached global mean, including its original FP16 rounding.
        # Residue tokens are not consulted, and no readout parameter remains.
        return FineECFunctionalRetriever.encode_protein(self, features['mean'])

    def explain_scores(self, reaction, protein):
        if self.matched_ablation == 'no_ec_branch':
            scale = self.logit_scale.float().clamp(max=math.log(100.)).exp()
            score = scale * (reaction['embedding'] @ protein['embedding'].T)
            return dict(latent=score, ec_levels=[], total=score)
        return super().explain_scores(reaction, protein)


def ec_objectives(model, reaction, protein, qt, pt, mask, *, alignment_weight):
    """Unchanged supervised losses, with explicit bypass for absent objectives."""
    zero = reaction['embedding'].new_zeros((), dtype=torch.float32)
    if getattr(model, 'matched_ablation', None) == 'no_ec_branch':
        return zero, zero, zero
    ec = .5 * (hierarchical_multi_positive_loss(reaction['logits'], qt)
               + hierarchical_multi_positive_loss(protein['logits'], pt))
    consistency = .5 * (hierarchy_consistency_loss(reaction['logits'], model.ec_router.parent_maps())
                        + hierarchy_consistency_loss(protein['logits'], model.ec_router.parent_maps()))
    if alignment_weight:
        edges = mask.nonzero()
        alignment = cross_modal_ec_alignment_loss(reaction['logits'],
            [v.index_select(0, edges[:, 1]) for v in protein['logits']], edges[:, 0])
    else:
        alignment = zero
    return ec, consistency, alignment
