"""Contextualize molecules inside each official undirected reaction set.

Inspired by set self-attention (Lee et al., ICML 2019), not a transition-state
graph or an enzyme-reaction cross-attention reranker. The original Morgan and
Uni-Mol moment representation is retained. A zero-initialized residual learns
information about component combinations before the shared predicted-EC head.

Source: https://proceedings.mlr.press/v97/lee19d.html
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from .residue_readout import ResidueReadoutRetriever
from .functional_set_retriever import molecule_moments


class MoleculeSetContext(nn.Module):
    """[B,M,F] -> [B,D], permutation invariant and confined to each reaction.

    Two attention heads by default, one contextual block, no positional or
    side/direction labels. TRAIN-fitted IDF appears as a key prior and the final
    pooling measure. Original component multiplicity is retained.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, heads=2):
        super().__init__()
        if min(input_dim, output_dim, hidden_dim, heads) < 1 or hidden_dim % heads:
            raise ValueError('Positive dimensions and hidden_dim divisible by heads required')
        self.input_dim, self.hidden_dim, self.heads = input_dim, hidden_dim, heads
        self.tokens = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                    nn.GELU(), nn.LayerNorm(hidden_dim))
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, molecules, mask, prior):
        if (molecules.ndim != 3 or molecules.shape[-1] != self.input_dim
                or mask.shape != molecules.shape[:2] or prior.shape != mask.shape):
            raise ValueError('Aligned molecule [B,M,F], mask/prior [B,M] required')
        if mask.dtype != torch.bool or not bool(mask.any(1).all()):
            raise ValueError('Each reaction needs a nonempty boolean component mask')
        clean = molecules.float().masked_fill(~mask.unsqueeze(-1), 0.)
        weights = prior.float().masked_fill(~mask, 0.)
        if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(weights).all()) or bool((weights[mask] <= 0).any()):
            raise ValueError('Finite valid molecules and positive finite component priors required')
        weights = weights / weights.sum(1, keepdim=True)
        x = self.tokens(clean)  # [B,M,H]
        batch, count, _ = x.shape
        def heads(layer):
            return layer(x).view(batch, count, self.heads, self.hidden_dim // self.heads).transpose(1, 2)
        q, k, v = heads(self.query), heads(self.key), heads(self.value)
        # [B,A,M,M], quadratic only in components of one reaction (max80 in
        # the audited cache), never in enzymes or retrieval candidates.
        with torch.autocast(device_type=molecules.device.type, enabled=False):
            logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.hidden_dim // self.heads)
            logits = logits + weights.clamp_min(1e-12).log()[:, None, None, :]
            attention = logits.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
            context = (attention @ v.float()).transpose(1, 2).reshape(batch, count, self.hidden_dim)
        contextual = self.context_norm(x.float() + self.mix(context))
        pooled = (contextual.float().masked_fill(~mask.unsqueeze(-1), 0.) * weights.unsqueeze(-1)).sum(1)
        return self.output(pooled)  # [B,D]; exactly zero at initialization


class MoleculeContextRetriever(ResidueReadoutRetriever):
    """Add component context before latent normalization and EC prediction.

    Existing protein encoding, losses, EC score weights and index width stay
    unchanged. Extra random initialization is isolated from the baseline's
    training RNG; common weights and initial predictions match the control.
    """
    def __init__(self, *, reaction_context='molecule_set', context_hidden_dim=128,
                 context_heads=2, context_gradient_mode='joint', **kwargs):
        if reaction_context != 'molecule_set' or kwargs.get('pooling') != 'moments':
            raise ValueError('Context residual is a separate moments-model experiment')
        if kwargs.get('scoring', 'explicit') != 'explicit' or kwargs.get('drop_unimol_probability', 0.) != 0:
            raise ValueError('Context experiment uses explicit scoring with an intact Uni-Mol view')
        if context_gradient_mode not in ('joint', 'isolated'):
            raise ValueError('Unknown context gradient mode')
        super().__init__(**kwargs)
        self.reaction_context = reaction_context
        self.context_gradient_mode = context_gradient_mode
        with torch.random.fork_rng(devices=[]):
            self.molecule_context = MoleculeSetContext(self.molecule_input_dim, self.embedding_dim,
                                                       context_hidden_dim, context_heads)

    def _apply_reaction_context(self, state, molecules, mask, prior):
        if self.training and self.context_gradient_mode == 'isolated':
            # The reference route receives precisely its original objective.
            # A separate contextual objective below trains only the residual.
            return state
        delta = self.molecule_context(molecules, mask, prior)
        # Preserve the current projection dtype, including BF16 training.
        return state + delta.to(dtype=state.dtype)

    def detached_context_reaction(self, base_reaction, features):
        """Refine this same TRAIN state, with gradients only into the context.

        Shared EC parameters are used at their current values with detached
        weights, not under no_grad: derivatives through the EC function to the
        residual input are retained. No extra reaction/protein projection runs.
        EC-head dropout must be isolated by the calling loss helper. Inference
        uses the ordinary contextual route above, without this second EC pass.
        """
        if not self.training or self.context_gradient_mode != 'isolated':
            raise ValueError('Detached refinement is for isolated TRAIN only')
        if set(features) != {'molecules', 'mask', 'weights', 'global_features'}:
            raise ValueError('Reaction-only cached features required')
        _, prior = molecule_moments(features['molecules'], features['mask'], features['weights'])
        delta = self.molecule_context(features['molecules'], features['mask'], prior)
        base_state = base_reaction['state'].detach()
        state = base_state + delta.to(dtype=base_state.dtype)  # [B,D]
        params = {k: p.detach() for k, p in self.ec_router.named_parameters()}
        buffers = {k: b.detach() for k, b in self.ec_router.named_buffers()}
        route = functional_call(self.ec_router, (params, buffers), (state,), strict=True)
        latent = F.normalize(state.float(), dim=-1)
        alpha = self.ec_weight().detach()
        roots = [F.normalize(p.clamp_min(1e-8).sqrt(), dim=-1)
                 for p in route['probabilities'][:self.ec_score_depth]]
        packed = torch.cat([(1-alpha).sqrt()*latent,
                            (alpha/self.ec_score_depth).sqrt()*torch.cat(roots, -1)], -1)
        return dict(state=state, embedding=packed, latent_embedding=latent,
                    ec_sqrt_probabilities=roots, **route)

    def isolated_parameter_groups(self):
        """Clip separately so a new residual cannot rescale base gradients."""
        if self.context_gradient_mode != 'isolated':
            raise ValueError('Separate clipping belongs to the isolated experiment')
        common, context = [], []
        for name, parameter in self.named_parameters():
            (context if name.startswith('molecule_context.') else common).append(parameter)
        return common, context
