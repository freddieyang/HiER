"""One jointly trained retrieval space with internally predicted EC pooling.

Uni-Mol/ESM features may be cached, but nothing in this module detaches input
features. Frozen-backbone training is the first experiment, not a claim that
the pretrained encoders have already been fine-tuned. No teacher, reference
query bank, ground-truth EC, candidate ID or post-hoc density term is needed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from .retrieval_scoring import unit_latent

from .function_transfer import FunctionTransferRetriever


class ReactionViewDropout(nn.Module):
    """Drop a whole normalized reaction view per TRAIN row, never both.

    This is subset training, not inverted element dropout: the retained view
    is not rescaled. Inference always uses both views. No learned parameters
    or extra retrieval spaces are introduced.
    """

    def __init__(self, morgan_probability: float = 0., unimol_probability: float = 0.):
        super().__init__()
        if not (0 <= morgan_probability <= 1 and 0 <= unimol_probability <= 1
                and morgan_probability + unimol_probability <= 1):
            raise ValueError("View-drop probabilities must be nonnegative and sum to at most one")
        self.morgan_probability = float(morgan_probability)
        self.unimol_probability = float(unimol_probability)

    def forward(self, morgan: torch.Tensor, unimol: torch.Tensor):
        if not self.training or self.morgan_probability + self.unimol_probability == 0:
            return morgan, unimol
        if morgan.ndim != 2 or unimol.ndim != 2 or morgan.shape[0] != unimol.shape[0]:
            raise ValueError("Reaction views must have matching batch rows")
        draw = torch.rand((morgan.shape[0], 1), device=morgan.device)
        keep_morgan = draw >= self.morgan_probability
        keep_unimol = (draw < self.morgan_probability) | (draw >= self.morgan_probability + self.unimol_probability)
        return morgan * keep_morgan, unimol * keep_unimol


def molecule_moments(features: torch.Tensor, mask: torch.Tensor,
                     weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable masked mean/std, preserving component multiplicity.

    Padding is removed before arithmetic (including NaN padding). The epsilon
    in sqrt makes single-component/constant sets have finite backward passes.
    """
    if features.ndim != 3 or mask.shape != features.shape[:2] or weights.shape != mask.shape:
        raise ValueError("Expected molecule features [B,M,F] and mask/weights [B,M]")
    mask = mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every reaction must contain at least one molecule")
    valid_weights = weights.float().masked_fill(~mask, 0.)
    if not bool(torch.isfinite(valid_weights).all()) or bool((valid_weights[mask] <= 0).any()):
        raise ValueError("Valid components require positive finite weights")
    clean = features.float().masked_fill(~mask.unsqueeze(-1), 0.)
    if not bool(torch.isfinite(clean).all()):
        raise ValueError("Valid molecule features must be finite")
    normalized = valid_weights / valid_weights.sum(dim=1, keepdim=True)
    mean = (clean * normalized.unsqueeze(-1)).sum(dim=1)  # [B,F]
    centered = (clean - mean.unsqueeze(1)).masked_fill(~mask.unsqueeze(-1), 0.)
    variance = (centered.square() * normalized.unsqueeze(-1)).sum(dim=1)
    std = (variance.clamp_min(0.) + 1e-8).sqrt()
    return torch.cat([mean, std], dim=-1), normalized  # [B,2F], [B,M]


class FunctionalSetRetriever(FunctionTransferRetriever):
    """Single dual encoder: functional context changes molecule aggregation.

    ``moments`` is the single-model statistics control. ``attention`` keeps
    the same EC supervision and late fusion but removes EC from the pooling
    query. ``ec_attention`` changes only that query. All three modes preserve
    the global Morgan/Uni-Mol state, and share the protein encoder and losses.
    """

    def __init__(
        self, molecule_input_dim: int, global_input_dim: int,
        protein_input_dim: int, ec_vocab_sizes: Sequence[int],
        child_to_parent: Sequence[torch.Tensor], projection_hidden_dim: int = 512,
        embedding_dim: int = 256, dropout: float = .1, temperature: float = .07,
        pooling: str = "ec_attention", ec_conditioning: bool = True,
        ec_router_normalization: str = "parent_prior",
        drop_morgan_probability: float = 0., drop_unimol_probability: float = 0.,
    ) -> None:
        if pooling not in {"moments", "attention", "ec_attention"}:
            raise ValueError("Unknown molecule pooling policy")
        if min(molecule_input_dim, global_input_dim, protein_input_dim, embedding_dim) < 1:
            raise ValueError("Feature dimensions must be positive")
        super().__init__(
            reaction_input_dim=global_input_dim + 2 * molecule_input_dim,
            protein_input_dim=protein_input_dim, ec_vocab_sizes=ec_vocab_sizes,
            child_to_parent=child_to_parent, projection_hidden_dim=projection_hidden_dim,
            embedding_dim=embedding_dim, dropout=dropout, temperature=temperature,
            ec_conditioning=ec_conditioning, ec_router_normalization=ec_router_normalization,
        )
        self.pooling = pooling
        self.molecule_input_dim = int(molecule_input_dim)
        self.global_input_dim = int(global_input_dim)
        self.global_norm = nn.LayerNorm(global_input_dim)
        self.moments_norm = nn.LayerNorm(2 * molecule_input_dim)
        self.molecule_projection = nn.Sequential(
            nn.LayerNorm(molecule_input_dim), nn.Linear(molecule_input_dim, embedding_dim),
            nn.GELU(), nn.LayerNorm(embedding_dim),
        )
        self.pool_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pool_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pool_value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pool_output = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.readout_norm = nn.LayerNorm(embedding_dim)
        if pooling != "moments" and drop_morgan_probability + drop_unimol_probability > 0:
            raise ValueError("Whole-view dropout requires moments pooling; token attention would bypass the mask")
        self.view_dropout = ReactionViewDropout(drop_morgan_probability, drop_unimol_probability)

    def _apply_reaction_context(self, state, molecules, mask, prior):
        """Default exact bypass; experimental subclasses must be explicit."""
        return state

    def _reaction_feature_keys(self):
        return {"molecules", "mask", "weights", "global_features"}

    def _refine_molecule_readout(self, features, molecules, mask, global_state, molecule_state):
        """Exact default bypass after view masking and before the sole MLP pass."""
        return molecule_state, {}

    def encode_reaction(self, reaction_features: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        expected = self._reaction_feature_keys()
        if set(reaction_features) != expected:
            raise ValueError("Reaction input contains missing or unexpected fields")
        molecules = reaction_features["molecules"]  # [B,M,F]
        mask = reaction_features["mask"].bool()  # [B,M]
        global_features = reaction_features["global_features"].float()  # [B,G]
        if molecules.shape[-1] != self.molecule_input_dim:
            raise ValueError("Molecule feature dimension mismatch")
        if global_features.shape != (molecules.shape[0], self.global_input_dim):
            raise ValueError("Global reaction feature dimension mismatch")
        if not bool(torch.isfinite(global_features).all()):
            raise ValueError("Global reaction features must be finite")
        moments, prior = molecule_moments(molecules, mask, reaction_features["weights"])
        global_state, molecule_state = self.view_dropout(self.global_norm(global_features), self.moments_norm(moments))
        molecule_state, readout = self._refine_molecule_readout(
            reaction_features, molecules, mask, global_state, molecule_state)
        combined = torch.cat([global_state, molecule_state], dim=-1)
        state = self.reaction_encoder(combined)  # [B,D]
        state = self._apply_reaction_context(state, molecules, mask, prior)
        route = self.ec_router(state)  # Soft EC is predicted before pooling.
        attention = prior
        if self.pooling != "moments":
            tokens = self.molecule_projection(molecules.float().masked_fill(~mask.unsqueeze(-1), 0.))
            context = route["context"] if self.pooling == "ec_attention" and self.ec_conditioning else torch.zeros_like(state)
            confidence = .1 + .9 * route["confidence"].mean(dim=1, keepdim=True)
            query = self.pool_query(state + confidence * context)  # [B,D]
            keys = self.pool_key(tokens)  # [B,M,D]
            logits = (keys.float() * query.float().unsqueeze(1)).sum(dim=-1) / self.embedding_dim ** .5
            logits = logits + torch.log(prior.clamp_min(1e-8))
            attention = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)  # [B,M]
            pooled = (attention.unsqueeze(-1) * self.pool_value(tokens).float()).sum(dim=1)
            state = self.readout_norm(state + self.pool_output(pooled))  # [B,D]
        conditioned = self.reaction_fusion(state, route["context"], route["confidence"]) if self.ec_conditioning else state
        return {"embedding": unit_latent(conditioned), "state": state,
                "molecule_attention": attention, **route, **readout}

    def forward(self, reaction_features: Mapping[str, torch.Tensor], protein_features: torch.Tensor):
        reaction = self.encode_reaction(reaction_features)
        protein = self.encode_protein(protein_features)
        return self.score_embeddings(reaction["embedding"], protein["embedding"]), reaction, protein
