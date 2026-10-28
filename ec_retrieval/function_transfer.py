"""Independent chemical/function retrieval with an internally predicted EC route.

Frozen Uni-Mol and ESM features are the only inputs. This module has no label,
reaction-ID, protein-ID, split, or ground-truth EC input path.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .reaction_ec_bridge import ReactionEncoder
from .structured_retriever import ConditionalECPathRouter, RouteFusion
from .retrieval_scoring import positive_score_scale, unit_latent


def fixed_weight_index_vectors(
    anchor_queries: torch.Tensor, anchor_candidates: torch.Tensor,
    transfer_queries: torch.Tensor, transfer_candidates: torch.Tensor,
    transfer_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent two cosine scores as a single pre-indexable dot product.

    Inputs are separately unit-normalized vectors with aligned entity order.
    Candidate vectors are independent of the query and of the mixture weight.
    """
    if not 0. <= transfer_weight <= 1.:
        raise ValueError("transfer_weight must lie in [0,1]")
    if any(t.ndim != 2 for t in (anchor_queries, anchor_candidates, transfer_queries, transfer_candidates)):
        raise ValueError("Every embedding matrix must be two-dimensional")
    if anchor_queries.shape[0] != transfer_queries.shape[0] or anchor_candidates.shape[0] != transfer_candidates.shape[0]:
        raise ValueError("The two spaces must have aligned entity rows")
    if anchor_queries.shape[1] != anchor_candidates.shape[1] or transfer_queries.shape[1] != transfer_candidates.shape[1]:
        raise ValueError("Query and candidate dimensions must match within each space")
    query = torch.cat([(1. - transfer_weight) * anchor_queries,
                       transfer_weight * transfer_queries], dim=-1)
    candidate = torch.cat([anchor_candidates, transfer_candidates], dim=-1)
    return query, candidate


class FunctionTransferRetriever(nn.Module):
    """A standalone, trainable retrieval space rather than an anchor residual."""

    def __init__(
        self,
        reaction_input_dim: int,
        protein_input_dim: int,
        ec_vocab_sizes: Sequence[int],
        child_to_parent: Sequence[torch.Tensor],
        projection_hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = .1,
        temperature: float = .07,
        ec_conditioning: bool = True,
        ec_router_normalization: str = "parent_prior",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.ec_conditioning = bool(ec_conditioning)
        self.embedding_dim = int(embedding_dim)
        self.reaction_encoder = ReactionEncoder(
            reaction_input_dim, projection_hidden_dim, embedding_dim, dropout)
        self.protein_encoder = ReactionEncoder(
            protein_input_dim, projection_hidden_dim, embedding_dim, dropout)
        self.ec_router = ConditionalECPathRouter(
            ec_vocab_sizes, child_to_parent, embedding_dim, dropout,
            normalization=ec_router_normalization)
        self.reaction_fusion = RouteFusion(embedding_dim, dropout)
        self.protein_fusion = RouteFusion(embedding_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1. / temperature)))

    def _encode(self, features: torch.Tensor, side: str) -> dict[str, Any]:
        state = getattr(self, side + "_encoder")(features)  # [B,D]
        route = self.ec_router(state)  # probabilities [B,V_l], context [B,D]
        # Both projections receive their own retrieval supervision. There is
        # no zero-initialized anchor multiplier blocking this space's gradients.
        conditioned = (
            getattr(self, side + "_fusion")(state, route["context"], route["confidence"])
            if self.ec_conditioning else state
        )
        embedding = unit_latent(conditioned)  # [B,D]; paired training learns alignment.
        return {"embedding": embedding, "state": state, **route}

    def encode_reaction(self, reaction_features: torch.Tensor) -> dict[str, Any]:
        return self._encode(reaction_features, "reaction")

    def encode_protein(self, protein_features: torch.Tensor) -> dict[str, Any]:
        return self._encode(protein_features, "protein")

    def score_embeddings(self, queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]:
            raise ValueError("Expected aligned [queries,D] and [candidates,D] embeddings")
        scale = positive_score_scale(self.logit_scale)
        return scale * (queries @ candidates.T)  # [Bq,Bp]

    def forward(self, reaction_features: torch.Tensor, protein_features: torch.Tensor):
        reaction = self.encode_reaction(reaction_features)
        protein = self.encode_protein(protein_features)
        return self.score_embeddings(reaction["embedding"], protein["embedding"]), reaction, protein
