"""Reaction-to-enzyme retrieval with a latent hierarchical EC bridge.

The public inference path accepts only reaction features. EC annotations are
optional training targets for the internal hierarchy heads and are never
arguments to ``forward`` or ``encode_reaction``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactionEncoder(nn.Module):
    """Project a fixed reaction fingerprint into the retrieval space."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, reaction_features: torch.Tensor) -> torch.Tensor:
        if reaction_features.ndim != 2:
            raise ValueError("reaction_features must have shape [batch, feature_dim]")
        return self.network(reaction_features.float())  # [B, D]


class LatentECHierarchy(nn.Module):
    """Predict soft EC prefixes from a reaction representation.

    Each level has its own prefix vocabulary. The expected prefix embeddings
    form an EC context, while normalized entropy measures how much the model
    should trust that context for a particular reaction.
    """

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        hidden_dim: int = 256,
        context_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(vocab_sizes) != 4 or any(int(size) <= 0 for size in vocab_sizes):
            raise ValueError("vocab_sizes must contain four positive values")
        self.vocab_sizes = tuple(map(int, vocab_sizes))
        self.classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, size) for size in self.vocab_sizes
        )
        self.prefix_embeddings = nn.ModuleList(
            nn.Embedding(size, context_dim) for size in self.vocab_sizes
        )
        self.level_weight_logits = nn.Parameter(torch.zeros(4))

    def context_from_logits(
        self, logits: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert four prefix distributions into one uncertainty-aware context."""

        if len(logits) != 4:
            raise ValueError("Expected logits for four EC levels")
        contexts: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for level_logits, embedding, vocab_size in zip(
            logits, self.prefix_embeddings, self.vocab_sizes
        ):
            if level_logits.ndim != 2 or level_logits.shape[1] != vocab_size:
                raise ValueError("EC logits do not match the configured vocabulary")
            probabilities = torch.softmax(level_logits.float(), dim=-1)
            level_context = probabilities @ embedding.weight.float()  # [B, D]
            if vocab_size == 1:
                confidence = torch.ones(
                    level_logits.shape[0],
                    device=level_logits.device,
                    dtype=probabilities.dtype,
                )
            else:
                entropy = -torch.sum(
                    probabilities * torch.log(probabilities.clamp_min(1e-8)), dim=-1
                )
                confidence = 1.0 - entropy / math.log(vocab_size)
            contexts.append(level_context)
            confidences.append(confidence.clamp(0.0, 1.0))

        confidence_matrix = torch.stack(confidences, dim=1)  # [B, 4]
        prior_weights = torch.softmax(self.level_weight_logits.float(), dim=0)
        adaptive_weights = prior_weights.unsqueeze(0) * (
            0.25 + 0.75 * confidence_matrix
        )
        adaptive_weights = adaptive_weights / adaptive_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        context_stack = torch.stack(contexts, dim=1)  # [B, 4, D]
        context = torch.sum(
            context_stack * adaptive_weights.unsqueeze(-1), dim=1
        )  # [B, D]
        return context, confidence_matrix

    def forward(
        self, reaction_states: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        if reaction_states.ndim != 2:
            raise ValueError("reaction_states must have shape [batch, hidden_dim]")
        logits = [
            classifier(reaction_states) for classifier in self.classifiers
        ]
        context, confidence = self.context_from_logits(logits)
        return logits, context, confidence


class ReactionECBridgeRetriever(nn.Module):
    """Retrieve enzymes from reactions through an internal soft EC bridge.

    The reaction residual keeps the query useful when the EC predictor is
    uncertain or encounters an unseen class. The EC branch can refine the
    query but cannot access a ground-truth EC value during inference.
    """

    def __init__(
        self,
        ec_vocab_sizes: Sequence[int],
        reaction_input_dim: int = 2048,
        protein_input_dim: int = 1280,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 0.07,
        use_ec_bridge: bool = True,
        use_protein_ec_bridge: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.use_ec_bridge = bool(use_ec_bridge)
        self.use_protein_ec_bridge = bool(use_protein_ec_bridge)
        self.reaction_encoder = ReactionEncoder(
            input_dim=reaction_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.ec_bridge = LatentECHierarchy(
            vocab_sizes=ec_vocab_sizes,
            hidden_dim=embedding_dim,
            context_dim=embedding_dim,
        )
        fusion_input_dim = 4 * embedding_dim
        self.bridge_delta = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.bridge_gate = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim + 1),
            nn.Linear(2 * embedding_dim + 1, embedding_dim),
            nn.Sigmoid(),
        )
        self.protein_encoder = nn.Sequential(
            nn.LayerNorm(protein_input_dim),
            nn.Linear(protein_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        if self.use_protein_ec_bridge:
            self.protein_ec_classifiers: nn.ModuleList | None = nn.ModuleList(
                nn.Linear(embedding_dim, int(size)) for size in ec_vocab_sizes
            )
            self.protein_bridge_delta: nn.Module | None = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.protein_bridge_gate: nn.Module | None = nn.Sequential(
                nn.LayerNorm(2 * embedding_dim + 1),
                nn.Linear(2 * embedding_dim + 1, embedding_dim),
                nn.Sigmoid(),
            )
        else:
            self.protein_ec_classifiers = None
            self.protein_bridge_delta = None
            self.protein_bridge_gate = None
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature), dtype=torch.float32)
        )

    def encode_reaction(
        self, reaction_features: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        reaction_state = self.reaction_encoder(reaction_features)  # [B, D]
        ec_logits, ec_context, confidence = self.ec_bridge(reaction_state)
        if self.use_ec_bridge:
            fusion = torch.cat(
                [
                    reaction_state,
                    ec_context,
                    reaction_state * ec_context,
                    torch.abs(reaction_state - ec_context),
                ],
                dim=-1,
            )  # [B, 4D]
            delta = self.bridge_delta(fusion)  # [B, D]
            mean_confidence = confidence.mean(dim=1, keepdim=True)  # [B, 1]
            gate = self.bridge_gate(
                torch.cat([reaction_state, ec_context, mean_confidence], dim=-1)
            )  # [B, D]
            reaction_state = reaction_state + gate * delta
        query = F.normalize(reaction_state, dim=-1)  # [B, D]
        return query, ec_logits, confidence

    def encode_protein_with_aux(
        self, protein_features: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
        if protein_features.ndim != 2:
            raise ValueError("protein_features must have shape [batch, feature_dim]")
        protein_state = self.protein_encoder(protein_features.float())  # [B, D]
        if not self.use_protein_ec_bridge:
            return F.normalize(protein_state, dim=-1), None, None
        if (
            self.protein_ec_classifiers is None
            or self.protein_bridge_delta is None
            or self.protein_bridge_gate is None
        ):
            raise RuntimeError("Protein EC bridge modules are not initialized")
        ec_logits = [
            classifier(protein_state) for classifier in self.protein_ec_classifiers
        ]
        ec_context, confidence = self.ec_bridge.context_from_logits(ec_logits)
        fusion = torch.cat(
            [
                protein_state,
                ec_context,
                protein_state * ec_context,
                torch.abs(protein_state - ec_context),
            ],
            dim=-1,
        )  # [B, 4D]
        delta = self.protein_bridge_delta(fusion)  # [B, D]
        gate = self.protein_bridge_gate(
            torch.cat(
                [protein_state, ec_context, confidence.mean(dim=1, keepdim=True)],
                dim=-1,
            )
        )  # [B, D]
        return F.normalize(protein_state + gate * delta, dim=-1), ec_logits, confidence

    def encode_protein(self, protein_features: torch.Tensor) -> torch.Tensor:
        embeddings, _, _ = self.encode_protein_with_aux(protein_features)
        return embeddings

    def score(
        self, query_embeddings: torch.Tensor, protein_embeddings: torch.Tensor
    ) -> torch.Tensor:
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * (query_embeddings @ protein_embeddings.T)

    def forward(
        self,
        reaction_features: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        dict[str, object],
    ]:
        queries, ec_logits, confidence = self.encode_reaction(reaction_features)
        proteins, protein_ec_logits, protein_confidence = self.encode_protein_with_aux(
            protein_features
        )
        scores = self.score(queries, proteins)  # [Bq, Bp]
        auxiliary = {
            "reaction_confidence": confidence,
            "protein_ec_logits": protein_ec_logits,
            "protein_confidence": protein_confidence,
        }
        return scores, queries, proteins, ec_logits, auxiliary


def hierarchical_multi_positive_loss(
    logits_by_level: Sequence[torch.Tensor],
    target_masks_by_level: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Average negative log probability mass over valid EC prefix targets."""

    if not logits_by_level or len(logits_by_level) != len(target_masks_by_level):
        raise ValueError("EC logits and targets must contain the same non-zero level count")
    losses: list[torch.Tensor] = []
    for logits, targets in zip(logits_by_level, target_masks_by_level):
        if logits.shape != targets.shape:
            raise ValueError("EC logits and target masks must have matching shapes")
        targets = targets.bool()
        valid = targets.any(dim=1)
        if not bool(valid.any()):
            continue
        selected = logits[valid].float()
        positives = selected.masked_fill(~targets[valid], -torch.inf)
        losses.append(
            (torch.logsumexp(selected, dim=1) - torch.logsumexp(positives, dim=1)).mean()
        )
    if not losses:
        return sum(value.sum() * 0.0 for value in logits_by_level)
    return torch.stack(losses).mean()


def hierarchy_consistency_loss(
    logits_by_level: Sequence[torch.Tensor],
    child_to_parent: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Match each parent distribution to its child-marginal distribution."""

    if not logits_by_level or len(child_to_parent) != len(logits_by_level) - 1:
        raise ValueError("Expected one child-to-parent map between every adjacent level")
    losses: list[torch.Tensor] = []
    for level in range(len(logits_by_level) - 1):
        parent = torch.softmax(logits_by_level[level].float(), dim=-1)
        child = torch.softmax(logits_by_level[level + 1].float(), dim=-1)
        mapping = child_to_parent[level].to(device=child.device, dtype=torch.long)
        if mapping.numel() != child.shape[1]:
            raise ValueError("child_to_parent size does not match child vocabulary")
        marginal = torch.zeros_like(parent)
        marginal.scatter_add_(
            1, mapping.unsqueeze(0).expand(child.shape[0], -1), child
        )
        midpoint = 0.5 * (parent + marginal)
        parent_log = torch.log(parent.clamp_min(1e-8))
        marginal_log = torch.log(marginal.clamp_min(1e-8))
        midpoint_log = torch.log(midpoint.clamp_min(1e-8))
        js = 0.5 * torch.sum(parent * (parent_log - midpoint_log), dim=1)
        js = js + 0.5 * torch.sum(
            marginal * (marginal_log - midpoint_log), dim=1
        )
        losses.append(js.mean())
    if not losses:
        return logits_by_level[0].sum() * 0.0
    return torch.stack(losses).mean()


def cross_modal_ec_alignment_loss(
    reaction_logits_by_level: Sequence[torch.Tensor],
    protein_logits_by_level: Sequence[torch.Tensor],
    protein_to_reaction_row: torch.Tensor,
) -> torch.Tensor:
    """Align reaction and paired-protein EC distributions without label input."""

    if (
        not reaction_logits_by_level
        or len(reaction_logits_by_level) != len(protein_logits_by_level)
    ):
        raise ValueError("Reaction and protein EC logits must have the same levels")
    if protein_to_reaction_row.ndim != 1:
        raise ValueError("protein_to_reaction_row must have shape [proteins]")
    losses: list[torch.Tensor] = []
    for reaction_logits, protein_logits in zip(
        reaction_logits_by_level, protein_logits_by_level
    ):
        if reaction_logits.shape[1] != protein_logits.shape[1]:
            raise ValueError("Reaction and protein EC vocabularies differ")
        reaction_prob = torch.softmax(reaction_logits.float(), dim=-1).index_select(
            0, protein_to_reaction_row
        )
        protein_prob = torch.softmax(protein_logits.float(), dim=-1)
        midpoint = 0.5 * (reaction_prob + protein_prob)
        reaction_log = torch.log(reaction_prob.clamp_min(1e-8))
        protein_log = torch.log(protein_prob.clamp_min(1e-8))
        midpoint_log = torch.log(midpoint.clamp_min(1e-8))
        js = 0.5 * torch.sum(
            reaction_prob * (reaction_log - midpoint_log), dim=1
        )
        js = js + 0.5 * torch.sum(
            protein_prob * (protein_log - midpoint_log), dim=1
        )
        losses.append(js.mean())
    return torch.stack(losses).mean()
