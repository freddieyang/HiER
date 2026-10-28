"""Structured, query-only reaction-to-enzyme dual encoder.

Inference accepts a reaction graph batch and a pre-indexable enzyme token batch.
Ground-truth EC labels are never forward inputs.  A shared, variable-depth EC
router predicts a soft path internally from each modality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as grouped_softmax
from torch_scatter import scatter_mean, scatter_sum, segment_csr

from .reaction_ec_bridge import ReactionEncoder
from .structured_data import ReactionGraphBatch


class EdgeGatedMessageBlock(nn.Module):
    """Residual message passing with optional edge attributes."""

    def __init__(self, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = (
            nn.Linear(edge_dim, hidden_dim, bias=False) if edge_dim > 0 else None
        )
        gate_input = 3 * hidden_dim if edge_dim > 0 else 2 * hidden_dim
        self.gate = nn.Sequential(nn.Linear(gate_input, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.update = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Invalid node or edge tensor shape")
        if edge_index.shape[1] == 0:
            aggregate = torch.zeros_like(x)
        else:
            source, target = edge_index
            source_state = x.index_select(0, source)
            target_state = x.index_select(0, target)
            message = self.message(source_state)
            if self.edge_projection is not None:
                if edge_attr is None or edge_attr.shape[0] != edge_index.shape[1]:
                    raise ValueError("Edge attributes are required and must align with edges")
                edge_state = self.edge_projection(edge_attr)
                message = message + edge_state
                gate_input = torch.cat([source_state, target_state, edge_state], dim=-1)
            else:
                gate_input = torch.cat([source_state, target_state], dim=-1)
            gated_message = torch.sigmoid(self.gate(gate_input)) * message
            aggregate = scatter_mean(
                gated_message, target, dim=0, dim_size=x.shape[0]
            )
        delta = self.update(torch.cat([x, aggregate], dim=-1))
        return self.output_norm(x + delta)


class AttentiveGroupPool(nn.Module):
    """Attention pooling over nodes assigned to contiguous group indices."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self, x: torch.Tensor, group: torch.Tensor, group_count: int
    ) -> torch.Tensor:
        if x.shape[0] != group.shape[0] or group_count <= 0:
            raise ValueError("Pooling group assignments are invalid")
        weights = grouped_softmax(self.score(x).squeeze(-1), group, num_nodes=group_count)
        return scatter_sum(weights.unsqueeze(-1) * x, group, dim=0, dim_size=group_count)


class AtomMotifReactionEncoder(nn.Module):
    """Encode molecule graphs and then the unordered reaction molecule multiset."""

    def __init__(
        self,
        atom_feature_dim: int,
        edge_feature_dim: int,
        motif_feature_dim: int,
        hidden_dim: int = 256,
        atom_layers: int = 3,
        motif_layers: int = 2,
        set_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.atom_input = nn.Sequential(
            nn.LayerNorm(atom_feature_dim), nn.Linear(atom_feature_dim, hidden_dim), nn.GELU()
        )
        self.atom_blocks = nn.ModuleList(
            EdgeGatedMessageBlock(hidden_dim, edge_feature_dim, dropout)
            for _ in range(atom_layers)
        )
        self.motif_input = nn.Sequential(
            nn.LayerNorm(motif_feature_dim), nn.Linear(motif_feature_dim, hidden_dim), nn.GELU()
        )
        self.motif_merge = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.motif_blocks = nn.ModuleList(
            EdgeGatedMessageBlock(hidden_dim, 0, dropout) for _ in range(motif_layers)
        )
        self.atom_pool = AttentiveGroupPool(hidden_dim)
        self.motif_pool = AttentiveGroupPool(hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.molecule_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            set_layer, num_layers=set_layers, enable_nested_tensor=False
        )
        self.pool_seed = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.set_pool = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        nn.init.normal_(self.pool_seed, std=0.02)

    @staticmethod
    def _pad_components(
        molecule_tokens: torch.Tensor,
        component_reaction: torch.Tensor,
        num_reactions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = scatter_sum(
            torch.ones_like(component_reaction),
            component_reaction,
            dim=0,
            dim_size=num_reactions,
        ).long()
        if bool((counts <= 0).any()):
            raise ValueError("Every reaction must contain at least one molecule")
        starts = torch.cumsum(counts, dim=0) - counts
        local_position = torch.arange(
            molecule_tokens.shape[0], device=molecule_tokens.device
        ) - torch.repeat_interleave(starts, counts)
        max_components = int(counts.max())
        padded = molecule_tokens.new_zeros(
            (num_reactions, max_components, molecule_tokens.shape[-1])
        )
        valid = torch.zeros(
            (num_reactions, max_components),
            dtype=torch.bool,
            device=molecule_tokens.device,
        )
        padded[component_reaction, local_position] = molecule_tokens
        valid[component_reaction, local_position] = True
        return padded, valid

    def forward(self, batch: ReactionGraphBatch) -> torch.Tensor:
        atom_state = self.atom_input(batch.atom_x.float())  # [Na, D]
        for block in self.atom_blocks:
            atom_state = block(atom_state, batch.edge_index, batch.edge_attr.float())

        atom_motif = scatter_mean(
            atom_state,
            batch.atom_to_motif,
            dim=0,
            dim_size=batch.motif_x.shape[0],
        )
        motif_raw = self.motif_input(batch.motif_x.float())
        motif_state = self.motif_merge(torch.cat([motif_raw, atom_motif], dim=-1))
        for block in self.motif_blocks:
            motif_state = block(motif_state, batch.motif_edge_index)

        component_count = int(batch.component_reaction.shape[0])
        atom_summary = self.atom_pool(atom_state, batch.atom_component, component_count)
        motif_summary = self.motif_pool(
            motif_state, batch.motif_component, component_count
        )
        gate = self.fusion_gate(torch.cat([atom_summary, motif_summary], dim=-1))
        molecule = gate * atom_summary + (1.0 - gate) * motif_summary
        molecule = molecule + self.molecule_refine(molecule)  # [Nc, D]

        padded, valid = self._pad_components(
            molecule, batch.component_reaction, batch.num_reactions
        )
        encoded = self.set_encoder(padded, src_key_padding_mask=~valid)
        seed = self.pool_seed.expand(batch.num_reactions, -1, -1)
        attended, _ = self.set_pool(
            seed, encoded, encoded, key_padding_mask=~valid, need_weights=False
        )
        mean = (encoded * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return self.output(torch.cat([attended.squeeze(1), mean], dim=-1))  # [B, D]


class LocalTokenProteinEncoder(nn.Module):
    """Pool ordered local ESM-2 tokens with learned catalytic motif queries."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        max_segments: int = 4,
        motif_queries: int = 4,
        layers: int = 1,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.max_segments = int(max_segments)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.segment_position = nn.Parameter(torch.empty(1, max_segments, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.motif_queries = nn.Parameter(torch.empty(1, motif_queries, hidden_dim))
        self.motif_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        nn.init.normal_(self.segment_position, std=0.02)
        nn.init.normal_(self.motif_queries, std=0.02)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("Protein tokens/mask must have shapes [B,S,F] and [B,S]")
        if tokens.shape[1] > self.max_segments:
            raise ValueError("Protein token count exceeds configured max_segments")
        valid_mask = valid_mask.bool()
        if bool((~valid_mask.any(dim=1)).any()):
            raise ValueError("Every protein must contain at least one valid token")
        state = self.input_projection(tokens.float())
        state = state + self.segment_position[:, : tokens.shape[1]]
        state = self.sequence_encoder(state, src_key_padding_mask=~valid_mask)
        queries = self.motif_queries.expand(tokens.shape[0], -1, -1)
        motif_state, _ = self.motif_attention(
            queries, state, state, key_padding_mask=~valid_mask, need_weights=False
        )
        sequence_mean = (state * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return self.output(
            torch.cat([sequence_mean, motif_state.mean(dim=1)], dim=-1)
        )  # [B, D]


def sibling_conditional_log_probabilities(raw_logits, parent_index, parent_count,
                                         sort_index=None, group_pointer=None):
    """Stable log p(child|parent), normalized only over each sibling group."""
    index = parent_index.unsqueeze(0).expand(raw_logits.shape[0], -1)
    values = raw_logits.float()
    if sort_index is None:
        sort_index = torch.argsort(parent_index, stable=True)
    if group_pointer is None:
        counts = torch.bincount(parent_index, minlength=parent_count)
        group_pointer = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    pointer = group_pointer.unsqueeze(0)
    # CSR segmented reductions are deterministic, unlike atomic scatter sums.
    # Cached index/pointer buffers keep group construction out of model forward.
    maxima = segment_csr(values.detach().index_select(1, sort_index), pointer, reduce="max")
    shifted = values - maxima.gather(1, index)
    totals = segment_csr(shifted.index_select(1, sort_index).exp(), pointer, reduce="sum")
    return shifted - totals.gather(1, index).clamp_min(torch.finfo(values.dtype).tiny).log()


class ConditionalECPathRouter(nn.Module):
    """Variable-depth top-down soft EC path router.

    The first level is predicted from the latent state.  Every subsequent child
    distribution is conditioned on both the expected parent embedding and the
    probability of each child's parent.  Thus the heads do not behave as four
    independent classifiers.  Four levels are used for complete EC labels in
    ReactZyme, while the implementation follows the supplied hierarchy depth.
    """

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        child_to_parent: Sequence[torch.Tensor],
        hidden_dim: int = 256,
        dropout: float = 0.1,
        normalization: str = "parent_prior",
    ) -> None:
        super().__init__()
        if normalization not in {"parent_prior", "sibling_conditional"}:
            raise ValueError("Unknown EC normalization policy")
        self.normalization = normalization
        if not vocab_sizes or any(int(size) <= 0 for size in vocab_sizes):
            raise ValueError("vocab_sizes must contain at least one positive value")
        if len(child_to_parent) != len(vocab_sizes) - 1:
            raise ValueError("One parent map is required between each adjacent EC level")
        self.vocab_sizes = tuple(map(int, vocab_sizes))
        self.state_norm = nn.LayerNorm(hidden_dim)
        heads: list[nn.Module] = [nn.Linear(hidden_dim, self.vocab_sizes[0])]
        for size in self.vocab_sizes[1:]:
            heads.append(
                nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, size),
                )
            )
        self.heads = nn.ModuleList(heads)
        self.prefix_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_dim) for size in self.vocab_sizes
        )
        self.level_weight_logits = nn.Parameter(torch.zeros(len(self.vocab_sizes)))
        self.context_norm = nn.LayerNorm(hidden_dim)
        for level, mapping in enumerate(child_to_parent, start=1):
            mapping = torch.as_tensor(mapping, dtype=torch.long)
            if mapping.numel() != self.vocab_sizes[level]:
                raise ValueError(f"Parent map at level {level + 1} has wrong size")
            if mapping.numel() and (
                int(mapping.min()) < 0 or int(mapping.max()) >= self.vocab_sizes[level - 1]
            ):
                raise ValueError(f"Parent map at level {level + 1} is out of range")
            if normalization == "sibling_conditional" and (
                torch.bincount(mapping, minlength=self.vocab_sizes[level-1]) == 0
            ).any():
                raise ValueError("Marginal-consistent routing requires a child for every parent")
            self.register_buffer(f"parent_index_{level}", mapping, persistent=True)
            if normalization == "sibling_conditional":
                order = torch.argsort(mapping, stable=True)
                counts = torch.bincount(mapping, minlength=self.vocab_sizes[level-1])
                pointer = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
                # Derived from existing parent maps: no checkpoint parameters
                # or persistent state are added to this controlled variant.
                self.register_buffer(f"child_sort_index_{level}", order, persistent=False)
                self.register_buffer(f"child_group_pointer_{level}", pointer, persistent=False)

    @property
    def depth(self) -> int:
        return len(self.vocab_sizes)

    def parent_maps(self) -> list[torch.Tensor]:
        return [getattr(self, f"parent_index_{level}") for level in range(1, self.depth)]

    def forward(self, state: torch.Tensor) -> dict[str, Any]:
        if state.ndim != 2:
            raise ValueError("EC router state must have shape [B,D]")
        state = self.state_norm(state)
        logits: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        contexts: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for level, (head, embedding, vocab_size) in enumerate(
            zip(self.heads, self.prefix_embeddings, self.vocab_sizes)
        ):
            if level == 0:
                level_logits = head(state)
            else:
                expected_parent = probabilities[-1] @ self.prefix_embeddings[
                    level - 1
                ].weight.float()
                raw_logits = head(torch.cat([state, expected_parent], dim=-1))
                parent_map = getattr(self, f"parent_index_{level}")
                if self.normalization == "sibling_conditional":
                    conditional = sibling_conditional_log_probabilities(
                        raw_logits, parent_map, self.vocab_sizes[level-1],
                        getattr(self, f"child_sort_index_{level}"),
                        getattr(self, f"child_group_pointer_{level}"))
                    # log_softmax keeps rare parent log probabilities finite,
                    # including cases where materialized probabilities underflow.
                    log_parent = torch.log_softmax(logits[-1].float(), dim=-1)
                    level_logits = conditional + log_parent.index_select(1, parent_map)
                else:
                    parent_prior = probabilities[-1].index_select(1, parent_map)
                    level_logits = raw_logits.float() + torch.log(parent_prior.clamp_min(1e-8))
            probability = torch.softmax(level_logits.float(), dim=-1)
            context = probability @ embedding.weight.float()
            if vocab_size == 1:
                confidence = torch.ones(
                    state.shape[0], device=state.device, dtype=probability.dtype
                )
            else:
                entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
                confidence = 1.0 - entropy / math.log(vocab_size)
            logits.append(level_logits)
            probabilities.append(probability)
            contexts.append(context)
            confidences.append(confidence.clamp(0.0, 1.0))

        confidence_matrix = torch.stack(confidences, dim=1)  # [B,L]
        adaptive = torch.softmax(
            self.level_weight_logits.float().unsqueeze(0)
            + torch.log((0.1 + 0.9 * confidence_matrix).clamp_min(1e-8)),
            dim=1,
        )
        context_stack = torch.stack(contexts, dim=1)  # [B,L,D]
        route_context = self.context_norm(
            torch.sum(adaptive.unsqueeze(-1) * context_stack, dim=1)
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "confidence": confidence_matrix,
            "level_weights": adaptive,
            "context": route_context,
        }


class RouteFusion(nn.Module):
    """Confidence-gated residual fusion of a base state and soft EC context."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + 1),
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, state: torch.Tensor, context: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        fusion = torch.cat(
            [state, context, state * context, torch.abs(state - context)], dim=-1
        )
        delta = self.delta(fusion)
        mean_confidence = confidence.mean(dim=1, keepdim=True)
        gate = self.gate(torch.cat([state, context, mean_confidence], dim=-1))
        return self.output_norm(state + gate * delta)


class StructuredReactionEnzymeRetriever(nn.Module):
    """ANN-compatible structured reaction/enzyme retrieval model."""

    def __init__(
        self,
        atom_feature_dim: int,
        edge_feature_dim: int,
        motif_feature_dim: int,
        ec_vocab_sizes: Sequence[int],
        child_to_parent: Sequence[torch.Tensor],
        protein_input_dim: int = 1280,
        protein_segments: int = 4,
        hidden_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.reaction_encoder = AtomMotifReactionEncoder(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            motif_feature_dim=motif_feature_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.protein_encoder = LocalTokenProteinEncoder(
            input_dim=protein_input_dim,
            hidden_dim=hidden_dim,
            max_segments=protein_segments,
            heads=heads,
            dropout=dropout,
        )
        self.reaction_route_adapter = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.protein_route_adapter = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.ec_router = ConditionalECPathRouter(
            vocab_sizes=ec_vocab_sizes,
            child_to_parent=child_to_parent,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.reaction_fusion = RouteFusion(hidden_dim, dropout)
        self.protein_fusion = RouteFusion(hidden_dim, dropout)
        self.route_mix_logit = nn.Parameter(torch.tensor(-1.3862944))  # sigmoid = 0.2
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature), dtype=torch.float32)
        )

    def _compose_embedding(
        self, state: torch.Tensor, route: dict[str, Any], fusion: RouteFusion
    ) -> torch.Tensor:
        fused = F.normalize(
            fusion(state, route["context"], route["confidence"]), dim=-1
        )
        route_embedding = F.normalize(route["context"], dim=-1)
        mix = torch.sigmoid(self.route_mix_logit).clamp(1e-4, 1.0 - 1e-4)
        # Concatenation preserves a single dot-product retrieval index while
        # expressing a learned convex combination of base and EC-route cosine.
        return torch.cat(
            [torch.sqrt(1.0 - mix) * fused, torch.sqrt(mix) * route_embedding],
            dim=-1,
        )

    def encode_reaction(self, batch: ReactionGraphBatch) -> dict[str, Any]:
        state = self.reaction_encoder(batch)  # [Br,D]
        route = self.ec_router(self.reaction_route_adapter(state))
        embedding = self._compose_embedding(state, route, self.reaction_fusion)
        return {"embedding": embedding, "state": state, **route}

    def encode_protein(
        self, protein_tokens: torch.Tensor, protein_mask: torch.Tensor
    ) -> dict[str, Any]:
        state = self.protein_encoder(protein_tokens, protein_mask)  # [Bp,D]
        route = self.ec_router(self.protein_route_adapter(state))
        embedding = self._compose_embedding(state, route, self.protein_fusion)
        return {"embedding": embedding, "state": state, **route}

    def score_embeddings(
        self, query_embeddings: torch.Tensor, protein_embeddings: torch.Tensor
    ) -> torch.Tensor:
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * (query_embeddings @ protein_embeddings.T)

    def forward(
        self,
        reaction_batch: ReactionGraphBatch,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        reaction = self.encode_reaction(reaction_batch)
        protein = self.encode_protein(protein_tokens, protein_mask)
        scores = self.score_embeddings(reaction["embedding"], protein["embedding"])
        return scores, reaction, protein


class GlobalAnchorProteinEncoder(nn.Module):
    """Project a frozen whole-sequence protein mean into retrieval space.

    The layer layout intentionally matches the clean ReactZyme baseline so its
    weights can be loaded without approximation.
    """

    def __init__(
        self,
        input_dim: int = 1280,
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
        )

    def forward(self, protein_features: torch.Tensor) -> torch.Tensor:
        if protein_features.ndim != 2:
            raise ValueError("protein_features must have shape [batch, feature_dim]")
        return self.network(protein_features.float())  # [B, D]


class GlobalAnchoredStructuredRetriever(nn.Module):
    """Frozen global retrieval anchors with bounded trainable residual branches.

    Morgan reaction fingerprints and whole-sequence ESM means retain the exact
    geometry of a previously validated clean baseline. Atom/motif graphs,
    ordered local protein tokens, and an internally predicted EC path can only
    enter through zero-initialized bounded residual scales. Consequently the
    initial embedding is exactly the normalized global anchor, while training
    can add task-specific local and hierarchical corrections. Ground-truth EC
    labels are auxiliary targets only and never forward inputs.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        edge_feature_dim: int,
        motif_feature_dim: int,
        ec_vocab_sizes: Sequence[int],
        child_to_parent: Sequence[torch.Tensor],
        reaction_global_dim: int = 2048,
        protein_global_dim: int = 1280,
        protein_token_dim: int = 1280,
        protein_segments: int = 4,
        anchor_hidden_dim: int = 512,
        hidden_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07,
        residual_scale_limit: float = 0.5,
        num_query_prototypes: int = 1,
        prototype_scale_limit: float = 0.25,
        prototype_aggregation_temperature: float = 1.0,
        freeze_global_anchors: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if residual_scale_limit <= 0:
            raise ValueError("residual_scale_limit must be positive")
        if num_query_prototypes <= 0:
            raise ValueError("num_query_prototypes must be positive")
        if prototype_scale_limit <= 0:
            raise ValueError("prototype_scale_limit must be positive")
        if prototype_aggregation_temperature <= 0:
            raise ValueError("prototype_aggregation_temperature must be positive")
        self.hidden_dim = int(hidden_dim)
        self.residual_scale_limit = float(residual_scale_limit)
        self.num_query_prototypes = int(num_query_prototypes)
        self.prototype_scale_limit = float(prototype_scale_limit)
        self.prototype_aggregation_temperature = float(
            prototype_aggregation_temperature
        )
        self.global_anchors_frozen = bool(freeze_global_anchors)

        # These two modules exactly match the clean Morgan/ESM baseline.
        self.reaction_global_encoder = ReactionEncoder(
            input_dim=reaction_global_dim,
            hidden_dim=anchor_hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.protein_global_encoder = GlobalAnchorProteinEncoder(
            input_dim=protein_global_dim,
            hidden_dim=anchor_hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        self.reaction_structured_encoder = AtomMotifReactionEncoder(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            motif_feature_dim=motif_feature_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.protein_local_encoder = LocalTokenProteinEncoder(
            input_dim=protein_token_dim,
            hidden_dim=hidden_dim,
            max_segments=protein_segments,
            heads=heads,
            dropout=dropout,
        )

        self.reaction_local_residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.protein_local_residual = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.reaction_route_adapter = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.protein_route_adapter = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.ec_router = ConditionalECPathRouter(
            vocab_sizes=ec_vocab_sizes,
            child_to_parent=child_to_parent,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.reaction_route_residual = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.protein_route_residual = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        if self.num_query_prototypes > 1:
            # A reaction can have valid enzymes in several remote sequence
            # families.  Generate all alternative query directions in one
            # vectorized pass while retaining a single pre-indexed vector per
            # candidate enzyme.
            self.reaction_prototype_residual = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    2 * hidden_dim,
                    (self.num_query_prototypes - 1) * hidden_dim,
                ),
            )
            self.reaction_prototype_scale_logit = nn.Parameter(torch.zeros(()))

        # ReZero-style scalars make the initial retrieval geometry identical to
        # the clean baseline. tanh bounds each branch's maximum perturbation.
        self.reaction_local_scale_logit = nn.Parameter(torch.zeros(()))
        self.reaction_route_scale_logit = nn.Parameter(torch.zeros(()))
        self.protein_local_scale_logit = nn.Parameter(torch.zeros(()))
        self.protein_route_scale_logit = nn.Parameter(torch.zeros(()))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature), dtype=torch.float32)
        )
        if self.global_anchors_frozen:
            self.freeze_global_anchor_parameters()

    def _bounded_scale(self, value: torch.Tensor) -> torch.Tensor:
        return self.residual_scale_limit * torch.tanh(value.float())

    def residual_scales(self) -> dict[str, torch.Tensor]:
        scales = {
            "reaction_local": self._bounded_scale(self.reaction_local_scale_logit),
            "reaction_route": self._bounded_scale(self.reaction_route_scale_logit),
            "protein_local": self._bounded_scale(self.protein_local_scale_logit),
            "protein_route": self._bounded_scale(self.protein_route_scale_logit),
        }
        if self.num_query_prototypes > 1:
            scales["reaction_prototype"] = self.prototype_scale_limit * torch.tanh(
                self.reaction_prototype_scale_logit.float()
            )
        return scales

    def freeze_global_anchor_parameters(self) -> None:
        self.global_anchors_frozen = True
        for module in (self.reaction_global_encoder, self.protein_global_encoder):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True) -> "GlobalAnchoredStructuredRetriever":
        super().train(mode)
        if self.global_anchors_frozen:
            # Frozen dropout must remain disabled so the anchor is stable and
            # validation exactly reproduces the source baseline.
            self.reaction_global_encoder.eval()
            self.protein_global_encoder.eval()
        return self

    def load_global_anchor_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load only the clean baseline encoders and temperature, strictly."""

        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("Anchor checkpoint has no model_state dictionary")
        reaction_prefix = "reaction_encoder."
        protein_prefix = "protein_encoder."
        reaction_state = {
            key[len(reaction_prefix) :]: value
            for key, value in state.items()
            if key.startswith(reaction_prefix)
        }
        protein_state = {
            f"network.{key[len(protein_prefix) :]}": value
            for key, value in state.items()
            if key.startswith(protein_prefix)
        }
        if not reaction_state or not protein_state:
            raise ValueError("Anchor checkpoint lacks reaction or protein encoder weights")
        self.reaction_global_encoder.load_state_dict(reaction_state, strict=True)
        self.protein_global_encoder.load_state_dict(protein_state, strict=True)
        if "logit_scale" in state:
            self.logit_scale.data.copy_(state["logit_scale"].float())
        if self.global_anchors_frozen:
            self.freeze_global_anchor_parameters()

    def _compose_embedding(
        self,
        anchor_state: torch.Tensor,
        local_state: torch.Tensor,
        route: dict[str, Any],
        local_residual: nn.Module,
        route_residual: nn.Module,
        local_scale_logit: torch.Tensor,
        route_scale_logit: torch.Tensor,
    ) -> torch.Tensor:
        local_delta = local_residual(local_state)
        route_delta = route_residual(
            torch.cat([route["router_state"], route["context"]], dim=-1)
        )
        state = (
            anchor_state
            + self._bounded_scale(local_scale_logit) * local_delta
            + self._bounded_scale(route_scale_logit) * route_delta
        )
        return F.normalize(state, dim=-1)

    def encode_reaction(
        self,
        batch: ReactionGraphBatch,
        reaction_global_features: torch.Tensor,
    ) -> dict[str, Any]:
        # Preserve the exact FP32 geometry of the clean source checkpoint even
        # when the trainable residual branches use BF16/FP16 autocast.
        with torch.amp.autocast(
            device_type=reaction_global_features.device.type, enabled=False
        ):
            anchor_state = self.reaction_global_encoder(
                reaction_global_features.float()
            )  # [Br,D]
        local_state = self.reaction_structured_encoder(batch)  # [Br,D]
        router_state = self.reaction_route_adapter(
            torch.cat([anchor_state, local_state], dim=-1)
        )
        route = self.ec_router(router_state)
        route["router_state"] = router_state
        embedding = self._compose_embedding(
            anchor_state,
            local_state,
            route,
            self.reaction_local_residual,
            self.reaction_route_residual,
            self.reaction_local_scale_logit,
            self.reaction_route_scale_logit,
        )
        if self.num_query_prototypes == 1:
            prototypes = embedding.unsqueeze(1)  # [Br,1,D]
        else:
            prototype_context = torch.cat(
                [anchor_state, local_state, route["context"]], dim=-1
            )
            prototype_delta = self.reaction_prototype_residual(
                prototype_context
            ).reshape(
                embedding.shape[0], self.num_query_prototypes - 1, self.hidden_dim
            )  # [Br,K-1,D]
            prototype_scale = self.prototype_scale_limit * torch.tanh(
                self.reaction_prototype_scale_logit.float()
            )
            alternatives = F.normalize(
                embedding.unsqueeze(1) + prototype_scale * prototype_delta,
                dim=-1,
            )
            prototypes = torch.cat(
                [embedding.unsqueeze(1), alternatives], dim=1
            )  # [Br,K,D]
        return {
            "embedding": embedding,
            "prototypes": prototypes,
            "anchor_state": anchor_state,
            "local_state": local_state,
            **route,
        }

    def encode_protein(
        self,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        protein_global_features: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.amp.autocast(
            device_type=protein_global_features.device.type, enabled=False
        ):
            anchor_state = self.protein_global_encoder(
                protein_global_features.float()
            )  # [Bp,D]
        local_state = self.protein_local_encoder(protein_tokens, protein_mask)  # [Bp,D]
        router_state = self.protein_route_adapter(
            torch.cat([anchor_state, local_state], dim=-1)
        )
        route = self.ec_router(router_state)
        route["router_state"] = router_state
        embedding = self._compose_embedding(
            anchor_state,
            local_state,
            route,
            self.protein_local_residual,
            self.protein_route_residual,
            self.protein_local_scale_logit,
            self.protein_route_scale_logit,
        )
        return {
            "embedding": embedding,
            "anchor_state": anchor_state,
            "local_state": local_state,
            **route,
        }

    def score_embeddings(
        self, query_embeddings: torch.Tensor, protein_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if protein_embeddings.ndim != 2:
            raise ValueError("protein_embeddings must have shape [candidates, dim]")
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        if query_embeddings.ndim == 2:
            return scale * (query_embeddings @ protein_embeddings.T)
        if query_embeddings.ndim != 3:
            raise ValueError(
                "query_embeddings must have shape [queries, dim] or "
                "[queries, prototypes, dim]"
            )
        if query_embeddings.shape[-1] != protein_embeddings.shape[-1]:
            raise ValueError("Query and protein embedding dimensions differ")
        prototype_scores = scale * torch.einsum(
            "qkd,cd->qkc", query_embeddings, protein_embeddings
        )  # [Bq,K,Bc]
        aggregation_temperature = self.prototype_aggregation_temperature
        return aggregation_temperature * (
            torch.logsumexp(
                prototype_scores / aggregation_temperature, dim=1
            )
            - math.log(query_embeddings.shape[1])
        )  # [Bq,Bc]

    def forward(
        self,
        reaction_batch: ReactionGraphBatch,
        reaction_global_features: torch.Tensor,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        protein_global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        reaction = self.encode_reaction(reaction_batch, reaction_global_features)
        protein = self.encode_protein(
            protein_tokens, protein_mask, protein_global_features
        )
        scores = self.score_embeddings(reaction["prototypes"], protein["embedding"])
        return scores, reaction, protein
