"""Hierarchy-informed dual-encoder for EC-to-enzyme retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalECEncoder(nn.Module):
    """Encode the four EC components as a causal ontology path."""

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(vocab_sizes) != 4:
            raise ValueError("EC paths require four per-level vocabularies")
        self.component_embeddings = nn.ModuleList(
            nn.Embedding(int(size), hidden_dim, padding_idx=0) for size in vocab_sizes
        )
        self.level_embedding = nn.Embedding(4, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.level_gate = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 4:
            raise ValueError("tokens must have shape [batch, 4]")
        if valid_mask.shape != tokens.shape:
            raise ValueError("valid_mask must match tokens")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every EC query needs at least one known prefix level")

        level_ids = torch.arange(4, device=tokens.device).unsqueeze(0)
        states = torch.stack(
            [embedding(tokens[:, level]) for level, embedding in enumerate(self.component_embeddings)],
            dim=1,
        )
        states = states + self.level_embedding(level_ids)
        causal_mask = torch.triu(
            torch.ones(4, 4, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        states = self.path_encoder(
            states,
            mask=causal_mask,
            src_key_padding_mask=~valid_mask.bool(),
        )
        gate_logits = self.level_gate(states).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~valid_mask.bool(), -torch.inf)
        weights = torch.softmax(gate_logits, dim=1)
        pooled = torch.sum(states * weights.unsqueeze(-1), dim=1)
        return F.normalize(self.output(pooled), dim=-1)


class ProteinEncoder(nn.Module):
    """Project cached protein-language-model vectors into retrieval space."""

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
        return F.normalize(self.network(protein_features.float()), dim=-1)


class PairwiseReranker(nn.Module):
    """Conditionally rescore a small Top-K set after dual-encoder retrieval."""

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4 * embedding_dim),
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 3:
            raise ValueError("Expected queries [B,D] and candidates [B,K,D]")
        if query_embeddings.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("Query and candidate batch dimensions differ")
        query = query_embeddings.unsqueeze(1).expand_as(candidate_embeddings)
        pair_features = torch.cat(
            [
                query,
                candidate_embeddings,
                query * candidate_embeddings,
                torch.abs(query - candidate_embeddings),
            ],
            dim=-1,
        )
        return self.network(pair_features).squeeze(-1)


class HiDER(nn.Module):
    """Hierarchy-informed Dual-stage Enzyme Retriever.

    Stage 1 is an indexable dual encoder. Stage 2 is an optional Top-K pairwise
    reranker and never touches the full enzyme library.
    """

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        protein_input_dim: int = 1280,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.ec_encoder = HierarchicalECEncoder(
            vocab_sizes=vocab_sizes,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.protein_encoder = ProteinEncoder(
            input_dim=protein_input_dim,
            hidden_dim=2 * hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.reranker = PairwiseReranker(embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    def encode_ec(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.ec_encoder(tokens, valid_mask)

    def encode_protein(self, protein_features: torch.Tensor) -> torch.Tensor:
        return self.protein_encoder(protein_features)

    def forward(
        self,
        ec_tokens: torch.Tensor,
        ec_valid_mask: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.encode_ec(ec_tokens, ec_valid_mask)  # [Bq, D]
        proteins = self.encode_protein(protein_features)  # [Bp, D]
        scores = queries @ proteins.T / self.temperature  # [Bq, Bp]
        return scores, queries, proteins

    def rerank_topk(
        self,
        query_embeddings: torch.Tensor,
        protein_embeddings: torch.Tensor,
        first_stage_scores: torch.Tensor,
        top_k: int,
        residual_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_weight < 0.0:
            raise ValueError("residual_weight must be non-negative")
        top_k = min(int(top_k), protein_embeddings.shape[0])
        base_scores, indices = torch.topk(first_stage_scores, k=top_k, dim=1)
        candidates = protein_embeddings[indices]  # [Bq, K, D]
        residual_scores = self.reranker(query_embeddings, candidates)
        bounded_residual = residual_weight * torch.tanh(residual_scores)
        return indices, base_scores + bounded_residual
