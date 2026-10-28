"""Optional current-model candidate sampling, confined to TRAIN associations.

The bank contains candidate row IDs, not a second encoder or inference support
memory. Unobserved pairs are comparison items, never certified negatives.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time

import numpy as np
import torch

from scripts.train_structured_retriever import sample_query_balanced_batch


@dataclass
class TrainCandidateDomain:
    adjacency: dict[int, tuple[int, ...]]
    universe: np.ndarray

    def __post_init__(self):
        self.universe = np.asarray(self.universe, dtype=np.int64)
        if self.universe.ndim != 1 or not len(self.universe):
            raise ValueError("A nonempty TRAIN candidate universe is required")
        if not np.array_equal(self.universe, np.unique(self.universe)) or self.universe.min() < 0:
            raise ValueError("Candidate rows must be sorted, unique and nonnegative")
        expected = {int(p) for values in self.adjacency.values() for p in values}
        if expected != set(map(int, self.universe)):
            raise ValueError("Universe must equal candidates in TRAIN associations")
        if not self.adjacency or any(not values for values in self.adjacency.values()):
            raise ValueError("Every TRAIN query requires a known positive")
        if any(q < 0 or len(values) != len(set(values)) for q, values in self.adjacency.items()):
            raise ValueError("Invalid query row or duplicated TRAIN association")
        self.query_rows = np.asarray(sorted(self.adjacency), dtype=np.int64)
        self.allowed = np.zeros(int(self.universe.max()) + 1, dtype=bool)
        self.allowed[self.universe] = True

    def check_candidates(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= len(self.allowed) or not self.allowed[rows].all()):
            raise ValueError("A non-TRAIN candidate entered optimization")


@contextmanager
def _evaluation_state(model):
    """Restore every module's training state, including on mining failure."""
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, state in states:
            module.training = state


@torch.inference_mode()
def mine_current_train_candidates(model, query_features, protein_features,
        domain: TrainCandidateDomain, *, bank_size=64, query_batch_size=64,
        protein_batch_size=2048, amp_enabled=False, amp_dtype=torch.float32):
    """Exact TRAIN top-k excluding all known positives for the mining query.

    Encoders are batched. A query chunk scores the TRAIN index with one matrix
    multiply [B,D]@[D,N]; sparse positive edges mask their scores at once.
    Scores/indices are detached. Candidate features are encoded afresh, with
    gradients, when selected for an ordinary training step.
    """
    if min(bank_size, query_batch_size, protein_batch_size) < 1:
        raise ValueError("Bank and batch sizes must be positive")
    if any(len(domain.universe) - len(v) < bank_size for v in domain.adjacency.values()):
        raise ValueError("Insufficient unobserved TRAIN candidates for the bank")
    device = protein_features.device
    began = time.perf_counter()
    global_to_local = {int(p): i for i, p in enumerate(domain.universe)}
    lengths = np.asarray([len(domain.adjacency[int(q)]) for q in domain.query_rows])
    offsets = np.r_[0, lengths.cumsum()]
    edge_columns = torch.tensor([global_to_local[p] for q in domain.query_rows
        for p in domain.adjacency[int(q)]], device=device, dtype=torch.long)
    edge_queries = torch.repeat_interleave(torch.arange(len(lengths), device=device),
        torch.as_tensor(lengths, device=device))
    candidate_rows = torch.as_tensor(domain.universe, device=device)
    banks, similarities = [], []
    with _evaluation_state(model):
        vectors = []
        for start in range(0, len(domain.universe), protein_batch_size):
            rows = domain.universe[start:start + protein_batch_size].tolist()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                vectors.append(model.encode_protein(protein_features[rows])["embedding"].float())
        proteins = torch.cat(vectors)  # [N_train,D]
        if not torch.isfinite(proteins).all():
            raise FloatingPointError("Nonfinite TRAIN candidate index")
        for start in range(0, len(domain.query_rows), query_batch_size):
            end = min(start + query_batch_size, len(domain.query_rows))
            rows = domain.query_rows[start:end].tolist()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                queries = model.encode_reaction(query_features[rows])["embedding"].float()
            # FP32 cosine ranking. A common positive temperature does not alter top-k.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                scores = queries @ proteins.T  # [B,N_train]
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite TRAIN mining score")
            a, b = int(offsets[start]), int(offsets[end])
            scores[edge_queries[a:b] - start, edge_columns[a:b]] = -torch.inf
            values, columns = torch.topk(scores, bank_size, dim=1, sorted=True)
            if not torch.isfinite(values).all():
                raise ValueError("Not enough unmasked candidates")
            banks.append(candidate_rows[columns].cpu())  # [B,K]
            similarities.append(values.cpu())
    row_matrix = torch.cat(banks)
    bank = {int(q): tuple(map(int, row)) for q, row in zip(domain.query_rows, row_matrix.tolist())}
    for q, values in bank.items():
        domain.check_candidates(values)
        if set(values) & set(domain.adjacency[q]):
            raise AssertionError("Known positive found in its own mining bank")
    artifact = dict(query_rows=torch.from_numpy(domain.query_rows.copy()), candidate_rows=row_matrix,
        similarities=torch.cat(similarities), training_splits=["train"], known_positive_overlap=0,
        candidate_count=len(domain.universe), query_count=len(domain.query_rows), bank_size=bank_size,
        model_mode="eval_without_view_masking", source="current_joint_model_before_epoch",
        selection="exact_topk_fp32_cosine; equal-score order follows torch.topk",
        seconds=time.perf_counter()-began, gradients_recorded=False)
    return bank, artifact


def sample_training_candidates(query_rows, domain, positives_per_query, rng, *,
        target_candidate_count, negative_rng, positive_exposure_counts,
        bank=None, hard_rng=None, hard_per_query=4, random_reserve_fraction=.25):
    """Preserve positive coverage and a random share within a fixed pool budget.

    With no bank, this calls the original sampler without changing RNG use.
    With a bank, direct positive draws consume exactly the same query RNG;
    independent RNGs supply mined proposals and random denominator filling.
    """
    if any(int(q) not in domain.adjacency for q in query_rows):
        raise ValueError("Only TRAIN queries may be sampled")
    if not 0 <= random_reserve_fraction <= 1 or hard_per_query < 0:
        raise ValueError("Invalid mixing settings")
    if target_candidate_count > len(domain.universe) or target_candidate_count < 1:
        raise ValueError("Candidate budget outside the TRAIN universe")
    if bank is None:
        candidates, mask = sample_query_balanced_batch(query_rows, domain.adjacency,
            positives_per_query, rng, candidate_universe=domain.universe,
            target_candidate_count=target_candidate_count, negative_rng=negative_rng,
            positive_exposure_counts=positive_exposure_counts)
        domain.check_candidates(candidates)
        return candidates, mask, dict(mined_added=0, random_reserved=0, bank_active=0)
    if hard_rng is None or set(bank) != set(domain.adjacency):
        raise ValueError("A complete TRAIN bank and separate mining RNG are required")
    direct, _ = sample_query_balanced_batch(query_rows, domain.adjacency,
        positives_per_query, rng, positive_exposure_counts=positive_exposure_counts)
    if len(direct) > target_candidate_count:
        raise ValueError("Direct positives exceed the fixed candidate budget")
    selected = set(direct)
    available_slots = target_candidate_count - len(selected)
    reserved = min(available_slots, int(np.ceil(random_reserve_fraction * target_candidate_count)))
    proposed = set()
    for q in query_rows:
        values = np.asarray(bank[int(q)], dtype=np.int64)
        domain.check_candidates(values)
        if set(values) & set(domain.adjacency[int(q)]):
            raise ValueError("Mining bank contains a query's known positive")
        k = min(hard_per_query, len(values))
        proposed.update(map(int, hard_rng.choice(values, size=k, replace=False)))
    proposed.difference_update(selected)
    cap = available_slots - reserved
    if len(proposed) > cap:
        proposed = set(map(int, hard_rng.choice(np.asarray(sorted(proposed)), size=cap, replace=False)))
    selected.update(proposed)
    available = domain.universe[~np.isin(domain.universe, np.asarray(sorted(selected)), assume_unique=True)]
    n_random = target_candidate_count - len(selected)
    if n_random:
        selected.update(map(int, negative_rng.choice(available, size=n_random, replace=False)))
    candidates = sorted(selected)
    domain.check_candidates(candidates)
    mask = torch.from_numpy(np.stack([np.isin(candidates, domain.adjacency[int(q)]) for q in query_rows]))
    if len(candidates) != target_candidate_count or not mask.any(dim=1).all():
        raise AssertionError("Candidate budget or direct positive supervision lost")
    return candidates, mask, dict(mined_added=len(proposed), random_reserved=reserved,
        random_added=n_random, direct_candidates=len(direct), bank_active=1)
