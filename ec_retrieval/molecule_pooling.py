"""TRAIN-fitted molecule-frequency weighting of frozen Uni-Mol representations."""
from collections import Counter
import math

import torch


def fit_molecule_document_frequency(components_by_query, train_ids):
    """Count a molecule once per TRAIN reaction, independent of enzyme degree."""
    train_ids = list(train_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("Unique nonempty training query IDs required")
    if any(q not in components_by_query or not components_by_query[q] for q in train_ids):
        raise ValueError("All training reactions must have molecule components")
    counts = Counter()
    for q in train_ids:
        counts.update(set(components_by_query[q]))
    return dict(counts), len(train_ids)


def smooth_molecule_idf(molecules, document_frequency, train_query_count):
    """Smoothed IDF; unseen molecules use df=0 without held-out refitting."""
    if train_query_count < 1 or any(n < 0 or n > train_query_count for n in document_frequency.values()):
        raise ValueError("Invalid TRAIN document frequencies")
    return torch.tensor([
        1. + math.log((1.+train_query_count)/(1.+document_frequency.get(m, 0)))
        for m in molecules], dtype=torch.float32)


def pool_molecule_features(vectors, component_rows, lengths, weights=None):
    """Vectorized ragged weighted mean: [M,D] -> [reactions,D].

    The official component multiset is preserved; repeated components retain
    multiplicity. Run feature-cache construction on CPU for fixed reductions.
    """
    if vectors.ndim != 2 or not torch.isfinite(vectors).all():
        raise ValueError("Finite molecule embeddings [M,D] required")
    lengths = torch.as_tensor(lengths, device=vectors.device, dtype=torch.long)
    rows = torch.as_tensor(component_rows, device=vectors.device, dtype=torch.long)
    if lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any() or int(lengths.sum()) != len(rows):
        raise ValueError("Nonempty reaction lengths must partition component rows")
    if rows.ndim != 1 or (rows < 0).any() or (rows >= len(vectors)).any():
        raise ValueError("Invalid molecule indices")
    if weights is None:
        weights = torch.ones(len(vectors), device=vectors.device)
    weights = weights.to(device=vectors.device, dtype=torch.float32)
    if weights.shape != (len(vectors),) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Positive finite molecule weights [M] required")
    group = torch.repeat_interleave(torch.arange(len(lengths), device=vectors.device), lengths)
    w = weights[rows]
    numerator = torch.zeros(len(lengths), vectors.shape[1], device=vectors.device)
    numerator.index_add_(0, group, vectors[rows].float()*w[:, None])
    denominator = torch.zeros(len(lengths), device=vectors.device).index_add_(0, group, w)
    return numerator/denominator[:, None]


def pool_molecule_moments(vectors, component_rows, lengths, weights=None):
    """Concatenate weighted mean and population std: [M,D] -> [R,2D].

    Two-pass centered variance avoids subtracting nearly equal raw moments.
    This deterministic feature-cache operation is not part of gradient flow.
    Every official component occurrence retains its original multiplicity.
    """
    mean = pool_molecule_features(vectors, component_rows, lengths, weights)
    lengths = torch.as_tensor(lengths, device=vectors.device, dtype=torch.long)
    rows = torch.as_tensor(component_rows, device=vectors.device, dtype=torch.long)
    group = torch.repeat_interleave(torch.arange(len(lengths), device=vectors.device), lengths)
    if weights is None:
        weights = torch.ones(len(vectors), device=vectors.device)
    w = weights.to(device=vectors.device, dtype=torch.float32)[rows]
    centered = vectors[rows].float()-mean[group]  # [total_components,D]
    variance = torch.zeros_like(mean).index_add_(0, group, centered.square()*w[:, None])
    total = torch.zeros(len(lengths), device=vectors.device).index_add_(0, group, w)
    std = (variance/total[:, None]).clamp_min(0.).sqrt()
    return torch.cat([mean, std], dim=-1)
