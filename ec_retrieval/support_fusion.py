"""Label-free, train-support-conditioned two-space retrieval control.

Support banks and scale calibration must contain official TRAIN queries only.
No relevance, EC targets, query IDs, or validation membership enter the gate.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.inference_mode()
def nearest_cosine_distance(queries, bank, batch_size=256, exclude_self=False):
    """Return exact nearest-neighbor cosine distances in query order."""
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1]:
        raise ValueError("Expected compatible [N,D] feature matrices")
    if bank.shape[0] < (2 if exclude_self else 1):
        raise ValueError("Insufficient support-bank rows")
    if exclude_self and (queries.shape != bank.shape or not torch.equal(queries, bank)):
        raise ValueError("Self exclusion requires the same ordered training matrix")
    if batch_size < 1 or not torch.isfinite(queries).all() or not torch.isfinite(bank).all():
        raise ValueError("Invalid batch size or non-finite support features")
    if (queries.norm(dim=-1) <= 0).any() or (bank.norm(dim=-1) <= 0).any():
        raise ValueError("Zero vectors cannot define cosine support")
    distances = []
    # The calibrated scale can be small; AMP-rounded cosine distances can
    # otherwise trigger novelty for identical training/query features.
    with torch.amp.autocast(device_type=queries.device.type, enabled=False):
        bank_norm = F.normalize(bank.float(), dim=-1)
        for start in range(0, len(queries), batch_size):
            q = F.normalize(queries[start:start+batch_size].float(), dim=-1)
            similarity = q @ bank_norm.T
            if exclude_self:
                row = torch.arange(len(q), device=q.device)
                similarity[row, start+row] = -torch.inf
            distances.append((1. - similarity.max(dim=1).values).clamp(0., 2.))
    return torch.cat(distances)


def support_scale(train_leave_self_out_distances, epsilon=1e-6):
    """Median positive leave-self-out distance, with explicit duplicate handling."""
    if epsilon <= 0 or not torch.isfinite(train_leave_self_out_distances).all():
        raise ValueError("Invalid scale calibration")
    positive = train_leave_self_out_distances[train_leave_self_out_distances > epsilon]
    # A constant feature space carries no evidence of locality. Keep an
    # explicit fallback instead of fitting any scale using held-out queries.
    return float(positive.median()) if len(positive) else 1.


def support_conditioned_weight(anchor_distance, transfer_distance, anchor_scale,
                               transfer_scale, base_weight=.25, maximum_weight=.5,
                               transfer_reliability=True, epsilon=1e-6):
    """Prefer the second expert only when the first lacks local train support.

    The reliability variant additionally requires local support in the new
    learned space. A familiar query retains the validated base mixture.
    """
    if not 0 <= base_weight <= maximum_weight <= 1 or min(anchor_scale, transfer_scale, epsilon) <= 0:
        raise ValueError("Invalid mixture bounds or calibration scale")
    if anchor_distance.ndim != 1 or anchor_distance.shape != transfer_distance.shape:
        raise ValueError("Distances must be aligned [B] vectors")
    if not torch.isfinite(anchor_distance).all() or not torch.isfinite(transfer_distance).all():
        raise ValueError("Non-finite support distances")
    novelty = ((anchor_distance - epsilon).clamp_min(0.) / anchor_scale).clamp_max(1.)
    reliability = torch.exp(-transfer_distance.clamp_min(0.) / transfer_scale) if transfer_reliability else 1.
    return base_weight + (maximum_weight-base_weight) * novelty * reliability


def tail_support_calibration(train_leave_self_out_distances, quantile=.95, epsilon=1e-6):
    """Fit a permissive support region using only TRAIN leave-self-out distances.

    Inside the specified training quantile there is no reliability penalty.
    This is an empirical support heuristic, not a conformal coverage guarantee:
    the projection itself was fitted on these training examples.
    """
    d = train_leave_self_out_distances.float()
    if d.ndim != 1 or len(d) < 2 or not torch.isfinite(d).all() or (d < 0).any():
        raise ValueError("Finite nonnegative TRAIN distances [N>=2] required")
    if not .5 < quantile < 1 or epsilon <= 0:
        raise ValueError("Require .5 < quantile < 1 and epsilon > 0")
    threshold = max(float(torch.quantile(d, quantile)), epsilon)
    decay_scale = max(threshold - float(torch.quantile(d, .5)), epsilon)
    return {"quantile": quantile, "threshold": threshold, "decay_scale": decay_scale}


def tail_guarded_weight(anchor_distance, transfer_distance, anchor_scale,
                        transfer_calibration, base_weight=.25, maximum_weight=.5,
                        epsilon=1e-6):
    """Retain novelty routing in supported regions, shrink it outside train support."""
    threshold = float(transfer_calibration["threshold"])
    scale = float(transfer_calibration["decay_scale"])
    if not torch.isfinite(torch.tensor([threshold, scale])).all() or min(threshold, scale) <= 0:
        raise ValueError("Invalid TRAIN tail calibration")
    novelty_weight = support_conditioned_weight(
        anchor_distance, transfer_distance, anchor_scale, scale,
        base_weight, maximum_weight, transfer_reliability=False, epsilon=epsilon)
    reliability = torch.exp(-(transfer_distance-threshold).clamp_min(0.)/scale)
    return base_weight + (novelty_weight-base_weight)*reliability


def query_weight_index_vectors(anchor_queries, anchor_candidates, transfer_queries,
                               transfer_candidates, transfer_weights):
    """Make per-query fusion one inner product, leaving candidate vectors fixed."""
    if any(t.ndim != 2 for t in (anchor_queries, anchor_candidates, transfer_queries, transfer_candidates)):
        raise ValueError("Expected embedding matrices")
    if anchor_queries.shape[0] != transfer_queries.shape[0] or anchor_candidates.shape[0] != transfer_candidates.shape[0]:
        raise ValueError("Expert entity alignment failure")
    if anchor_queries.shape[1] != anchor_candidates.shape[1] or transfer_queries.shape[1] != transfer_candidates.shape[1]:
        raise ValueError("Dimension mismatch within an expert")
    if transfer_weights.shape != (len(anchor_queries),) or not torch.isfinite(transfer_weights).all():
        raise ValueError("Finite weights [B] required")
    if ((transfer_weights < 0) | (transfer_weights > 1)).any():
        raise ValueError("Weights must be probabilities")
    w = transfer_weights.to(anchor_queries.device, dtype=anchor_queries.dtype).unsqueeze(1)
    return (torch.cat([(1-w)*anchor_queries, w*transfer_queries], dim=-1),
            torch.cat([anchor_candidates, transfer_candidates], dim=-1))
