"""The existing HiER retrieval equations, without new parameters or operations.

Separate modality projections learn compatibility through paired retrieval
training. Normalization below fixes vector length; it is not an additional
learned alignment layer. EC arguments are predictions, never input labels.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def unit_latent(state: torch.Tensor) -> torch.Tensor:
    """h [B,D] -> z [B,D], using the original FP32 L2 normalization."""
    return F.normalize(state.float(), dim=-1)


def positive_score_scale(logit_scale: torch.Tensor) -> torch.Tensor:
    """Shared positive softmax scale, with the original upper bound of 100."""
    return logit_scale.float().clamp(max=math.log(100.)).exp()


def ec_root_blocks(probabilities, depth: int):
    """Predicted p_l [B,V_l] -> normalized sqrt(p_l); preserve epsilon."""
    return [F.normalize(p.clamp_min(1e-8).sqrt(), dim=-1)
            for p in probabilities[:depth]]


def joint_index_vector(latent, roots, alpha, depth):
    """Pack z and EC blocks into [B,D + sum_l V_l] for one inner product.

    Keep the original arithmetic order and square-root mixture weights.
    This function adds no normalization, detach, projection or temperature.
    """
    return torch.cat([(1-alpha).sqrt()*latent,
                      (alpha/depth).sqrt()*torch.cat(roots, -1)], -1)


def score_components(reaction, protein, alpha, scale, depth):
    """Weighted latent / per-level EC contributions, each [B_r,B_e].

    The unweighted latent similarity is z_r @ z_e.T (cosine similarity).
    The returned terms include the mixture weight and softmax scale, exactly
    as the original explain_scores API. They are score attributions, not
    mechanistic explanations. Packed and decomposed scores can differ by
    FP32 accumulation roundoff because their matrix products have different
    reduction lengths.
    """
    latent = scale*(1-alpha)*(reaction['latent_embedding'] @ protein['latent_embedding'].T)
    levels = [scale*alpha/depth*(q @ p.T)
              for q, p in zip(reaction['ec_sqrt_probabilities'],
                              protein['ec_sqrt_probabilities'])]
    return dict(latent=latent, ec_levels=levels,
                total=latent+torch.stack(levels).sum(0))
