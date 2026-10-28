"""Direct TRAIN supervision of the existing latent retrieval representation.

This auxiliary objective uses no EC probabilities, alpha, learned temperature,
new parameters or new model forward.  Both directions reuse the primary loss's
candidate identities and complete known-positive masks. Inference is unchanged.
"""
from __future__ import annotations

import math
import torch

from .bidirectional_loss import bidirectional_retrieval_loss
from .reverse_pool_extension import expanded_bidirectional_loss


def add_latent_retrieval_supervision(
    base_loss, reaction, protein, positive_mask, *, weight=0.0,
    temperature=0.07, extra_reaction=None, extended_mask=None,
    reverse_weight=0.5, mode="legacy", all_positive_weight=0.2,
    tail_risk_weight=0.1, tail_fraction=0.2,
):
    """Return total loss and detached diagnostics; weight=0 is an exact bypass.

    Latents are existing unit vectors [Bq,D], [Bp,D] (and optionally [Be,D]).
    The expanded reverse matrix is [Bp,Bq+Be]. A fixed temperature prevents
    the auxiliary objective from updating the primary score's scale or alpha.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Latent auxiliary weight must be finite and nonnegative")
    if weight == 0:
        return base_loss, {}
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Latent auxiliary temperature must be finite and positive")
    q, p = reaction["latent_embedding"], protein["latent_embedding"]
    if q.ndim != 2 or p.ndim != 2 or q.shape[1] != p.shape[1]:
        raise ValueError("Expected aligned latent matrices [queries,D], [proteins,D]")
    scores = (q.float() @ p.float().T) / temperature  # [Bq,Bp]
    settings = dict(reverse_weight=reverse_weight, mode=mode,
                    all_positive_weight=all_positive_weight,
                    tail_risk_weight=tail_risk_weight, tail_fraction=tail_fraction)
    if extra_reaction is None:
        if extended_mask is not None:
            raise ValueError("Extended positive mask requires extra reaction encodings")
        auxiliary, diagnostics = bidirectional_retrieval_loss(scores, positive_mask, **settings)
    else:
        if extended_mask is None:
            raise ValueError("Expanded latent retrieval requires the complete TRAIN mask")
        extra = extra_reaction["latent_embedding"]
        extra_scores = (extra.float() @ p.float().T) / temperature  # [Be,Bp]
        auxiliary, diagnostics = expanded_bidirectional_loss(
            scores, positive_mask, extra_scores, extended_mask, **settings)
    logged = {"latent_auxiliary": auxiliary.detach()}
    logged.update({"latent_aux_" + k: v.detach() for k, v in diagnostics.items()})
    return base_loss + weight * auxiliary, logged
