"""Query-macro retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def retrieval_metrics(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float | int]:
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must be aligned 2-D tensors")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")

    rankings = torch.argsort(scores, dim=1, descending=True, stable=True)
    per_query: list[dict[str, float]] = []
    for row in range(scores.shape[0]):
        positives = positive_mask[row].bool()
        total_positives = int(positives.sum())
        if total_positives == 0:
            continue
        relevance = positives[rankings[row]].to(torch.float64)
        positive_ranks = torch.nonzero(relevance, as_tuple=False).flatten() + 1
        metrics: dict[str, float] = {
            "mrr": 1.0 / float(positive_ranks[0]),
        }
        for k in ks:
            cutoff = min(int(k), relevance.numel())
            rel_k = relevance[:cutoff]
            hits = float(rel_k.sum())
            metrics[f"hit@{k}"] = float(hits > 0)
            metrics[f"recall@{k}"] = hits / total_positives

            cumulative = torch.cumsum(rel_k, dim=0)
            denom = torch.arange(1, cutoff + 1, dtype=torch.float64)
            precision = cumulative / denom
            metrics[f"map@{k}"] = float((precision * rel_k).sum()) / min(
                total_positives, cutoff
            )

            discounts = 1.0 / torch.log2(
                torch.arange(2, cutoff + 2, dtype=torch.float64)
            )
            dcg = float((rel_k * discounts).sum())
            ideal_count = min(total_positives, cutoff)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
        per_query.append(metrics)

    if not per_query:
        raise ValueError("No evaluable query contains an observed positive")
    output: dict[str, float | int] = {"num_queries": len(per_query)}
    for key in per_query[0]:
        output[key] = sum(item[key] for item in per_query) / len(per_query)
    return output
