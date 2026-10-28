"""Validation-only checkpoint selection for the deployed, support-gated fusion.

The frozen anchor and query support geometry never enter the optimizer. New
expert support is recomputed on official TRAIN reactions at each checkpoint.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from ec_retrieval.support_fusion import (
    nearest_cosine_distance, support_scale, tail_support_calibration,
    tail_guarded_weight, query_weight_index_vectors,
)
from scripts.train_structured_retriever import amp_policy, rank_metrics


def head_guards_pass(metrics, reference, hit1_tolerance=.003, mrr_tolerance=.002):
    if min(hit1_tolerance, mrr_tolerance) < 0:
        raise ValueError("Head-metric tolerances must be nonnegative")
    keys = ("official_reactzyme_mean_rank", "hit@1", "official_reactzyme_mrr")
    if not all(torch.isfinite(torch.tensor(metrics[k])) for k in keys):
        return False
    return bool(metrics["hit@1"] >= reference["hit@1"]-hit1_tolerance
                and metrics["official_reactzyme_mrr"] >= reference["official_reactzyme_mrr"]-mrr_tolerance)


class FrozenFusionSelection:
    def __init__(self, anchor_queries, anchor_candidates, anchor_distances, anchor_scale,
                 train_rows, train_ids, query_ids, candidate_ids, reference_metrics,
                 device, amp_dtype="bfloat16", query_batch_size=128, tail_quantile=.95,
                 hit1_tolerance=.003, mrr_tolerance=.002):
        self.anchor_queries = anchor_queries
        self.anchor_candidates = anchor_candidates
        self.anchor_distances = anchor_distances
        self.anchor_scale = anchor_scale
        self.train_rows, self.train_ids = train_rows, train_ids
        self.query_ids, self.candidate_ids = query_ids, candidate_ids
        self.reference_metrics = reference_metrics
        self.device, self.amp_dtype = device, amp_dtype
        self.query_batch_size, self.tail_quantile = query_batch_size, tail_quantile
        self.hit1_tolerance, self.mrr_tolerance = hit1_tolerance, mrr_tolerance

    @classmethod
    def from_artifacts(cls, args, reference, train_ids, query_row, query_ids, candidate_ids, device):
        plan = json.loads((args.fusion_selection_reference_study / "training_plan.json").read_text())
        job = next(j for j in plan["jobs"] if j["seed"] == args.seed)
        if plan["training_splits"] != ["train"] or Path(job["reference_run"]).resolve() != args.reference_run.resolve():
            raise ValueError("Fusion anchor is not the matching train-only reference")
        cache = torch.load(args.fusion_selection_reference_study / "baseline_validation" /
                          args.reference_run.name / "validation_embeddings.pt", map_location="cpu", weights_only=False)
        if cache["split"] != "val" or cache["query_ids"] != query_ids or cache["candidate_ids"] != candidate_ids:
            raise ValueError("Frozen anchor must match the full official validation library")
        support = args.fusion_selection_support_study / f"seed{args.seed}"
        bank = torch.load(support / "support_bank.pt", map_location="cpu", weights_only=False)
        ordered_train_ids = sorted(train_ids)
        if bank["training_splits"] != ["train"] or bank["train_ids"] != ordered_train_ids:
            raise ValueError("Support bank differs from official TRAIN reaction IDs")
        distances = pd.read_parquet(support / "query_weights_before_relevance.parquet")
        if distances.query_id.duplicated().any() or set(distances.query_id) != set(query_ids):
            raise ValueError("Support-query geometry does not cover exactly validation")
        d = torch.tensor(distances.set_index("query_id").loc[query_ids, "anchor_distance"].to_numpy())
        scale = support_scale(bank["anchor_leave_self_out_distance"])
        if abs(scale-bank["anchor_scale"]) > 1e-8:
            raise ValueError("TRAIN support scale mismatch")
        guard = json.loads((args.fusion_selection_guard_study / "combined_validation" /
                            f"seed{args.seed}" / "metrics.json").read_text())
        return cls(cache["query_embeddings"], cache["candidate_embeddings"], d, scale,
            [query_row[q] for q in ordered_train_ids], ordered_train_ids, query_ids, candidate_ids,
            guard, device, args.amp_dtype, args.eval_query_batch_size,
            args.fusion_selection_tail_quantile, args.fusion_selection_hit1_tolerance,
            args.fusion_selection_mrr_tolerance)

    @torch.inference_mode()
    def evaluate(self, model, reaction_features, queries, candidates, relevance):
        if model.training:
            raise ValueError("Checkpoint validation requires model.eval(); no stochastic dropout")
        enabled, dtype = amp_policy(self.device, self.amp_dtype)
        chunks = []
        for start in range(0, len(self.train_rows), self.query_batch_size):
            with torch.amp.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled):
                result = model.encode_reaction(reaction_features[self.train_rows[start:start+self.query_batch_size]])
            chunks.append(result["embedding"].float())
        bank = torch.cat(chunks)
        loo = nearest_cosine_distance(bank, bank, exclude_self=True)
        calibration = tail_support_calibration(loo, self.tail_quantile)
        distance = nearest_cosine_distance(queries.to(self.device), bank).cpu()
        weights = tail_guarded_weight(self.anchor_distances, distance, self.anchor_scale, calibration)
        q, p = query_weight_index_vectors(self.anchor_queries, self.anchor_candidates, queries, candidates, weights)
        metrics, ranks = rank_metrics(q, p, self.query_ids, self.candidate_ids,
                                     relevance, self.device, self.query_batch_size, 20)
        eligible = head_guards_pass(metrics, self.reference_metrics,
                                   self.hit1_tolerance, self.mrr_tolerance)
        payload = {"train_ids": self.train_ids, "training_splits": ["train"],
            "transfer_bank": bank.cpu(), "transfer_leave_self_out_distance": loo.cpu(),
            "calibration": calibration, "anchor_scale": self.anchor_scale,
            "query_ids": self.query_ids, "query_weights": weights,
            "query_transfer_distances": distance, "reference_metrics": self.reference_metrics,
            "validation_used_for_support_fitting": False, "test_opened": False}
        return metrics, ranks, eligible, payload
