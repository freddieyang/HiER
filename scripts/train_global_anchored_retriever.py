#!/usr/bin/env python3
"""Train the global-anchored structured retriever on official ReactZyme.

The frozen Morgan/whole-sequence-ESM checkpoint is evaluated before any update
and saved as epoch -1. Atom/motif, local-sequence, and latent EC branches enter
through zero-initialized bounded residuals. Validation can therefore retain the
clean anchor whenever learned residuals do not improve it. Test remains sealed
until checkpoint selection is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_retrieval import (  # noqa: E402
    GlobalAnchoredStructuredRetriever,
    ReactionGraphStore,
    cross_modal_ec_alignment_loss,
    hierarchical_multi_positive_loss,
    hierarchy_consistency_loss,
    multi_positive_infonce,
    multi_positive_tail_risk_loss,
)
from scripts.train_structured_retriever import (  # noqa: E402
    DEFAULT_GRAPH_BUNDLE,
    DEFAULT_OFFICIAL_UNIPROT_SOURCE,
    DEFAULT_PROCESSED,
    DEFAULT_TEST_CONTRACT,
    DEFAULT_TOKEN_MANIFEST,
    DEFAULT_VALIDATION_CONTRACT,
    ProteinTokenStore,
    amp_policy,
    append_jsonl,
    atomic_json,
    build_ec_vocab,
    build_train_adjacency,
    dense_target_masks,
    ec_diagnostics,
    infer_ec_depth,
    load_contract,
    load_official_ec_annotations,
    protein_target_tokens,
    rank_metrics,
    sample_query_balanced_batch,
    seed_everything,
    target_tokens_for_ids,
    validate_heldout_test_inputs,
    validate_official_inputs,
)


DEFAULT_GLOBAL_BUNDLE = (
    PROJECT_ROOT / "data/features/global/features.pt"
)
DEFAULT_PROTEIN_IDS = (
    DEFAULT_PROCESSED
    / "external_baselines/clipzyme/protein_mean_embeddings.ids.json"
)
DEFAULT_PROTEIN_EMBEDDINGS = (
    DEFAULT_PROCESSED
    / "external_baselines/clipzyme/protein_mean_embeddings.safetensors"
)
DEFAULT_ANCHOR_CHECKPOINT = (
    PROJECT_ROOT / "runs/legacy_anchor/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--graph-bundle", type=Path, default=DEFAULT_GRAPH_BUNDLE)
    parser.add_argument("--protein-token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--global-feature-bundle", type=Path, default=DEFAULT_GLOBAL_BUNDLE)
    parser.add_argument("--protein-global-ids", type=Path, default=DEFAULT_PROTEIN_IDS)
    parser.add_argument(
        "--protein-global-embeddings", type=Path, default=DEFAULT_PROTEIN_EMBEDDINGS
    )
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_CHECKPOINT)
    parser.add_argument(
        "--validation-contract", type=Path, default=DEFAULT_VALIDATION_CONTRACT
    )
    parser.add_argument("--test-contract", type=Path, default=DEFAULT_TEST_CONTRACT)
    parser.add_argument(
        "--official-uniprot-source", type=Path, default=DEFAULT_OFFICIAL_UNIPROT_SOURCE
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument(
        "--positive-sampling", choices=("random", "coverage"), default="random"
    )
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--hard-candidate-bank-size", type=int, default=0)
    parser.add_argument("--hard-candidates-per-query", type=int, default=0)
    parser.add_argument("--hard-candidate-query-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--anchor-hidden-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--residual-scale-limit", type=float, default=0.5)
    parser.add_argument("--num-query-prototypes", type=int, default=1)
    parser.add_argument("--prototype-scale-limit", type=float, default=0.25)
    parser.add_argument(
        "--prototype-aggregation-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--symmetric-retrieval-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ec-loss-weight", type=float, default=0.10)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.02)
    parser.add_argument("--cross-modal-ec-weight", type=float, default=0.02)
    parser.add_argument("--all-positive-weight", type=float, default=0.0)
    parser.add_argument("--tail-risk-weight", type=float, default=0.0)
    parser.add_argument("--tail-risk-fraction", type=float, default=0.20)
    parser.add_argument(
        "--tail-risk-score", choices=("raw", "positive_count_corrected"), default="raw",
        help="Score used to select hard queries; corrected removes the log positive-count floor.",
    )
    parser.add_argument("--aux-warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-query-batch-size", type=int, default=64)
    parser.add_argument("--eval-protein-batch-size", type=int, default=2048)
    parser.add_argument("--rank-query-batch-size", type=int, default=128)
    parser.add_argument("--top-k-output", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=("first_mrr", "mean_rank_pareto"),
        default="first_mrr",
    )
    parser.add_argument("--selection-hit1-tolerance", type=float, default=0.003)
    parser.add_argument("--selection-official-mrr-tolerance", type=float, default=0.002)
    parser.add_argument("--selection-mean-rank-min-improvement", type=float, default=0.0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-eval-queries", type=int, default=0)
    parser.add_argument("--max-eval-candidates", type=int, default=0)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument(
        "--refit-train-val",
        action="store_true",
        help=(
            "Disabled legacy option: validation associations may never be used "
            "for optimization under the strict train-only benchmark."
        ),
    )
    parser.add_argument(
        "--warm-graph-cache", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


class GlobalFeatureStore:
    """Aligned query-only global features for the two frozen anchor encoders."""

    def __init__(
        self,
        feature_bundle: Path,
        protein_ids_path: Path,
        protein_embeddings_path: Path,
        expected_reaction_ids: Sequence[str],
        expected_protein_ids: Sequence[str],
    ) -> None:
        bundle = torch.load(feature_bundle, map_location="cpu", weights_only=False)
        required = {"reaction_ids", "reaction_fingerprints", "feature_contract"}
        if not required.issubset(bundle):
            raise ValueError("Global reaction feature bundle is incomplete")
        contract = bundle["feature_contract"]
        forbidden_true = (
            "uses_pair_associations",
            "uses_split_labels",
            "uses_enzyme_or_ec_labels",
            "uses_reaction_direction",
        )
        if any(bool(contract.get(key)) for key in forbidden_true):
            raise ValueError("Global reaction feature bundle violates query-only contract")
        self.reaction_ids = list(map(str, bundle["reaction_ids"]))
        if self.reaction_ids != list(map(str, expected_reaction_ids)):
            raise ValueError("Morgan rows are not exactly aligned to reaction graphs")
        self.reaction_features = torch.as_tensor(
            bundle["reaction_fingerprints"]
        ).float()
        if self.reaction_features.shape[0] != len(self.reaction_ids):
            raise ValueError("Morgan feature rows and IDs differ")

        ids_payload = json.loads(protein_ids_path.read_text(encoding="utf-8"))
        self.protein_ids = list(map(str, ids_payload["enzyme_ids"]))
        if self.protein_ids != list(map(str, expected_protein_ids)):
            raise ValueError("Whole-sequence ESM rows are not exactly aligned to local tokens")
        with safe_open(
            str(protein_embeddings_path), framework="pt", device="cpu"
        ) as handle:
            if list(handle.keys()) != ["embeddings"]:
                raise ValueError("Unexpected global protein embedding keys")
            self.protein_features = handle.get_tensor("embeddings")
        if self.protein_features.shape[0] != len(self.protein_ids):
            raise ValueError("Global protein feature rows and IDs differ")
        if not torch.isfinite(self.reaction_features).all():
            raise ValueError("Morgan features contain non-finite values")

    def reaction(self, rows: Sequence[int]) -> torch.Tensor:
        return self.reaction_features[torch.as_tensor(rows, dtype=torch.long)]

    def protein(self, rows: Sequence[int]) -> torch.Tensor:
        return self.protein_features[torch.as_tensor(rows, dtype=torch.long)]


def validate_anchor_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    processed: Path,
    global_feature_bundle: Path,
    protein_ids_path: Path,
    protein_embeddings_path: Path,
    hidden_dim: int,
    seed: int,
) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    if config.get("reaction_only") is not True:
        raise ValueError("Global anchor must be a clean reaction-only checkpoint")
    if config.get("ground_truth_ec_is_model_input") is not False:
        raise ValueError("Anchor checkpoint does not certify query-only inference")
    if Path(config.get("processed_dir", "")).resolve() != processed.resolve():
        raise ValueError("Anchor checkpoint was trained on a different processed dataset")
    if int(config.get("embedding_dim", -1)) != hidden_dim:
        raise ValueError("Anchor embedding dimension differs from hybrid hidden dimension")
    if int(config.get("seed", -1)) != int(seed):
        raise ValueError("Anchor checkpoint seed differs from the hybrid run seed")
    expected_paths = {
        "feature_bundle": global_feature_bundle,
        "protein_ids": protein_ids_path,
        "protein_embeddings": protein_embeddings_path,
    }
    for key, expected in expected_paths.items():
        if Path(config.get(key, "")).resolve() != expected.resolve():
            raise ValueError(f"Anchor checkpoint {key} differs from hybrid inputs")
    validation_metrics = checkpoint.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        raise ValueError("Anchor checkpoint lacks validation selection metrics")
    source_run_dir = Path(config.get("output_dir", "")).resolve()
    source_summary_path = source_run_dir / "summary.json"
    if not source_summary_path.exists():
        raise FileNotFoundError(source_summary_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("test_used_for_selection") is not False:
        raise ValueError("Anchor source does not certify validation-only selection")
    if int(source_summary.get("best_epoch", -999)) != int(checkpoint["epoch"]):
        raise ValueError("Anchor checkpoint epoch differs from its source summary")
    return {
        "status": "passed",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_seed": int(config["seed"]),
        "reaction_only": True,
        "ground_truth_ec_is_model_input": False,
        "test_used_for_weight_selection": False,
        "source_summary": str(source_summary_path),
        "feature_paths_match_hybrid_inputs": True,
        "source_validation_metrics": validation_metrics,
    }


def validate_global_feature_provenance(
    feature_bundle: Path,
    protein_ids_path: Path,
    protein_embeddings_path: Path,
    processed: Path,
    reaction_count: int,
    protein_count: int,
) -> dict[str, Any]:
    """Verify source manifests for both global anchors without pair-table access."""

    reaction_manifest_path = feature_bundle.parent / "manifest.json"
    protein_report_path = protein_embeddings_path.with_name(
        "protein_mean_embeddings.report.json"
    )
    for path in (reaction_manifest_path, protein_report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    reaction_manifest = json.loads(reaction_manifest_path.read_text(encoding="utf-8"))
    if reaction_manifest.get("status") != "completed":
        raise ValueError("Morgan inventory feature manifest is incomplete")
    for key in (
        "pair_or_split_tables_read",
        "enzyme_associations_read",
        "validation_or_test_associations_read",
        "ec_labels_read",
        "reaction_direction_used",
    ):
        if reaction_manifest.get(key) is not False:
            raise ValueError(f"Morgan global feature provenance failed: {key}")
    if Path(reaction_manifest["source_reactions"]).resolve() != (
        processed / "reactions.parquet"
    ).resolve():
        raise ValueError("Morgan features were built from another reaction table")
    if Path(reaction_manifest["feature_bundle"]).resolve() != feature_bundle.resolve():
        raise ValueError("Morgan feature manifest points to another bundle")
    if int(reaction_manifest["reaction_count"]) != int(reaction_count):
        raise ValueError("Morgan manifest reaction count differs from graph rows")

    protein_report = json.loads(protein_report_path.read_text(encoding="utf-8"))
    if protein_report.get("baseline") != "CLIPZyme-ESM-Frozen":
        raise ValueError("Unexpected whole-sequence protein feature source")
    if int(protein_report.get("explicit_negative_pairs_loaded", -1)) != 0:
        raise ValueError("Protein global feature build loaded explicit negatives")
    if Path(protein_report["ids_path"]).resolve() != protein_ids_path.resolve():
        raise ValueError("Protein global feature report points to other IDs")
    if Path(protein_report["output"]).resolve() != protein_embeddings_path.resolve():
        raise ValueError("Protein global feature report points to another tensor")
    if int(protein_report["enzyme_count"]) != int(protein_count):
        raise ValueError("Protein global report count differs from token rows")
    source_manifest = Path(protein_report["source_manifest"]).resolve()
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    return {
        "status": "passed",
        "reaction_manifest": str(reaction_manifest_path),
        "reaction_source": str((processed / "reactions.parquet").resolve()),
        "reaction_pair_split_or_ec_sources_read": False,
        "reaction_direction_used": False,
        "protein_report": str(protein_report_path),
        "protein_source_manifest": str(source_manifest),
        "protein_feature_method": protein_report["baseline"],
        "explicit_negative_pairs_loaded": 0,
        "reaction_rows": int(reaction_count),
        "protein_rows": int(protein_count),
    }


@torch.inference_mode()
def encode_queries(
    model: GlobalAnchoredStructuredRetriever,
    graph_store: ReactionGraphStore,
    global_store: GlobalFeatureStore,
    reaction_rows: Sequence[int],
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.eval()
    amp_enabled, autocast_dtype = amp_policy(device, amp_dtype)
    vectors: list[torch.Tensor] = []
    predictions: list[list[torch.Tensor]] = [[] for _ in range(model.ec_router.depth)]
    for start in range(0, len(reaction_rows), batch_size):
        rows = reaction_rows[start : start + batch_size]
        graph_batch = graph_store.collate(rows).to(device)
        global_features = global_store.reaction(rows).to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
        ):
            encoded = model.encode_reaction(graph_batch, global_features)
        query_vectors = (
            encoded["prototypes"]
            if model.num_query_prototypes > 1
            else encoded["embedding"]
        )
        vectors.append(query_vectors.float().cpu())
        for level, value in enumerate(encoded["logits"]):
            predictions[level].append(value.float().argmax(dim=1).cpu())
    return torch.cat(vectors), [torch.cat(values) for values in predictions]


@torch.inference_mode()
def encode_candidates(
    model: GlobalAnchoredStructuredRetriever,
    token_store: ProteinTokenStore,
    global_store: GlobalFeatureStore,
    candidate_rows: Sequence[int],
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> torch.Tensor:
    model.eval()
    amp_enabled, autocast_dtype = amp_policy(device, amp_dtype)
    vectors: list[torch.Tensor] = []
    for start in range(0, len(candidate_rows), batch_size):
        rows = candidate_rows[start : start + batch_size]
        tokens, mask = token_store.get(rows)
        global_features = global_store.protein(rows)
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        global_features = global_features.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
        ):
            encoded = model.encode_protein(tokens, mask, global_features)
        vectors.append(encoded["embedding"].float().cpu())
    return torch.cat(vectors)


@torch.inference_mode()
def build_global_anchor_hard_candidate_bank(
    model: GlobalAnchoredStructuredRetriever,
    global_store: GlobalFeatureStore,
    query_rows: Sequence[int],
    candidate_rows: Sequence[int] | np.ndarray,
    adjacency: dict[int, tuple[int, ...]],
    device: torch.device,
    *,
    bank_size: int,
    query_batch_size: int,
    protein_batch_size: int,
    association_splits: Sequence[str] = ("train",),
) -> tuple[dict[int, tuple[int, ...]], dict[str, Any]]:
    """Mine globally confusing optimization candidates with frozen anchors.

    Observed positives are removed before top-k selection.  Remaining entries
    are unobserved denominator candidates rather than asserted biological
    negatives.  Only the caller-provided optimization splits are represented.
    """

    if bank_size <= 0:
        return {}, {
            "status": "disabled",
            "bank_size": 0,
            "query_count": len(query_rows),
        }
    if query_batch_size <= 0 or protein_batch_size <= 0:
        raise ValueError("Hard-candidate mining batch sizes must be positive")
    candidate_rows = list(map(int, candidate_rows))
    if not candidate_rows:
        raise ValueError("Hard-candidate mining received no candidates")
    model.eval()
    protein_vectors: list[torch.Tensor] = []
    for start in range(0, len(candidate_rows), protein_batch_size):
        rows = candidate_rows[start : start + protein_batch_size]
        features = global_store.protein(rows).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            vector = model.protein_global_encoder(features.float())
        protein_vectors.append(F.normalize(vector.float(), dim=-1))
    protein_matrix = torch.cat(protein_vectors, dim=0)  # [Ct,D]
    candidate_column = {row: column for column, row in enumerate(candidate_rows)}
    maximum_bank_size = max(
        min(bank_size, len(candidate_rows) - len(adjacency[int(query)]))
        for query in query_rows
    )
    if maximum_bank_size <= 0:
        raise ValueError("No unobserved train candidates remain after positive masking")

    bank: dict[int, tuple[int, ...]] = {}
    for start in range(0, len(query_rows), query_batch_size):
        batch_rows = list(map(int, query_rows[start : start + query_batch_size]))
        features = global_store.reaction(batch_rows).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            query_vectors = model.reaction_global_encoder(features.float())
        scores = F.normalize(query_vectors.float(), dim=-1) @ protein_matrix.T
        mask_rows: list[int] = []
        mask_columns: list[int] = []
        for local_row, query_row in enumerate(batch_rows):
            for positive in adjacency[query_row]:
                column = candidate_column.get(int(positive))
                if column is not None:
                    mask_rows.append(local_row)
                    mask_columns.append(column)
        if mask_rows:
            scores[
                torch.tensor(mask_rows, device=device),
                torch.tensor(mask_columns, device=device),
            ] = -torch.inf
        top_columns = torch.topk(
            scores, k=maximum_bank_size, dim=1, largest=True, sorted=True
        ).indices.cpu()
        for local_row, query_row in enumerate(batch_rows):
            allowed_count = min(
                bank_size, len(candidate_rows) - len(adjacency[query_row])
            )
            values = tuple(
                candidate_rows[int(column)]
                for column in top_columns[local_row, :allowed_count]
            )
            if set(values) & set(adjacency[query_row]):
                raise RuntimeError("Observed positive leaked into hard-candidate bank")
            bank[query_row] = values

    lengths = [len(values) for values in bank.values()]
    audit = {
        "status": "completed",
        "source": "frozen_global_anchor_cosine",
        "association_splits_read": list(map(str, association_splits)),
        "validation_or_test_associations_read": any(
            split in {"val", "test"} for split in association_splits
        ),
        "test_associations_read": "test" in set(association_splits),
        "candidate_semantics": "unobserved_train_pool_denominator_only",
        "query_count": len(bank),
        "candidate_universe_count": len(candidate_rows),
        "requested_bank_size": int(bank_size),
        "minimum_bank_size": int(min(lengths)),
        "maximum_bank_size": int(max(lengths)),
        "observed_positive_overlap": 0,
    }
    return bank, audit


@torch.inference_mode()
def evaluate_split(
    model: GlobalAnchoredStructuredRetriever,
    split: str,
    contract_path: Path,
    graph_store: ReactionGraphStore,
    token_store: ProteinTokenStore,
    global_store: GlobalFeatureStore,
    reaction_row: dict[str, int],
    split_labels: dict[str, set[str]],
    token_to_index: list[dict[str, int]],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    query_ids, candidate_ids, relevance = load_contract(
        contract_path,
        split,
        max_queries=args.max_eval_queries,
        max_candidates=args.max_eval_candidates,
    )
    missing_queries = [value for value in query_ids if value not in reaction_row]
    missing_candidates = [value for value in candidate_ids if value not in token_store.row]
    if missing_queries or missing_candidates:
        raise ValueError(
            f"Evaluation features missing queries={missing_queries[:3]}, "
            f"candidates={missing_candidates[:3]}"
        )
    query_vectors, top1 = encode_queries(
        model,
        graph_store,
        global_store,
        [reaction_row[value] for value in query_ids],
        device,
        args.eval_query_batch_size,
        args.amp_dtype,
    )
    candidate_vectors = encode_candidates(
        model,
        token_store,
        global_store,
        [token_store.row[value] for value in candidate_ids],
        device,
        args.eval_protein_batch_size,
        args.amp_dtype,
    )
    metrics, predictions = rank_metrics(
        query_vectors,
        candidate_vectors,
        query_ids,
        candidate_ids,
        relevance,
        device,
        args.rank_query_batch_size,
        args.top_k_output,
        score_function=model.score_embeddings,
    )
    metrics.update(ec_diagnostics(query_ids, top1, split_labels, token_to_index))
    return metrics, predictions


def checkpoint_payload(
    model: GlobalAnchoredStructuredRetriever,
    epoch: int,
    validation_metrics: dict[str, Any],
    vocab_sizes: Sequence[int],
    ec_depth: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "epoch": int(epoch),
        "validation_metrics": validation_metrics,
        "ec_vocab_sizes": list(map(int, vocab_sizes)),
        "ec_depth": int(ec_depth),
        "config": config,
        "global_anchor_frozen": True,
    }


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_is_better(
    candidate: dict[str, Any],
    best: dict[str, Any],
    anchor: dict[str, Any],
    *,
    mode: str,
    hit1_tolerance: float,
    official_mrr_tolerance: float,
    mean_rank_min_improvement: float,
) -> bool:
    """Apply validation-only checkpoint selection with an anchor safeguard."""

    if mode == "first_mrr":
        return float(candidate["mrr"]) > float(best["mrr"]) + 1e-12
    if mode != "mean_rank_pareto":
        raise ValueError(f"Unknown selection mode: {mode}")
    preserves_head = (
        float(candidate["hit@1"])
        >= float(anchor["hit@1"]) - hit1_tolerance
    )
    preserves_all_relevant_mrr = (
        float(candidate["official_reactzyme_mrr"])
        >= float(anchor["official_reactzyme_mrr"])
        - official_mrr_tolerance
    )
    if not (preserves_head and preserves_all_relevant_mrr):
        return False
    return (
        float(candidate["official_reactzyme_mean_rank"])
        < float(best["official_reactzyme_mean_rank"])
        - mean_rank_min_improvement
    )


def validate_train_only_protocol(refit_train_val: bool) -> None:
    """Fail before loading data or creating a run if refit was requested."""
    if refit_train_val:
        raise ValueError(
            "Train+validation refit is disabled. Only official train associations "
            "may update parameters; validation is reserved for model selection."
        )


def main() -> None:
    args = parse_args()
    validate_train_only_protocol(args.refit_train_val)
    positive_ints = (
        args.epochs,
        args.patience,
        args.query_batch_size,
        args.positives_per_query,
        args.hard_candidate_query_batch_size,
        args.num_query_prototypes,
        args.eval_query_batch_size,
        args.eval_protein_batch_size,
    )
    if any(value <= 0 for value in positive_ints):
        raise ValueError("Epoch, batch, patience, and positive counts must be positive")
    if any(
        value < 0
        for value in (
            args.ec_loss_weight,
            args.consistency_loss_weight,
            args.cross_modal_ec_weight,
            args.all_positive_weight,
            args.tail_risk_weight,
            args.aux_warmup_epochs,
            args.candidate_pool_size,
            args.hard_candidate_bank_size,
            args.hard_candidates_per_query,
            args.selection_hit1_tolerance,
            args.selection_official_mrr_tolerance,
            args.selection_mean_rank_min_improvement,
        )
    ):
        raise ValueError("Loss weights, warm-up, and candidate pool must be non-negative")
    if args.hard_candidates_per_query > args.hard_candidate_bank_size:
        raise ValueError(
            "hard-candidates-per-query cannot exceed hard-candidate-bank-size"
        )
    if args.prototype_scale_limit <= 0:
        raise ValueError("prototype-scale-limit must be positive")
    if args.prototype_aggregation_temperature <= 0:
        raise ValueError("prototype-aggregation-temperature must be positive")
    if args.all_positive_weight > 1 or args.tail_risk_weight > 1:
        raise ValueError("Tail-aware retrieval weights must lie in [0, 1]")
    if not 0 < args.tail_risk_fraction <= 1:
        raise ValueError("tail-risk-fraction must lie in (0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_enabled, autocast_dtype = amp_policy(device, args.amp_dtype)

    processed = args.processed_dir.resolve()
    graph_bundle_path = args.graph_bundle.resolve()
    token_manifest_path = args.protein_token_manifest.resolve()
    validation_contract_path = args.validation_contract.resolve()
    test_contract_path = args.test_contract.resolve()
    official_uniprot_source = args.official_uniprot_source.resolve()
    official_audit = validate_official_inputs(
        processed,
        graph_bundle_path,
        token_manifest_path,
        validation_contract_path,
        official_uniprot_source,
    )
    atomic_json(output_dir / "official_data_and_leakage_audit.json", official_audit)

    graph_bundle = torch.load(graph_bundle_path, map_location="cpu", weights_only=False)
    graph_store = ReactionGraphStore(graph_bundle)
    reaction_row = {value: index for index, value in enumerate(graph_store.reaction_ids)}
    token_store = ProteinTokenStore(token_manifest_path)
    global_store = GlobalFeatureStore(
        args.global_feature_bundle.resolve(),
        args.protein_global_ids.resolve(),
        args.protein_global_embeddings.resolve(),
        graph_store.reaction_ids,
        token_store.ids,
    )
    global_feature_audit = validate_global_feature_provenance(
        args.global_feature_bundle.resolve(),
        args.protein_global_ids.resolve(),
        args.protein_global_embeddings.resolve(),
        processed,
        len(graph_store.reaction_ids),
        len(token_store.ids),
    )
    atomic_json(output_dir / "global_feature_provenance_audit.json", global_feature_audit)

    training_splits = ("train",)
    optimization_pairs = pd.read_parquet(
        processed / "pairs.parquet",
        columns=["reaction_id", "enzyme_id", "split"],
        filters=[("split", "in", list(training_splits))],
    )
    if set(optimization_pairs["split"].astype(str)) != set(training_splits):
        raise ValueError("Optimization pair loader exposed an unexpected split")
    missing_reactions = sorted(
        set(optimization_pairs["reaction_id"].astype(str)) - set(reaction_row)
    )
    missing_proteins = sorted(
        set(optimization_pairs["enzyme_id"].astype(str)) - set(token_store.row)
    )
    if missing_reactions or missing_proteins:
        raise ValueError(
            f"Training features missing reactions={missing_reactions[:3]}, "
            f"proteins={missing_proteins[:3]}"
        )

    train_val_labels, ec_audit = load_official_ec_annotations(
        processed, official_uniprot_source, ["train", "val"]
    )
    atomic_json(output_dir / "official_train_val_ec_supervision_audit.json", ec_audit)
    train_labels = train_val_labels.get("train", {})
    val_labels = train_val_labels.get("val", {})
    optimization_labels = {
        reaction_id: set(values) for reaction_id, values in train_labels.items()
    }
    ec_depth = infer_ec_depth(train_labels)
    vocabularies, token_to_index, child_to_parent = build_ec_vocab(
        train_labels, ec_depth
    )
    vocab_sizes = [len(values) for values in vocabularies]
    reaction_targets = target_tokens_for_ids(
        graph_store.reaction_ids, optimization_labels, token_to_index
    )
    protein_targets = protein_target_tokens(
        token_store.ids, optimization_pairs, optimization_labels, token_to_index
    )
    adjacency = build_train_adjacency(
        optimization_pairs, reaction_row, token_store.row
    )
    train_candidate_rows = np.asarray(
        sorted(
            {
                token_store.row[str(value)]
                for value in optimization_pairs["enzyme_id"]
            }
        ),
        dtype=np.int64,
    )
    train_query_rows = sorted(adjacency)
    if args.max_train_queries > 0:
        train_query_rows = train_query_rows[: args.max_train_queries]
    if args.warm_graph_cache:
        graph_store.warm(train_query_rows)

    anchor_checkpoint = torch.load(
        args.anchor_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    anchor_audit = validate_anchor_checkpoint(
        anchor_checkpoint,
        args.anchor_checkpoint.resolve(),
        processed,
        args.global_feature_bundle.resolve(),
        args.protein_global_ids.resolve(),
        args.protein_global_embeddings.resolve(),
        args.hidden_dim,
        args.seed,
    )
    atomic_json(output_dir / "global_anchor_audit.json", anchor_audit)
    model = GlobalAnchoredStructuredRetriever(
        atom_feature_dim=graph_store.atom_feature_dim,
        edge_feature_dim=graph_store.edge_feature_dim,
        motif_feature_dim=graph_store.motif_feature_dim,
        ec_vocab_sizes=vocab_sizes,
        child_to_parent=child_to_parent,
        reaction_global_dim=int(global_store.reaction_features.shape[1]),
        protein_global_dim=int(global_store.protein_features.shape[1]),
        protein_token_dim=int(token_store.shape[2]),
        protein_segments=int(token_store.shape[1]),
        anchor_hidden_dim=args.anchor_hidden_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        temperature=args.temperature,
        residual_scale_limit=args.residual_scale_limit,
        num_query_prototypes=args.num_query_prototypes,
        prototype_scale_limit=args.prototype_scale_limit,
        prototype_aggregation_temperature=args.prototype_aggregation_temperature,
        freeze_global_anchors=True,
    )
    model.load_global_anchor_checkpoint(anchor_checkpoint)
    model.to(device)

    hard_candidate_bank, hard_candidate_audit = (
        build_global_anchor_hard_candidate_bank(
            model,
            global_store,
            train_query_rows,
            train_candidate_rows,
            adjacency,
            device,
            bank_size=args.hard_candidate_bank_size,
            query_batch_size=args.hard_candidate_query_batch_size,
            protein_batch_size=args.eval_protein_batch_size,
            association_splits=training_splits,
        )
        if args.hard_candidate_bank_size > 0
        else (
            {},
            {
                "status": "disabled",
                "bank_size": 0,
                "query_count": len(train_query_rows),
                "association_splits_read": list(training_splits),
                "validation_or_test_associations_read": bool(
                    args.refit_train_val
                ),
            },
        )
    )
    if args.refit_train_val:
        hard_candidate_audit["validation_associations_used_for_refit"] = True
        hard_candidate_audit["test_associations_read"] = False
        hard_candidate_audit["validation_or_test_associations_read"] = True
    atomic_json(output_dir / "hard_candidate_mining_audit.json", hard_candidate_audit)
    if hard_candidate_bank:
        atomic_torch_save(
            output_dir / "hard_candidate_bank.pt",
            {
                "bank": hard_candidate_bank,
                "reaction_ids": graph_store.reaction_ids,
                "protein_ids": token_store.ids,
                "training_splits": list(training_splits),
                "audit": hard_candidate_audit,
            },
        )

    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "task_definition": "reaction_query_to_enzyme_retrieval",
            "inference_inputs": [
                "official_reaction_molecule_inventory",
                "candidate_protein_sequence",
            ],
            "ground_truth_ec_is_model_input": False,
            "architecture": (
                "global_anchored_multi_prototype_structured_retriever"
                if args.num_query_prototypes > 1
                else "frozen_global_anchor_plus_bounded_structured_ec_residuals"
            ),
            "global_anchor_frozen": True,
            "initial_checkpoint_policy": "epoch_minus_one_exact_clean_anchor",
            "selection_metric": (
                "validation_mrr_first_relevant"
                if args.selection_mode == "first_mrr"
                else "validation_mean_rank_with_anchor_hit1_and_all_relevant_mrr_guards"
            ),
            "test_used_for_selection": False,
            "candidate_indexable": True,
            "training_splits": list(training_splits),
            "training_mode": (
                "fixed_epoch_train_validation_refit"
                if args.refit_train_val
                else "validation_selected_train_only"
            ),
            "checkpoint_selection": (
                "disabled_fixed_epoch"
                if args.refit_train_val
                else "validation_only"
            ),
            "ec_depth": ec_depth,
            "ec_depth_policy": "auto_detect_complete_depth_from_train_exact_one_only",
            "external_rhea_to_ec_snapshot_used": False,
        }
    )
    atomic_json(output_dir / "config.json", config)
    atomic_json(
        output_dir / "ec_vocabulary.json",
        {
            "source": "training_split_strict_exact_one_only",
            "depth": ec_depth,
            "levels": {
                f"level_{level + 1}": values
                for level, values in enumerate(vocabularies)
            },
            "test_annotations_loaded_during_vocabulary_build": False,
        },
    )
    atomic_json(
        output_dir / "data_audit.json",
        {
            "official_gate": "passed",
            "optimization_splits": list(training_splits),
            "optimization_positive_edges": int(len(optimization_pairs)),
            "optimization_queries": len(adjacency),
            "optimization_enzymes": int(
                optimization_pairs["enzyme_id"].nunique()
            ),
            "strict_ec_labeled_train_reactions": len(train_labels),
            "strict_ec_labeled_optimization_reactions": len(optimization_labels),
            "ec_depth": ec_depth,
            "ec_vocab_sizes": vocab_sizes,
            "global_reaction_features": "summed_Morgan_inventory_query_only",
            "global_protein_features": "frozen_whole_sequence_ESM_mean_label_independent",
            "structured_reaction_features": "atom_motif_graph_query_only",
            "local_protein_features": "frozen_ESM_contiguous_segment_means",
            "positive_sampling": (
                "query_balanced_least_exposed_train_only"
                if args.positive_sampling == "coverage"
                else "query_balanced_random_train_only"
            ),
            "positives_per_query": args.positives_per_query,
            "candidate_pool_size": args.candidate_pool_size,
            "hard_candidate_bank_size": args.hard_candidate_bank_size,
            "hard_candidates_per_query": args.hard_candidates_per_query,
            "hard_candidate_source": "frozen_anchor_optimization_pool_only",
            "unobserved_pair_policy": "contrastive_denominator_only",
            "test_pairs_loaded_before_selection": False,
            "test_relevance_loaded_before_selection": False,
        },
    )
    atomic_json(
        output_dir / "provenance.json",
        {
            "processed_dataset": str(processed),
            "reaction_global_bundle": str(args.global_feature_bundle.resolve()),
            "protein_global_embeddings": str(args.protein_global_embeddings.resolve()),
            "reaction_graph_bundle": str(graph_bundle_path),
            "protein_token_manifest": str(token_manifest_path),
            "global_anchor_checkpoint": str(args.anchor_checkpoint.resolve()),
            "validation_contract": str(validation_contract_path),
            "test_contract_sealed_until_checkpoint_selection": str(test_contract_path),
            "official_uniprot_rhea_source": str(official_uniprot_source),
        },
    )

    train_log = output_dir / "train_metrics.jsonl"
    val_log = output_dir / "validation_metrics.jsonl"
    train_log.unlink(missing_ok=True)
    val_log.unlink(missing_ok=True)
    best_path = output_dir / "best.pt"

    # Evaluate and save the exact clean anchor before any residual update.
    initial_metrics, _ = evaluate_split(
        model,
        "val",
        validation_contract_path,
        graph_store,
        token_store,
        global_store,
        reaction_row,
        val_labels,
        token_to_index,
        device,
        args,
    )
    initial_metrics = {"epoch": -1, **initial_metrics}
    append_jsonl(val_log, initial_metrics)
    source_metrics = anchor_checkpoint["validation_metrics"]
    full_validation = args.max_eval_queries == 0 and args.max_eval_candidates == 0
    metric_deltas = (
        {
            key: abs(float(initial_metrics[key]) - float(source_metrics[key]))
            for key in ("mrr", "hit@1", "hit@10", "official_reactzyme_mrr")
        }
        if full_validation
        else {}
    )
    equivalent = not full_validation or max(metric_deltas.values()) <= 1e-3
    initial_audit = {
        "status": "passed" if equivalent else "failed",
        "scope": "full_official_validation" if full_validation else "subset_smoke_only",
        "metric_absolute_deltas": metric_deltas,
        "tolerance": 1e-3,
        "residual_scales": {
            key: float(value.detach().cpu())
            for key, value in model.residual_scales().items()
        },
        "hybrid_initial_validation_metrics": initial_metrics,
        "source_anchor_validation_metrics": source_metrics,
    }
    atomic_json(output_dir / "initial_anchor_equivalence.json", initial_audit)
    if not equivalent:
        raise RuntimeError("Hybrid zero-residual state did not reproduce the clean anchor")
    atomic_torch_save(
        best_path,
        checkpoint_payload(model, -1, initial_metrics, vocab_sizes, ec_depth, config),
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16"
    )

    best_metrics = initial_metrics
    best_mrr = float(initial_metrics["mrr"])
    best_epoch = -1
    epochs_without_improvement = 0
    rng = np.random.default_rng(args.seed)
    negative_rng = np.random.default_rng(args.seed + 1_000_003)
    hard_candidate_rng = np.random.default_rng(args.seed + 2_000_033)
    positive_exposure_counts = (
        {
            query: np.zeros(len(positives), dtype=np.int64)
            for query, positives in adjacency.items()
        }
        if args.positive_sampling == "coverage"
        else None
    )
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        permutation = rng.permutation(np.asarray(train_query_rows, dtype=np.int64))
        totals = defaultdict(float)
        steps = 0
        for start in range(0, len(permutation), args.query_batch_size):
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
            query_rows = list(map(int, permutation[start : start + args.query_batch_size]))
            candidate_rows, positive_mask_cpu = sample_query_balanced_batch(
                query_rows,
                adjacency,
                args.positives_per_query,
                rng,
                candidate_universe=train_candidate_rows,
                target_candidate_count=args.candidate_pool_size,
                negative_rng=negative_rng,
                positive_exposure_counts=positive_exposure_counts,
                hard_candidate_bank=(
                    hard_candidate_bank if hard_candidate_bank else None
                ),
                hard_candidates_per_query=args.hard_candidates_per_query,
                hard_candidate_rng=hard_candidate_rng,
            )
            graph_batch = graph_store.collate(query_rows).to(device)
            reaction_global = global_store.reaction(query_rows).to(
                device, non_blocking=True
            )
            protein_tokens, protein_mask = token_store.get(candidate_rows)
            protein_global = global_store.protein(candidate_rows)
            protein_tokens = protein_tokens.to(device, non_blocking=True)
            protein_mask = protein_mask.to(device, non_blocking=True)
            protein_global = protein_global.to(device, non_blocking=True)
            positive_mask = positive_mask_cpu.to(device)
            reaction_targets_batch = dense_target_masks(
                query_rows, reaction_targets, vocab_sizes, device
            )
            protein_targets_batch = dense_target_masks(
                candidate_rows, protein_targets, vocab_sizes, device
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
            ):
                scores, reaction, protein = model(
                    graph_batch,
                    reaction_global,
                    protein_tokens,
                    protein_mask,
                    protein_global,
                )
                if args.all_positive_weight or args.tail_risk_weight:
                    retrieval_loss, retrieval_components = (
                        multi_positive_tail_risk_loss(
                            scores,
                            positive_mask,
                            all_positive_weight=args.all_positive_weight,
                            tail_risk_weight=args.tail_risk_weight,
                            tail_fraction=args.tail_risk_fraction,
                            symmetric=args.symmetric_retrieval_loss,
                            tail_risk_score=args.tail_risk_score,
                        )
                    )
                else:
                    retrieval_loss = multi_positive_infonce(
                        scores,
                        positive_mask,
                        symmetric=args.symmetric_retrieval_loss,
                    )
                    retrieval_components = {
                        "any_positive": retrieval_loss,
                        "all_positive": retrieval_loss,
                        "tail_positive": retrieval_loss,
                    }
                reaction_ec_loss = hierarchical_multi_positive_loss(
                    reaction["logits"], reaction_targets_batch
                )
                protein_ec_loss = hierarchical_multi_positive_loss(
                    protein["logits"], protein_targets_batch
                )
                reaction_consistency = hierarchy_consistency_loss(
                    reaction["logits"], model.ec_router.parent_maps()
                )
                protein_consistency = hierarchy_consistency_loss(
                    protein["logits"], model.ec_router.parent_maps()
                )
                positive_edges = positive_mask.nonzero(as_tuple=False)
                paired_protein_logits = [
                    value.index_select(0, positive_edges[:, 1])
                    for value in protein["logits"]
                ]
                cross_modal_loss = cross_modal_ec_alignment_loss(
                    reaction["logits"],
                    paired_protein_logits,
                    positive_edges[:, 0],
                )
                ec_loss = 0.5 * (reaction_ec_loss + protein_ec_loss)
                consistency_loss = 0.5 * (
                    reaction_consistency + protein_consistency
                )
                warmup = (
                    1.0
                    if args.aux_warmup_epochs == 0
                    else min(1.0, (epoch + 1) / args.aux_warmup_epochs)
                )
                loss = retrieval_loss + warmup * (
                    args.ec_loss_weight * ec_loss
                    + args.consistency_loss_weight * consistency_loss
                    + args.cross_modal_ec_weight * cross_modal_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += float(loss.detach())
            totals["retrieval_loss"] += float(retrieval_loss.detach())
            for key, value in retrieval_components.items():
                totals[f"retrieval_{key}_loss"] += float(value.detach())
            totals["ec_loss"] += float(ec_loss.detach())
            totals["consistency_loss"] += float(consistency_loss.detach())
            totals["cross_modal_ec_loss"] += float(cross_modal_loss.detach())
            totals["gradient_norm"] += float(gradient_norm)
            totals["observed_positive_edges_in_pool"] += int(positive_mask.sum())
            totals["candidate_enzymes"] += len(candidate_rows)
            steps += 1

        if steps == 0:
            raise RuntimeError("No training steps were executed")
        scheduler.step()
        scales = {
            key: float(value.detach().cpu())
            for key, value in model.residual_scales().items()
        }
        train_metrics = {
            "epoch": epoch,
            "steps": steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
            "auxiliary_warmup_factor": warmup,
            **{key: value / steps for key, value in totals.items()},
            **{f"residual_scale_{key}": value for key, value in scales.items()},
        }
        append_jsonl(train_log, train_metrics)

        if args.refit_train_val:
            fixed_epoch_metadata = {
                "selection_disabled": True,
                "training_splits": list(training_splits),
                "fixed_epoch_count": args.epochs,
                "completed_epoch_count": epoch + 1,
            }
            atomic_torch_save(
                best_path,
                checkpoint_payload(
                    model,
                    epoch,
                    fixed_epoch_metadata,
                    vocab_sizes,
                    ec_depth,
                    config,
                ),
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "training_mode": "fixed_epoch_train_validation_refit",
                        "checkpoint_selection": "disabled",
                        "residual_scales": scales,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        val_metrics, _ = evaluate_split(
            model,
            "val",
            validation_contract_path,
            graph_store,
            token_store,
            global_store,
            reaction_row,
            val_labels,
            token_to_index,
            device,
            args,
        )
        val_metrics = {"epoch": epoch, **val_metrics, **{
            f"residual_scale_{key}": value for key, value in scales.items()
        }}
        append_jsonl(val_log, val_metrics)
        current_mrr = float(val_metrics["mrr"])
        improved = checkpoint_is_better(
            val_metrics,
            best_metrics,
            initial_metrics,
            mode=args.selection_mode,
            hit1_tolerance=args.selection_hit1_tolerance,
            official_mrr_tolerance=args.selection_official_mrr_tolerance,
            mean_rank_min_improvement=args.selection_mean_rank_min_improvement,
        )
        if improved:
            best_metrics = val_metrics
            best_mrr = current_mrr
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                best_path,
                checkpoint_payload(
                    model, epoch, val_metrics, vocab_sizes, ec_depth, config
                ),
            )
        else:
            epochs_without_improvement += 1
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_mrr": current_mrr,
                    "val_hit@1": val_metrics["hit@1"],
                    "val_hit@10": val_metrics["hit@10"],
                    "val_official_mean_rank": val_metrics[
                        "official_reactzyme_mean_rank"
                    ],
                    "val_official_mrr": val_metrics["official_reactzyme_mrr"],
                    "best_epoch": best_epoch,
                    "best_mrr": best_mrr,
                    "selection_mode": args.selection_mode,
                    "improved_over_anchor": best_epoch >= 0,
                    "residual_scales": scales,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.refit_train_val and epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    selected_scales = {
        key: float(value.detach().cpu()) for key, value in model.residual_scales().items()
    }
    summary: dict[str, Any] = {
        "status": "completed",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selected_residual_scales": selected_scales,
        "anchor_fallback_available": True,
        "ground_truth_ec_is_inference_input": False,
        "test_used_for_selection": False,
        "official_data_gate": "passed",
        "training_splits": list(training_splits),
        "training_mode": (
            "fixed_epoch_train_validation_refit"
            if args.refit_train_val
            else "validation_selected_train_only"
        ),
        "checkpoint_selection": (
            "disabled_fixed_epoch"
            if args.refit_train_val
            else "validation_only"
        ),
    }
    if args.refit_train_val:
        summary.update(
            {
                "fixed_epoch_count": args.epochs,
                "fixed_epoch_metadata": checkpoint["validation_metrics"],
                "learned_residual_selected": True,
            }
        )
    else:
        summary.update(
            {
                "best_epoch": int(checkpoint["epoch"]),
                "best_validation_metrics": checkpoint["validation_metrics"],
                "learned_residual_selected": int(checkpoint["epoch"]) >= 0,
            }
        )
    if not args.skip_final_test:
        heldout_audit = validate_heldout_test_inputs(
            processed,
            test_contract_path,
            set(optimization_pairs["reaction_id"].astype(str)),
        )
        heldout_audit["loaded_after_checkpoint_selection"] = not args.refit_train_val
        heldout_audit["loaded_after_fixed_refit"] = args.refit_train_val
        atomic_json(output_dir / "official_test_data_audit.json", heldout_audit)
        test_label_payload, test_ec_audit = load_official_ec_annotations(
            processed, official_uniprot_source, ["test"]
        )
        atomic_json(
            output_dir / "official_test_ec_diagnostic_audit.json", test_ec_audit
        )
        test_metrics, predictions = evaluate_split(
            model,
            "test",
            test_contract_path,
            graph_store,
            token_store,
            global_store,
            reaction_row,
            test_label_payload.get("test", {}),
            token_to_index,
            device,
            args,
        )
        prediction_path = output_dir / "test_topk.jsonl"
        prediction_path.unlink(missing_ok=True)
        for row in predictions:
            append_jsonl(prediction_path, row)
        summary["test_metrics"] = test_metrics
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
