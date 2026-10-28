#!/usr/bin/env python3
"""Train the structured reaction-only-query enzyme retriever on ReactZyme.

The launcher fails closed unless the processed dataset declares the official
positive-only ReactZyme archive and the feature manifests prove that no split,
pair, enzyme label, EC label, or reaction-direction reconstruction entered the
query features.  Only official training positives optimize the model;
validation selects a checkpoint and test is evaluated once afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ec_retrieval import (  # noqa: E402
    ReactionGraphStore,
    StructuredReactionEnzymeRetriever,
    cross_modal_ec_alignment_loss,
    hierarchical_multi_positive_loss,
    hierarchy_consistency_loss,
    multi_positive_infonce,
)


DEFAULT_PROCESSED = PROJECT_ROOT / "data/processed/time"
DEFAULT_GRAPH_BUNDLE = PROJECT_ROOT / "data/features/legacy/reaction_graphs.pt"
DEFAULT_TOKEN_MANIFEST = PROJECT_ROOT / "data/features/legacy/residue_token_manifest.json"
DEFAULT_CONTRACT_ROOT = DEFAULT_PROCESSED
DEFAULT_VALIDATION_CONTRACT = DEFAULT_CONTRACT_ROOT / "validation_contract.json"
DEFAULT_TEST_CONTRACT = DEFAULT_CONTRACT_ROOT / "test_contract.json"
DEFAULT_OFFICIAL_UNIPROT_SOURCE = PROJECT_ROOT / "data/ReactZyme/cleaned_uniprot_rhea.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--graph-bundle", type=Path, default=DEFAULT_GRAPH_BUNDLE)
    parser.add_argument("--protein-token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument(
        "--validation-contract", type=Path, default=DEFAULT_VALIDATION_CONTRACT
    )
    parser.add_argument("--test-contract", type=Path, default=DEFAULT_TEST_CONTRACT)
    parser.add_argument(
        "--official-uniprot-source",
        type=Path,
        default=DEFAULT_OFFICIAL_UNIPROT_SOURCE,
        help=(
            "Frozen UniProt/Rhea table shipped with the official ReactZyme source. "
            "No later Rhea-to-EC snapshot is accepted."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=1024,
        help=(
            "Target number of unique train enzymes in each contrastive denominator. "
            "Observed positives are always retained; 0 disables random expansion."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help=(
            "CUDA autocast dtype. bfloat16 is the stable default for the large "
            "hierarchical heads; float32 disables autocast."
        ),
    )
    parser.add_argument(
        "--symmetric-retrieval-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also optimize enzyme-to-reaction retrieval. Disabled by default because "
            "the benchmark and deployment direction are reaction-to-enzyme."
        ),
    )
    parser.add_argument("--ec-loss-weight", type=float, default=0.10)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.02)
    parser.add_argument("--cross-modal-ec-weight", type=float, default=0.02)
    parser.add_argument("--aux-warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-query-batch-size", type=int, default=64)
    parser.add_argument("--eval-protein-batch-size", type=int, default=2048)
    parser.add_argument("--rank-query-batch-size", type=int, default=128)
    parser.add_argument("--top-k-output", type=int, default=20)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-eval-queries", type=int, default=0)
    parser.add_argument("--max-eval-candidates", type=int, default=0)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument(
        "--warm-graph-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_policy(device: torch.device, name: str) -> tuple[bool, torch.dtype]:
    """Return autocast enablement and dtype for the requested precision policy."""

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported AMP dtype: {name}")
    enabled = device.type == "cuda" and name != "float32"
    if enabled and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested but this CUDA device lacks support")
    return enabled, mapping[name]


def _resolved(path: str | Path) -> str:
    return str(Path(path).resolve())


def validate_official_inputs(
    processed: Path,
    graph_bundle_path: Path,
    token_manifest_path: Path,
    validation_contract_path: Path,
    official_uniprot_source: Path,
) -> dict[str, Any]:
    """Fail closed without loading held-out test pairs or relevance."""

    ingestion_path = processed / "ingestion_report.json"
    split_audit_path = processed / "split_audit.json"
    pairs_path = processed / "pairs.parquet"
    reactions_path = processed / "reactions.parquet"
    for path in (
        ingestion_path,
        split_audit_path,
        validation_contract_path,
        pairs_path,
        reactions_path,
        graph_bundle_path,
        token_manifest_path,
        official_uniprot_source,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))
    metadata = ingestion.get("metadata", {})
    if not ingestion.get("valid"):
        raise ValueError("ReactZyme ingestion report is not valid")
    required_ingestion = {
        "input_format": "official_positive_pt",
        "split_type": "time",
        "explicit_negative_pairs_loaded": 0,
    }
    for key, expected in required_ingestion.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Official ReactZyme gate failed: metadata[{key!r}]="
                f"{metadata.get(key)!r}, expected {expected!r}"
            )
    if int(ingestion.get("counts", {}).get("explicit_negative_pairs_loaded", -1)) != 0:
        raise ValueError("Generated/explicit negative pairs are forbidden")

    pairs = pd.read_parquet(
        pairs_path,
        columns=["reaction_id", "enzyme_id", "split"],
        filters=[("split", "in", ["train", "val"])],
    )
    split_counts = pairs["split"].astype(str).value_counts().to_dict()
    if set(split_counts) != {"train", "val"}:
        raise ValueError(f"Unexpected split labels: {sorted(split_counts)}")
    if split_counts["train"] + split_counts["val"] != int(
        metadata["official_train_val_records"]
    ):
        raise ValueError("Train/validation rows do not match official train-val archive")
    if pairs.duplicated(["reaction_id", "enzyme_id", "split"]).any():
        raise ValueError("Official positive pair table contains duplicate rows")

    contract = json.loads(validation_contract_path.read_text(encoding="utf-8"))
    if contract.get("adapter") != "ReactZymeAdapter":
        raise ValueError("Evaluation contract does not use ReactZymeAdapter")
    if contract.get("direction") != "reaction_to_enzyme":
        raise ValueError("Evaluation contract is not reaction-to-enzyme")
    if contract.get("split") != "val" or contract.get("copy_only") is not True:
        raise ValueError("Validation contract is not the sealed copy-only protocol")
    if _resolved(contract.get("source_contract", "")) != _resolved(
        processed / "evaluation_contract.json"
    ):
        raise ValueError("Validation contract was not derived from this processed dataset")

    graph_manifest_path = graph_bundle_path.parent / "reaction_graph_manifest.json"
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    if graph_manifest.get("status") != "completed":
        raise ValueError("Reaction graph feature build is incomplete")
    forbidden_graph_flags = {
        "pair_or_split_tables_read": False,
        "enzyme_associations_read": False,
        "validation_or_test_associations_read": False,
        "ec_labels_read": False,
        "reaction_direction_used": False,
    }
    for key, expected in forbidden_graph_flags.items():
        if graph_manifest.get(key) is not expected:
            raise ValueError(f"Reaction graph leakage gate failed: {key}")
    if _resolved(graph_manifest["source_reactions"]) != _resolved(reactions_path):
        raise ValueError("Reaction graph source is not this official processed dataset")
    if _resolved(graph_manifest["feature_bundle"]) != _resolved(graph_bundle_path):
        raise ValueError("Reaction graph manifest points to another feature bundle")

    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    if token_manifest.get("status") != "completed":
        raise ValueError("Protein local-token feature build is incomplete")
    token_contract = token_manifest.get("feature_contract", {})
    for key in (
        "reads_pair_associations",
        "reads_split_labels",
        "reads_ec_labels",
        "reads_validation_or_test_associations",
        "reads_reaction_features",
    ):
        if token_contract.get(key) is not False:
            raise ValueError(f"Protein token leakage gate failed: {key}")
    expected_ids_path = (
        processed
        / "external_baselines/clipzyme/protein_mean_embeddings.ids.json"
    )
    if _resolved(token_manifest["canonical_ids"]) != _resolved(expected_ids_path):
        raise ValueError("Protein token IDs are not aligned to the official candidate library")
    if int(token_manifest["processed_enzyme_count"]) != int(
        ingestion["counts"]["enzymes"]
    ):
        raise ValueError("Protein token count differs from official enzyme count")

    official_header = pd.read_csv(
        official_uniprot_source, sep="\t", dtype=str, nrows=1
    ).columns
    required_official_columns = {
        "Entry",
        "EC number",
        "Rhea ID",
        "Date of creation",
        "Sequence",
    }
    if not required_official_columns.issubset(set(official_header)):
        raise ValueError("Frozen official UniProt/Rhea source has an unexpected schema")

    return {
        "status": "passed",
        "dataset": "ReactZyme",
        "split_type": "time",
        "source_input_format": metadata["input_format"],
        "official_train_val_records": int(metadata["official_train_val_records"]),
        "official_test_records_declared_but_not_loaded": int(
            metadata["official_test_records"]
        ),
        "processed_split_counts": {key: int(value) for key, value in split_counts.items()},
        "explicit_negative_pairs_loaded": 0,
        "evaluation_adapter": contract["adapter"],
        "evaluation_direction": contract["direction"],
        "reaction_features_query_only": True,
        "protein_features_label_independent": True,
        "test_pairs_loaded_before_selection": False,
        "test_relevance_loaded_before_selection": False,
        "paths": {
            "ingestion_report": str(ingestion_path),
            "split_audit": str(split_audit_path),
            "validation_contract": str(validation_contract_path),
            "pairs": str(pairs_path),
            "reaction_graph_manifest": str(graph_manifest_path),
            "protein_token_manifest": str(token_manifest_path),
            "official_uniprot_rhea_source": str(official_uniprot_source),
        },
    }


def validate_heldout_test_inputs(
    processed: Path,
    test_contract_path: Path,
    train_reactions: set[str],
) -> dict[str, Any]:
    """Validate and expose test protocol only after checkpoint selection."""

    ingestion = json.loads((processed / "ingestion_report.json").read_text(encoding="utf-8"))
    metadata = ingestion["metadata"]
    test_pairs = pd.read_parquet(
        processed / "pairs.parquet",
        columns=["reaction_id", "enzyme_id", "split"],
        filters=[("split", "==", "test")],
    )
    if set(test_pairs["split"].astype(str)) != {"test"}:
        raise ValueError("Held-out pair loader exposed a non-test split")
    if len(test_pairs) != int(metadata["official_test_records"]):
        raise ValueError("Test rows do not match the official test archive")
    if test_pairs.duplicated(["reaction_id", "enzyme_id", "split"]).any():
        raise ValueError("Official test table contains duplicate positive pairs")

    contract = json.loads(test_contract_path.read_text(encoding="utf-8"))
    if contract.get("adapter") != "ReactZymeAdapter":
        raise ValueError("Test contract does not use ReactZymeAdapter")
    if contract.get("direction") != "reaction_to_enzyme":
        raise ValueError("Test contract is not reaction-to-enzyme")
    if contract.get("split") != "test" or contract.get("copy_only") is not True:
        raise ValueError("Test contract is not the sealed copy-only protocol")
    if _resolved(contract.get("source_contract", "")) != _resolved(
        processed / "evaluation_contract.json"
    ):
        raise ValueError("Test contract was not derived from this processed dataset")
    test_reactions = set(test_pairs["reaction_id"].astype(str))
    overlap = train_reactions & test_reactions
    return {
        "status": "passed",
        "loaded_after_checkpoint_selection": True,
        "official_test_positive_edges": int(len(test_pairs)),
        "official_test_queries": len(test_reactions),
        "sealed_test_contract": str(test_contract_path),
        "train_test_reaction_id_overlap_count": len(overlap),
        "train_test_reaction_id_overlap_fraction_of_test": (
            len(overlap) / len(test_reactions) if test_reactions else 0.0
        ),
        "overlap_interpretation": (
            "property of the released official time split; reported separately and "
            "not introduced by local feature construction"
        ),
    }


def ec_parts(ec: str) -> tuple[str, ...]:
    parts = tuple(str(ec).strip().split("."))
    if not parts or any(not part or part == "-" for part in parts):
        raise ValueError(f"Expected a complete EC number, got {ec!r}")
    return parts


def ec_prefixes(ec: str, depth: int) -> tuple[str, ...]:
    parts = ec_parts(ec)
    if len(parts) != depth:
        raise ValueError(f"Expected EC depth {depth}, got {ec!r}")
    return tuple(".".join(parts[:index]) for index in range(1, depth + 1))


def load_official_ec_annotations(
    processed: Path,
    official_uniprot_source: Path,
    allowed_splits: Sequence[str],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any]]:
    """Derive strict reaction EC labels from the frozen ReactZyme source only.

    The function first filters the official positive pair table to the explicitly
    allowed splits, then joins only those enzymes to the UniProt EC annotations
    shipped with the ReactZyme source.  A reaction is supervised only when all
    observed complete/partial annotations reduce to exactly one complete EC and
    no partial EC remains.  Later Rhea-to-EC snapshots are not read.
    """

    allowed = tuple(map(str, allowed_splits))
    if not allowed:
        return {}, {"allowed_splits": [], "status_counts": {}}
    pairs = pd.read_parquet(
        processed / "pairs.parquet",
        columns=["reaction_id", "enzyme_id", "split"],
        filters=[("split", "in", list(allowed))],
    )
    if not set(pairs["split"].astype(str)).issubset(set(allowed)):
        raise ValueError("Official EC loader exposed an unrequested split")
    enzymes = pd.read_parquet(
        processed / "enzymes.parquet", columns=["enzyme_id", "uniprot_id"]
    )
    enzyme_to_uniprot = {
        str(enzyme_id): str(uniprot_id)
        for enzyme_id, uniprot_id in enzymes.itertuples(index=False, name=None)
    }
    needed_uniprot = {
        enzyme_to_uniprot[str(enzyme_id)]
        for enzyme_id in pairs["enzyme_id"].astype(str)
    }
    raw = pd.read_csv(
        official_uniprot_source,
        sep="\t",
        dtype=str,
        usecols=["Entry", "EC number"],
    )
    raw = raw.loc[raw["Entry"].astype(str).isin(needed_uniprot)]
    if raw["Entry"].duplicated().any():
        raise ValueError("Frozen official UniProt source contains duplicate entries")
    ec_by_uniprot = {
        str(entry): str(value)
        for entry, value in raw.itertuples(index=False, name=None)
        if pd.notna(value)
    }
    missing_uniprot = needed_uniprot - set(ec_by_uniprot)
    if missing_uniprot:
        raise ValueError(
            f"Official UniProt EC annotations miss {len(missing_uniprot)} selected enzymes"
        )

    output: dict[str, dict[str, set[str]]] = defaultdict(dict)
    status_by_split: dict[str, dict[str, int]] = {}
    for split, split_pairs in pairs.groupby("split", sort=True):
        status_counts: defaultdict[str, int] = defaultdict(int)
        for reaction_id, group in split_pairs.groupby("reaction_id", sort=True):
            complete: set[str] = set()
            partial: set[str] = set()
            for enzyme_id in group["enzyme_id"].astype(str):
                annotation = ec_by_uniprot[enzyme_to_uniprot[enzyme_id]]
                for token in (value.strip() for value in annotation.split(";")):
                    if not token:
                        continue
                    parts = token.split(".")
                    if len(parts) == 4 and all(parts) and "-" not in parts:
                        complete.add(token)
                    else:
                        partial.add(token)
            if len(complete) == 1 and not partial:
                status = "exact_one"
                output[str(split)][str(reaction_id)] = set(complete)
            elif len(complete) == 1:
                status = "exact_one_with_partial_quarantined"
            elif len(complete) > 1:
                status = "exact_multi_quarantined"
            elif partial:
                status = "partial_only_quarantined"
            else:
                status = "missing_quarantined"
            status_counts[status] += 1
        status_by_split[str(split)] = dict(status_counts)
    audit = {
        "source": str(official_uniprot_source),
        "source_semantics": "frozen UniProt/Rhea table shipped with ReactZyme",
        "later_rhea_to_ec_snapshot_read": False,
        "allowed_splits": list(allowed),
        "selected_positive_pairs": int(len(pairs)),
        "selected_uniprot_entries": len(needed_uniprot),
        "status_counts_by_split": status_by_split,
        "strict_policy": (
            "exactly one complete four-component EC across observed positive enzymes; "
            "partial, multi-label, and missing cases quarantined"
        ),
    }
    return {split: dict(values) for split, values in output.items()}, audit


def infer_ec_depth(training_labels: dict[str, set[str]]) -> int:
    depths = {
        len(ec_parts(ec)) for values in training_labels.values() for ec in values
    }
    if len(depths) != 1:
        raise ValueError(f"Training EC labels have inconsistent depths: {sorted(depths)}")
    depth = next(iter(depths))
    if depth <= 0:
        raise ValueError("Training EC labels are empty")
    return depth


def build_ec_vocab(
    training_labels: dict[str, set[str]], depth: int
) -> tuple[list[list[str]], list[dict[str, int]], list[torch.Tensor]]:
    values_by_level = [set() for _ in range(depth)]
    for ecs in training_labels.values():
        for ec in ecs:
            for level, prefix in enumerate(ec_prefixes(ec, depth)):
                values_by_level[level].add(prefix)
    vocabularies = [sorted(values) for values in values_by_level]
    if any(not values for values in vocabularies):
        raise ValueError("At least one EC vocabulary level is empty")
    token_to_index = [
        {token: index for index, token in enumerate(values)}
        for values in vocabularies
    ]
    child_to_parent: list[torch.Tensor] = []
    for level in range(1, depth):
        parent_ids: list[int] = []
        for child in vocabularies[level]:
            parent = ".".join(child.split(".")[:-1])
            if parent not in token_to_index[level - 1]:
                raise ValueError(f"Missing training parent {parent!r} for child {child!r}")
            parent_ids.append(token_to_index[level - 1][parent])
        child_to_parent.append(torch.tensor(parent_ids, dtype=torch.long))
    return vocabularies, token_to_index, child_to_parent


def target_tokens_for_ids(
    ids: Sequence[str],
    labels: dict[str, set[str]],
    token_to_index: list[dict[str, int]],
) -> list[list[tuple[int, ...]]]:
    depth = len(token_to_index)
    output: list[list[tuple[int, ...]]] = [[] for _ in range(depth)]
    for value in ids:
        ecs = labels.get(str(value), set())
        prefixes = [ec_prefixes(ec, depth) for ec in ecs]
        for level, mapping in enumerate(token_to_index):
            target = {
                mapping[path[level]] for path in prefixes if path[level] in mapping
            }
            output[level].append(tuple(sorted(target)))
    return output


def protein_target_tokens(
    protein_ids: Sequence[str],
    train_pairs: pd.DataFrame,
    train_labels: dict[str, set[str]],
    token_to_index: list[dict[str, int]],
) -> list[list[tuple[int, ...]]]:
    depth = len(token_to_index)
    protein_row = {value: index for index, value in enumerate(protein_ids)}
    targets: list[list[set[int]]] = [
        [set() for _ in protein_ids] for _ in range(depth)
    ]
    for reaction_id, enzyme_id in train_pairs[
        ["reaction_id", "enzyme_id"]
    ].itertuples(index=False, name=None):
        row = protein_row[str(enzyme_id)]
        for ec in train_labels.get(str(reaction_id), set()):
            path = ec_prefixes(ec, depth)
            for level, mapping in enumerate(token_to_index):
                if path[level] in mapping:
                    targets[level][row].add(mapping[path[level]])
    return [
        [tuple(sorted(values)) for values in level_targets]
        for level_targets in targets
    ]


def dense_target_masks(
    selected_rows: Sequence[int] | torch.Tensor,
    targets_by_level: list[list[tuple[int, ...]]],
    vocab_sizes: Sequence[int],
    device: torch.device,
) -> list[torch.Tensor]:
    rows = (
        [int(value) for value in selected_rows.detach().cpu().tolist()]
        if isinstance(selected_rows, torch.Tensor)
        else [int(value) for value in selected_rows]
    )
    masks: list[torch.Tensor] = []
    for level, vocab_size in enumerate(vocab_sizes):
        mask = torch.zeros((len(rows), int(vocab_size)), dtype=torch.bool, device=device)
        target_rows: list[int] = []
        target_columns: list[int] = []
        for batch_row, source_row in enumerate(rows):
            for token_id in targets_by_level[level][source_row]:
                target_rows.append(batch_row)
                target_columns.append(token_id)
        if target_rows:
            mask[
                torch.tensor(target_rows, device=device),
                torch.tensor(target_columns, device=device),
            ] = True
        masks.append(mask)
    return masks


class ProteinTokenStore:
    """Read fixed local protein tokens from a label-independent memory map."""

    def __init__(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError("Protein token manifest is not complete")
        self.manifest = manifest
        self.shape = tuple(map(int, manifest["shape"]))
        self.tokens = np.memmap(
            manifest["token_path"], mode="r", dtype=np.float16, shape=self.shape
        )
        self.mask = np.memmap(
            manifest["mask_path"],
            mode="r",
            dtype=np.uint8,
            shape=self.shape[:2],
        )
        ids_payload = json.loads(Path(manifest["canonical_ids"]).read_text(encoding="utf-8"))
        self.ids = list(map(str, ids_payload["enzyme_ids"]))
        if len(self.ids) != self.shape[0]:
            raise ValueError("Protein token and ID row counts differ")
        self.row = {value: index for index, value in enumerate(self.ids)}

    def get(self, rows: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = (
            np.asarray(rows.detach().cpu(), dtype=np.int64)
            if isinstance(rows, torch.Tensor)
            else np.asarray(list(rows), dtype=np.int64)
        )
        token_rows = np.asarray(self.tokens[indices]).copy()
        mask_rows = np.asarray(self.mask[indices]).copy().astype(bool)
        return torch.from_numpy(token_rows), torch.from_numpy(mask_rows)


def build_train_adjacency(
    train_pairs: pd.DataFrame,
    reaction_row: dict[str, int],
    protein_row: dict[str, int],
) -> dict[int, tuple[int, ...]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for reaction_id, enzyme_id in train_pairs[
        ["reaction_id", "enzyme_id"]
    ].itertuples(index=False, name=None):
        adjacency[reaction_row[str(reaction_id)]].add(protein_row[str(enzyme_id)])
    if not adjacency or any(not values for values in adjacency.values()):
        raise ValueError("Training adjacency contains an empty query")
    return {row: tuple(sorted(values)) for row, values in adjacency.items()}


def sample_query_balanced_batch(
    query_rows: Sequence[int],
    adjacency: dict[int, tuple[int, ...]],
    positives_per_query: int,
    rng: np.random.Generator,
    candidate_universe: Sequence[int] | np.ndarray | None = None,
    target_candidate_count: int = 0,
    negative_rng: np.random.Generator | None = None,
    positive_exposure_counts: dict[int, np.ndarray] | None = None,
    hard_candidate_bank: dict[int, tuple[int, ...]] | None = None,
    hard_candidates_per_query: int = 0,
    hard_candidate_rng: np.random.Generator | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Build a query-balanced positive set plus train-only denominator candidates.

    When ``positive_exposure_counts`` is supplied, each query selects its
    least-exposed positives first, using ``rng`` only to break ties.  This gives
    every known train association coverage before repeatedly sampling easy or
    lucky positives.  A query-specific bank may inject globally confusing,
    train-only denominator candidates before random pool filling.  The output
    mask still marks every known association that happens to enter the shared
    candidate pool.
    """

    selected_candidates: set[int] = set()
    for query_row in query_rows:
        positives = adjacency[int(query_row)]
        count = min(positives_per_query, len(positives))
        positive_array = np.asarray(positives, dtype=np.int64)
        if positive_exposure_counts is None:
            chosen = rng.choice(positive_array, size=count, replace=False)
        else:
            exposure = positive_exposure_counts.setdefault(
                int(query_row), np.zeros(len(positives), dtype=np.int64)
            )
            if exposure.shape != (len(positives),):
                raise ValueError("Positive exposure state does not match adjacency")
            tie_breaker = rng.random(len(positives))
            chosen_indices = np.lexsort((tie_breaker, exposure))[:count]
            exposure[chosen_indices] += 1
            chosen = positive_array[chosen_indices]
        selected_candidates.update(map(int, np.atleast_1d(chosen)))
    if hard_candidates_per_query < 0:
        raise ValueError("hard_candidates_per_query must be non-negative")
    if hard_candidates_per_query and hard_candidate_bank is None:
        raise ValueError(
            "hard_candidate_bank is required when hard candidates are requested"
        )
    if hard_candidate_bank is not None and hard_candidates_per_query:
        bank_rng = hard_candidate_rng if hard_candidate_rng is not None else rng
        proposed_hard_candidates: set[int] = set()
        for query_row in query_rows:
            bank = np.asarray(
                hard_candidate_bank.get(int(query_row), ()), dtype=np.int64
            )
            if bank.size == 0:
                continue
            count = min(hard_candidates_per_query, int(bank.size))
            chosen = bank_rng.choice(bank, size=count, replace=False)
            proposed_hard_candidates.update(map(int, np.atleast_1d(chosen)))
        proposed_hard_candidates.difference_update(selected_candidates)
        if target_candidate_count > 0:
            remaining_slots = max(
                target_candidate_count - len(selected_candidates), 0
            )
            if len(proposed_hard_candidates) > remaining_slots:
                proposed_array = np.asarray(
                    sorted(proposed_hard_candidates), dtype=np.int64
                )
                proposed_hard_candidates = set(
                    map(
                        int,
                        bank_rng.choice(
                            proposed_array,
                            size=remaining_slots,
                            replace=False,
                        ),
                    )
                )
        selected_candidates.update(proposed_hard_candidates)
    if target_candidate_count > len(selected_candidates):
        if candidate_universe is None:
            raise ValueError("candidate_universe is required for pool expansion")
        universe = np.asarray(candidate_universe, dtype=np.int64)
        selected_array = np.fromiter(
            selected_candidates, dtype=np.int64, count=len(selected_candidates)
        )
        available = universe[
            ~np.isin(universe, selected_array, assume_unique=True)
        ]
        extra_count = min(
            target_candidate_count - len(selected_candidates), len(available)
        )
        if extra_count:
            pool_rng = negative_rng if negative_rng is not None else rng
            extra = pool_rng.choice(available, size=extra_count, replace=False)
            selected_candidates.update(map(int, np.atleast_1d(extra)))
    candidate_rows = sorted(selected_candidates)
    candidate_column = {row: index for index, row in enumerate(candidate_rows)}
    positive_mask = torch.zeros(
        (len(query_rows), len(candidate_rows)), dtype=torch.bool
    )
    for query_index, query_row in enumerate(query_rows):
        for protein in adjacency[int(query_row)]:
            column = candidate_column.get(protein)
            if column is not None:
                positive_mask[query_index, column] = True
    if bool((~positive_mask.any(dim=1)).any()):
        raise RuntimeError("Balanced batch contains a query without an observed positive")
    return candidate_rows, positive_mask


def load_contract(
    path: Path,
    split: str,
    max_queries: int = 0,
    max_candidates: int = 0,
) -> tuple[list[str], list[str], dict[str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != split or payload.get("copy_only") is not True:
        raise ValueError(f"Expected sealed {split!r} evaluation contract")
    query_ids = list(map(str, payload["query_ids"]))
    candidate_ids = list(map(str, payload["candidate_ids"]))
    relevance = {
        str(query): set(map(str, enzymes))
        for query, enzymes in payload["relevance"].items()
    }
    if max_queries > 0:
        query_ids = query_ids[:max_queries]
    if max_candidates > 0:
        required = set().union(*(relevance[query] for query in query_ids))
        selected = list(sorted(required))
        selected_set = set(selected)
        for candidate in candidate_ids:
            if len(selected) >= max_candidates:
                break
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        candidate_ids = sorted(selected)
    candidate_set = set(candidate_ids)
    relevance = {query: relevance[query] & candidate_set for query in query_ids}
    query_ids = [query for query in query_ids if relevance[query]]
    return query_ids, candidate_ids, relevance


@torch.inference_mode()
def encode_queries(
    model: StructuredReactionEnzymeRetriever,
    graph_store: ReactionGraphStore,
    reaction_rows: Sequence[int],
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.eval()
    amp_enabled, autocast_dtype = amp_policy(device, amp_dtype)
    vectors: list[torch.Tensor] = []
    predictions: list[list[torch.Tensor]] = [
        [] for _ in range(model.ec_router.depth)
    ]
    for start in range(0, len(reaction_rows), batch_size):
        rows = reaction_rows[start : start + batch_size]
        graph_batch = graph_store.collate(rows).to(device)
        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
        ):
            encoded = model.encode_reaction(graph_batch)
        vectors.append(encoded["embedding"].float().cpu())
        for level, value in enumerate(encoded["logits"]):
            predictions[level].append(value.float().argmax(dim=1).cpu())
    return torch.cat(vectors), [torch.cat(values) for values in predictions]


@torch.inference_mode()
def encode_candidates(
    model: StructuredReactionEnzymeRetriever,
    token_store: ProteinTokenStore,
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
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled
        ):
            encoded = model.encode_protein(tokens, mask)
        vectors.append(encoded["embedding"].float().cpu())
    return torch.cat(vectors)


def ec_diagnostics(
    query_ids: Sequence[str],
    top1_by_level: Sequence[torch.Tensor],
    split_labels: dict[str, set[str]],
    token_to_index: list[dict[str, int]],
) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    depth = len(token_to_index)
    for level in range(depth):
        supported = 0
        hits = 0
        for row, reaction_id in enumerate(query_ids):
            targets = {
                token_to_index[level][path[level]]
                for ec in split_labels.get(str(reaction_id), set())
                for path in [ec_prefixes(ec, depth)]
                if path[level] in token_to_index[level]
            }
            if not targets:
                continue
            supported += 1
            hits += int(int(top1_by_level[level][row]) in targets)
        output[f"ec_l{level + 1}_supported_queries"] = supported
        output[f"ec_l{level + 1}_supported_fraction"] = (
            supported / len(query_ids) if query_ids else 0.0
        )
        output[f"ec_l{level + 1}_hit1_supported"] = hits / supported if supported else 0.0
    return output


def rank_metrics(
    query_vectors: torch.Tensor,
    candidate_vectors: torch.Tensor,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    relevance: dict[str, set[str]],
    device: torch.device,
    query_batch_size: int,
    top_k_output: int,
    score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    candidate_vectors_device = candidate_vectors.to(device)
    candidate_row = {value: index for index, value in enumerate(candidate_ids)}
    ks = (1, 5, 10, 20, 50, 100)
    totals = defaultdict(float)
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(query_ids), query_batch_size):
        stop = min(start + query_batch_size, len(query_ids))
        query_batch = query_vectors[start:stop].to(device)
        scores = (
            score_function(query_batch, candidate_vectors_device)
            if score_function is not None
            else query_batch @ candidate_vectors_device.T
        )
        if scores.shape != (stop - start, len(candidate_ids)):
            raise ValueError("score_function returned an invalid score matrix")
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        inverse = torch.empty_like(order)
        inverse.scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=device).unsqueeze(0).expand_as(order),
        )
        top_indices = order[:, : min(top_k_output, order.shape[1])].cpu()
        inverse = inverse.cpu()
        for local_row, query_id in enumerate(query_ids[start:stop]):
            positive_rows = sorted(
                candidate_row[value]
                for value in relevance[str(query_id)]
                if value in candidate_row
            )
            ranks = sorted(int(inverse[local_row, row]) + 1 for row in positive_rows)
            if not ranks:
                continue
            totals["mrr"] += 1.0 / ranks[0]
            totals["mean_first_positive_rank"] += ranks[0]
            totals["official_reactzyme_mrr_all_relevant"] += sum(
                1.0 / rank for rank in ranks
            ) / len(ranks)
            totals["official_reactzyme_mean_relevant_rank"] += sum(ranks) / len(ranks)
            for k in ks:
                hits = sum(rank <= k for rank in ranks)
                totals[f"hit@{k}"] += float(hits > 0)
                totals[f"recall@{k}"] += hits / len(ranks)
                totals[f"official_reactzyme_acc_n_at_{k}"] += hits / k
                totals[f"map@{k}"] += sum(
                    index / rank
                    for index, rank in enumerate(ranks, start=1)
                    if rank <= k
                ) / min(len(ranks), k)
                dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= k)
                idcg = sum(
                    1.0 / math.log2(rank + 1)
                    for rank in range(1, min(len(ranks), k) + 1)
                )
                totals[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0
            predictions.append(
                {
                    "query_id": str(query_id),
                    "positive_enzyme_ids": sorted(relevance[str(query_id)]),
                    "positive_ranks": ranks,
                    "first_positive_rank": ranks[0],
                    "top_enzyme_ids": [
                        candidate_ids[int(index)] for index in top_indices[local_row]
                    ],
                }
            )
    count = len(predictions)
    if count == 0:
        raise ValueError("No evaluable query")
    metrics: dict[str, float | int] = {
        key: value / count for key, value in totals.items()
    }
    metrics["official_reactzyme_mean_rank"] = metrics[
        "official_reactzyme_mean_relevant_rank"
    ]
    metrics["official_reactzyme_mrr"] = metrics[
        "official_reactzyme_mrr_all_relevant"
    ]
    metrics["query_count"] = count
    metrics["candidate_count"] = len(candidate_ids)
    metrics["median_first_positive_rank"] = float(
        np.median([row["first_positive_rank"] for row in predictions])
    )
    return metrics, predictions


@torch.inference_mode()
def evaluate_split(
    model: StructuredReactionEnzymeRetriever,
    split: str,
    contract_path: Path,
    graph_store: ReactionGraphStore,
    token_store: ProteinTokenStore,
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
            f"Official evaluation features missing queries={missing_queries[:3]}, "
            f"candidates={missing_candidates[:3]}"
        )
    query_vectors, top1 = encode_queries(
        model,
        graph_store,
        [reaction_row[value] for value in query_ids],
        device,
        args.eval_query_batch_size,
        args.amp_dtype,
    )
    candidate_vectors = encode_candidates(
        model,
        token_store,
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
    )
    metrics.update(ec_diagnostics(query_ids, top1, split_labels, token_to_index))
    return metrics, predictions


def main() -> None:
    args = parse_args()
    positive_ints = (
        args.epochs,
        args.patience,
        args.query_batch_size,
        args.positives_per_query,
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
            args.aux_warmup_epochs,
            args.candidate_pool_size,
        )
    ):
        raise ValueError("Loss weights and warm-up must be non-negative")
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
    reaction_row = {
        value: index for index, value in enumerate(graph_store.reaction_ids)
    }
    token_store = ProteinTokenStore(token_manifest_path)
    pairs_path = processed / "pairs.parquet"
    train_pairs = pd.read_parquet(
        pairs_path,
        columns=["reaction_id", "enzyme_id", "split"],
        filters=[("split", "==", "train")],
    )
    if set(train_pairs["split"].astype(str)) != {"train"}:
        raise ValueError("Training pair loader exposed a non-train split")

    missing_reactions = sorted(set(train_pairs["reaction_id"].astype(str)) - set(reaction_row))
    missing_proteins = sorted(set(train_pairs["enzyme_id"].astype(str)) - set(token_store.row))
    if missing_reactions or missing_proteins:
        raise ValueError(
            f"Training features missing reactions={missing_reactions[:3]}, "
            f"proteins={missing_proteins[:3]}"
        )

    # Test pairs, test relevance, and test EC annotations are not opened here.
    train_val_labels, train_val_ec_audit = load_official_ec_annotations(
        processed, official_uniprot_source, ["train", "val"]
    )
    atomic_json(output_dir / "official_train_val_ec_supervision_audit.json", train_val_ec_audit)
    train_labels = train_val_labels.get("train", {})
    val_labels = train_val_labels.get("val", {})
    ec_depth = infer_ec_depth(train_labels)
    vocabularies, token_to_index, child_to_parent = build_ec_vocab(
        train_labels, ec_depth
    )
    vocab_sizes = [len(values) for values in vocabularies]
    reaction_targets = target_tokens_for_ids(
        graph_store.reaction_ids, train_labels, token_to_index
    )
    protein_targets = protein_target_tokens(
        token_store.ids, train_pairs, train_labels, token_to_index
    )
    adjacency = build_train_adjacency(
        train_pairs, reaction_row, token_store.row
    )
    train_candidate_rows = np.asarray(
        sorted({token_store.row[str(value)] for value in train_pairs["enzyme_id"]}),
        dtype=np.int64,
    )
    train_query_rows = sorted(adjacency)
    if args.max_train_queries > 0:
        train_query_rows = train_query_rows[: args.max_train_queries]
    if args.warm_graph_cache:
        graph_store.warm(train_query_rows)

    model = StructuredReactionEnzymeRetriever(
        atom_feature_dim=graph_store.atom_feature_dim,
        edge_feature_dim=graph_store.edge_feature_dim,
        motif_feature_dim=graph_store.motif_feature_dim,
        ec_vocab_sizes=vocab_sizes,
        child_to_parent=child_to_parent,
        protein_input_dim=int(token_store.shape[2]),
        protein_segments=int(token_store.shape[1]),
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.05,
    )
    # BF16 has FP32-like exponent range and does not use dynamic loss scaling.
    # GradScaler remains available for the explicit FP16 ablation only.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16"
    )

    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "task_definition": "reaction_query_to_enzyme_retrieval",
            "inference_inputs": ["official_reaction_molecule_inventory"],
            "ground_truth_ec_is_model_input": False,
            "ec_depth": ec_depth,
            "ec_depth_policy": "auto_detect_complete_depth_from_train_exact_one_only",
            "selection_metric": "validation_mrr_first_relevant",
            "test_used_for_selection": False,
            "candidate_indexable": True,
            "official_positive_pairs_only": True,
            "query_positive_sampling_seed": args.seed,
            "negative_pool_sampling_seed": args.seed + 1_000_003,
            "external_rhea_to_ec_snapshot_used": False,
            "ec_supervision_source": "frozen_reactzyme_cleaned_uniprot_rhea_tsv",
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
            "test_relevance_loaded_during_vocabulary_build": False,
        },
    )
    data_audit = {
        "official_gate": "passed",
        "train_positive_edges": int(len(train_pairs)),
        "train_queries": len(adjacency),
        "train_enzymes": int(train_pairs["enzyme_id"].nunique()),
        "strict_ec_labeled_train_reactions": len(train_labels),
        "ec_depth": ec_depth,
        "ec_vocab_sizes": vocab_sizes,
        "sampling_unit": "unique_train_reaction_query",
        "positives_per_query": args.positives_per_query,
        "candidate_pool_size": args.candidate_pool_size,
        "candidate_pool_source": "train_enzyme_ids_only",
        "query_positive_and_negative_pool_rngs_are_independent": True,
        "all_known_in_pool_positives_marked": True,
        "unobserved_pair_policy": (
            "contrastive denominator only; never persisted or claimed as a verified negative"
        ),
        "reaction_feature_policy": "query-only official molecule inventory",
        "protein_feature_policy": "frozen ESM-2 residue embeddings without labels",
        "test_ec_loaded_before_selection": False,
        "test_pairs_loaded_before_selection": False,
        "test_relevance_loaded_before_selection": False,
        "external_rhea_to_ec_snapshot_used": False,
    }
    atomic_json(output_dir / "data_audit.json", data_audit)
    atomic_json(
        output_dir / "provenance.json",
        {
            "processed_dataset": str(processed),
            "pairs": str(pairs_path),
            "validation_contract": str(validation_contract_path),
            "test_contract_sealed_until_checkpoint_selection": str(test_contract_path),
            "reaction_graph_bundle": str(graph_bundle_path),
            "protein_token_manifest": str(token_manifest_path),
            "official_uniprot_rhea_source": str(official_uniprot_source),
            "later_rhea_to_ec_snapshot_used": False,
            "official_ingestion_report": str(processed / "ingestion_report.json"),
            "official_split_audit": str(processed / "split_audit.json"),
            "official_preprocessing_hashes": str(processed / "preprocessing_hashes.json"),
        },
    )

    train_log = output_dir / "train_metrics.jsonl"
    val_log = output_dir / "validation_metrics.jsonl"
    train_log.unlink(missing_ok=True)
    val_log.unlink(missing_ok=True)
    best_path = output_dir / "best.pt"
    best_mrr = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    rng = np.random.default_rng(args.seed)
    negative_rng = np.random.default_rng(args.seed + 1_000_003)

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        query_permutation = rng.permutation(np.asarray(train_query_rows, dtype=np.int64))
        totals = defaultdict(float)
        steps = 0
        for start in range(0, len(query_permutation), args.query_batch_size):
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
            query_rows = list(
                map(int, query_permutation[start : start + args.query_batch_size])
            )
            candidate_rows, positive_mask_cpu = sample_query_balanced_batch(
                query_rows,
                adjacency,
                args.positives_per_query,
                rng,
                candidate_universe=train_candidate_rows,
                target_candidate_count=args.candidate_pool_size,
                negative_rng=negative_rng,
            )
            graph_batch = graph_store.collate(query_rows).to(device)
            protein_tokens, protein_mask = token_store.get(candidate_rows)
            protein_tokens = protein_tokens.to(device, non_blocking=True)
            protein_mask = protein_mask.to(device, non_blocking=True)
            positive_mask = positive_mask_cpu.to(device)
            reaction_ec_targets = dense_target_masks(
                query_rows, reaction_targets, vocab_sizes, device
            )
            protein_ec_targets = dense_target_masks(
                candidate_rows, protein_targets, vocab_sizes, device
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                scores, reaction, protein = model(
                    graph_batch, protein_tokens, protein_mask
                )
                retrieval_loss = multi_positive_infonce(
                    scores,
                    positive_mask,
                    symmetric=args.symmetric_retrieval_loss,
                )
                reaction_ec_loss = hierarchical_multi_positive_loss(
                    reaction["logits"], reaction_ec_targets
                )
                protein_ec_loss = hierarchical_multi_positive_loss(
                    protein["logits"], protein_ec_targets
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
                model.parameters(),
                args.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += float(loss.detach())
            totals["retrieval_loss"] += float(retrieval_loss.detach())
            totals["ec_loss"] += float(ec_loss.detach())
            totals["consistency_loss"] += float(consistency_loss.detach())
            totals["cross_modal_ec_loss"] += float(cross_modal_loss.detach())
            totals["gradient_norm"] += float(gradient_norm)
            totals["queries"] += len(query_rows)
            totals["candidate_enzymes"] += len(candidate_rows)
            totals["negative_only_candidate_columns"] += int(
                (~positive_mask.any(dim=0)).sum()
            )
            totals["observed_positive_edges_in_pool"] += int(positive_mask.sum())
            totals["ec_supervised_queries"] += int(
                reaction_ec_targets[-1].any(dim=1).sum()
            )
            totals["ec_supervised_candidates"] += int(
                protein_ec_targets[-1].any(dim=1).sum()
            )
            steps += 1

        if steps == 0:
            raise RuntimeError("No training steps were executed")
        scheduler.step()
        train_metrics = {
            "epoch": epoch,
            "steps": steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "auxiliary_warmup_factor": (
                1.0
                if args.aux_warmup_epochs == 0
                else min(1.0, (epoch + 1) / args.aux_warmup_epochs)
            ),
            "seconds": time.perf_counter() - epoch_start,
            **{key: value / steps for key, value in totals.items()},
        }
        append_jsonl(train_log, train_metrics)

        val_metrics, _ = evaluate_split(
            model,
            "val",
            validation_contract_path,
            graph_store,
            token_store,
            reaction_row,
            val_labels,
            token_to_index,
            device,
            args,
        )
        val_metrics = {"epoch": epoch, **val_metrics}
        append_jsonl(val_log, val_metrics)
        current_mrr = float(val_metrics["mrr"])
        improved = current_mrr > best_mrr + 1e-12
        if improved:
            best_mrr = current_mrr
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "ec_vocab_sizes": vocab_sizes,
                "ec_depth": ec_depth,
                "config": config,
            }
            temporary = best_path.with_name(f".{best_path.name}.tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, best_path)
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
                    "best_epoch": best_epoch,
                    "best_mrr": best_mrr,
                    "improved": improved,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    summary: dict[str, Any] = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "ground_truth_ec_is_inference_input": False,
        "test_used_for_selection": False,
        "official_data_gate": "passed",
        "test_annotations_loaded_before_checkpoint_selection": False,
        "test_pairs_loaded_before_checkpoint_selection": False,
        "test_relevance_loaded_before_checkpoint_selection": False,
    }
    if not args.skip_final_test:
        # Held-out EC labels are loaded only now, after the checkpoint is fixed,
        # and are used solely for auxiliary diagnostics, never for retrieval rank.
        heldout_audit = validate_heldout_test_inputs(
            processed,
            test_contract_path,
            set(train_pairs["reaction_id"].astype(str)),
        )
        atomic_json(output_dir / "official_test_data_audit.json", heldout_audit)
        test_label_payload, test_ec_audit = load_official_ec_annotations(
            processed, official_uniprot_source, ["test"]
        )
        atomic_json(output_dir / "official_test_ec_diagnostic_audit.json", test_ec_audit)
        test_labels = test_label_payload.get("test", {})
        test_metrics, test_predictions = evaluate_split(
            model,
            "test",
            test_contract_path,
            graph_store,
            token_store,
            reaction_row,
            test_labels,
            token_to_index,
            device,
            args,
        )
        summary["test_metrics"] = test_metrics
        prediction_path = output_dir / "test_topk.jsonl"
        prediction_path.unlink(missing_ok=True)
        for row in test_predictions:
            append_jsonl(prediction_path, row)
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
