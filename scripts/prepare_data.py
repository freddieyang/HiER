#!/usr/bin/env python3
"""Prepare the official ReactZyme positive archives and optional frozen-cache views.

This script does not train a model or extract pretrained features. Archive order,
accession identity and the original 10% / seed-42 validation policy are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SPLITS = {"time": "time_split", "seq_smi": "enzyme_smi_split", "mol_smi": "reaction_smi_split"}
REFERENCE_PATH_KEYS = (
    "processed_dir", "global_feature_bundle", "protein_global_embeddings",
    "protein_global_ids", "graph_bundle", "protein_token_manifest",
)
REQUIRED_FEATURE_KEYS = REFERENCE_PATH_KEYS[:4]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path, parent: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def reaction_id(molecules: str) -> str:
    return "RZR_" + hashlib.sha256(molecules.encode()).hexdigest()[:20]


def load_archive(path: Path) -> Mapping:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"Expected a nonempty positive-record mapping: {path}")
    for key, value in payload.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid (molecule_set, sequence) record {key!r}: {path}")
        if not str(value[0]) or not str(value[1]):
            raise ValueError(f"Empty input in record {key!r}: {path}")
    return payload


def accession_lookup(raw: Path) -> dict[tuple[str, str], list[str]]:
    cleaned = pd.read_csv(raw / "cleaned_uniprot_rhea.tsv", sep="\t", dtype=str).fillna("")
    molecules = pd.read_csv(raw / "uniprot_molecules.tsv", sep="\t", dtype=str).fillna("")
    for name, frame, required in (
        ("cleaned_uniprot_rhea.tsv", cleaned, {"Entry", "Sequence", "EC number"}),
        ("uniprot_molecules.tsv", molecules, {"uniprot_id", "molecules"}),
    ):
        if missing := required - set(frame.columns):
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    joined = cleaned[["Entry", "Sequence"]].merge(
        molecules[["uniprot_id", "molecules"]], left_on="Entry", right_on="uniprot_id",
        validate="one_to_one",
    )
    entries: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry, sequence, _, molecule_set in joined.itertuples(index=False, name=None):
        entries[(molecule_set.replace("*", "C"), sequence.upper())].add(entry)
    return {key: sorted(values) for key, values in entries.items()}


def build_split(raw: Path, split: str, entries: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    directory = raw / SPLITS[split]
    archives = {
        "train_val": directory / f"positive_train_val_{split}.pt",
        "test": directory / f"positive_test_{split}.pt",
    }
    train_val, test = (load_archive(archives[phase]) for phase in ("train_val", "test"))
    validation_rows = set(np.random.default_rng(42).permutation(len(train_val))[:round(0.1 * len(train_val))])
    if not validation_rows or len(validation_rows) == len(train_val):
        raise ValueError("The published validation policy needs nonempty train and validation partitions")
    occurrence: Counter = Counter()
    rows, enzyme_rows, reaction_rows, identities = [], {}, {}, {}
    for phase, payload in (("train_val", train_val), ("test", test)):
        phase_identities = set()
        for ordinal, (source_key, value) in enumerate(payload.items()):
            molecules = str(value[0]).replace("*", "C")
            sequence = str(value[1]).replace(" ", "").upper()
            key = (molecules, sequence)
            candidates = entries.get(key, [])
            index = occurrence[key]
            occurrence[key] += 1
            if index >= len(candidates):
                raise ValueError(f"No unambiguous official accession: {split}/{phase}/{source_key}")
            eid, rid = candidates[index], reaction_id(molecules)
            if eid in enzyme_rows and enzyme_rows[eid]["sequence"] != sequence:
                raise ValueError(f"Inconsistent sequence for accession {eid}")
            enzyme_rows[eid] = {"enzyme_id": eid, "uniprot_id": eid, "sequence": sequence}
            if rid in reaction_rows and reaction_rows[rid] != molecules:
                raise ValueError(f"Reaction identity collision: {rid}")
            reaction_rows[rid] = molecules
            membership = "test" if phase == "test" else ("val" if ordinal in validation_rows else "train")
            rows.append({
                "reaction_id": rid, "enzyme_id": eid, "split": membership,
                "source_dataset": "reactzyme", "source_pair_id": f"{archives[phase].stem}:{source_key}",
                "source_archive_ordinal": ordinal, "is_observed_positive": True,
            })
            phase_identities.add(key)
        identities[phase] = phase_identities
    if identities["train_val"] & identities["test"]:
        raise ValueError(f"Exact reaction/sequence pairs cross official train/test: {split}")
    pairs = pd.DataFrame(rows).sort_values(["split", "reaction_id", "enzyme_id"]).reset_index(drop=True)
    enzymes = pd.DataFrame(enzyme_rows.values()).sort_values("enzyme_id").reset_index(drop=True)
    if len(pairs) != len(train_val) + len(test) or pairs.duplicated(["reaction_id", "enzyme_id", "split"]).any():
        raise ValueError(f"Official archive rows cannot be preserved uniquely: {split}")
    metadata = {
        "official_train_val_records": len(train_val), "official_test_records": len(test),
        "source_archives": [str(archives["train_val"]), str(archives["test"])],
    }
    return pairs, enzymes, reaction_rows, metadata


def evaluation_contract(pairs: pd.DataFrame, enzymes: pd.DataFrame, processed: Path, phase: str) -> dict:
    selected = pairs.loc[pairs.split == phase]
    sequences = dict(zip(enzymes.enzyme_id, enzymes.sequence))
    groups: dict[str, list[str]] = defaultdict(list)
    for eid in sorted(set(selected.enzyme_id)):
        groups[sequences[eid]].append(eid)
    representatives = {eid: min(ids) for ids in groups.values() for eid in ids}
    relevance: dict[str, set[str]] = defaultdict(set)
    for rid, eid in selected[["reaction_id", "enzyme_id"]].itertuples(index=False, name=None):
        relevance[rid].add(representatives[eid])
    return {
        "adapter": "ReactZymeAdapter", "direction": "reaction_to_enzyme", "split": phase,
        "copy_only": True, "source_contract": str(processed / "evaluation_contract.json"),
        "query_ids": sorted(relevance), "candidate_ids": sorted(set(representatives.values())),
        "relevance": {rid: sorted(relevance[rid]) for rid in sorted(relevance)},
        "candidate_identity": "exact_full_amino_acid_sequence",
        "source_positive_rows": len(selected), "unique_positive_pairs": sum(map(len, relevance.values())),
        "identical_sequence_alias_count": len(representatives) - len(set(representatives.values())),
    }


def verify_feature_files(reference: dict, unimol: Path, residue_manifest: Path) -> None:
    """Check input identities and metadata without reading the large residue array."""
    from safetensors import safe_open

    source = Path(reference["processed_dir"])
    reactions = pd.read_parquet(source / "reactions.parquet", columns=["reaction_id", "official_molecule_set_smiles"])
    proteins = pd.read_parquet(source / "enzymes.parquet", columns=["enzyme_id", "sequence"])
    ids = read_json(Path(reference["protein_global_ids"]))["enzyme_ids"]
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(proteins.enzyme_id)):
        raise ValueError("Frozen protein IDs must be unique and occur in the source enzyme inventory")
    mean_path = Path(reference["protein_global_embeddings"])
    with safe_open(str(mean_path), framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["embeddings"] or handle.get_slice("embeddings").get_shape() != [len(ids), 1280]:
            raise ValueError("Expected protein mean embeddings of shape [number of IDs, 1280]")
    report = read_json(mean_path.with_name("protein_mean_embeddings.report.json"))
    if report.get("baseline") != "CLIPZyme-ESM-Frozen" or report.get("explicit_negative_pairs_loaded") != 0:
        raise ValueError("Frozen protein feature report has an incompatible source/protocol")
    for key, expected in (("ids_path", reference["protein_global_ids"]), ("output", mean_path)):
        if Path(report[key]).resolve() != Path(expected).resolve():
            raise ValueError(f"Frozen protein report {key} does not match the feature reference")
    if report["enzyme_count"] != len(ids) or not Path(report["source_manifest"]).is_file():
        raise ValueError("Frozen protein report has an invalid row count or missing source manifest")
    bundle_path = Path(reference["global_feature_bundle"])
    manifest = read_json(bundle_path.parent / "manifest.json")
    forbidden = ("pair_or_split_tables_read", "enzyme_associations_read", "validation_or_test_associations_read",
                 "ec_labels_read", "reaction_direction_used")
    if manifest.get("status") != "completed" or any(manifest.get(key) is not False for key in forbidden):
        raise ValueError("Morgan manifest must certify label-independent, direction-free features")
    if (Path(manifest["source_reactions"]).resolve() != source / "reactions.parquet"
            or Path(manifest["feature_bundle"]).resolve() != bundle_path
            or manifest["reaction_count"] != len(reactions)):
        raise ValueError("Morgan manifest paths/count differ from the source inventory")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if (bundle["reaction_ids"] != reactions.reaction_id.tolist()
            or tuple(bundle["reaction_fingerprints"].shape) != (len(reactions), 2048)):
        raise ValueError("Morgan feature rows/shape differ from the reaction inventory")
    forbidden_bundle = ("uses_pair_associations", "uses_split_labels", "uses_enzyme_or_ec_labels", "uses_reaction_direction")
    if any(bundle["feature_contract"].get(key) is not False for key in forbidden_bundle):
        raise ValueError("Morgan bundle has an incompatible feature contract")
    uni = read_json(unimol / "reaction_feature_manifest.json")
    if (uni.get("status") != "completed" or uni.get("primary_output") != "reaction_atomic_mean.pt"
            or uni.get("label_use") != {"reads_pair_labels": False, "reads_test_labels": False}
            or Path(uni["inputs"]["reactions"]["path"]).resolve() != source / "reactions.parquet"):
        raise ValueError("Uni-Mol manifest must match the source reaction inventory and atomic-mean protocol")
    encoded = torch.load(unimol / "reaction_atomic_mean.pt", map_location="cpu", weights_only=True)
    if encoded["entries"] != reactions.reaction_id.tolist() or tuple(encoded["embeddings"].shape) != (len(reactions), 512):
        raise ValueError("Uni-Mol reaction rows/shape differ from the reaction inventory")
    for shard in uni["shards"]:
        if not Path(shard["path"]).is_file():
            raise FileNotFoundError(f"Missing Uni-Mol molecule shard: {shard['path']}")
    cache = read_json(residue_manifest)
    if (cache.get("status") != "completed" or cache.get("mean_reconstruction_checked_rows") != len(ids)
            or cache.get("maximum_residues") != 1022 or cache.get("dtype") != "float16"
            or Path(cache["source_mean"]).resolve() != mean_path):
        raise ValueError("Residue cache must retain the verified 1,022-residue frozen-mean window")
    if read_json(Path(cache["ids_path"]))["enzyme_ids"] != ids:
        raise ValueError("Residue cache and mean-feature protein IDs/order differ")
    contract = cache["feature_contract"]
    if (any(contract.get(key) is not False for key in ("reads_pair_labels", "reads_ec_labels", "reads_split_labels"))
            or contract.get("retains_baseline_residue_window") is not True):
        raise ValueError("Residue cache has an incompatible feature contract")
    offsets = np.load(cache["offsets_path"], allow_pickle=False)
    if (offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 1)
            or np.any(np.diff(offsets) > 1022) or cache["shape"] != [int(offsets[-1]), 1280]
            or Path(cache["cache_path"]).stat().st_size != int(offsets[-1]) * 1280 * 2):
        raise ValueError("Residue cache offsets, shape or byte length are inconsistent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", choices=list(SPLITS), nargs="+", default=list(SPLITS))
    parser.add_argument("--feature-reference", type=Path, help="Data-only JSON referring to existing verified frozen caches")
    parser.add_argument("--unimol-dir", type=Path, help="Default: unimol/ next to the feature-reference JSON")
    parser.add_argument("--residue-cache", type=Path, help="Global residue manifest.json or its directory")
    args = parser.parse_args()
    raw, output = args.raw_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory to avoid replacing processed data: {output}")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits must not contain duplicates")
    reference, source_reactions, source_proteins = None, None, None
    missing = ["--feature-reference and its frozen feature files", "--residue-cache"]
    unimol, residue_manifest = None, None
    if args.feature_reference:
        reference_path = args.feature_reference.resolve()
        original = read_json(reference_path)
        absent = [key for key in REQUIRED_FEATURE_KEYS if not original.get(key)]
        if absent:
            raise ValueError(f"Feature reference is missing required keys: {', '.join(absent)}")
        reference = {key: str(resolve_path(original[key], reference_path.parent))
                     for key in REFERENCE_PATH_KEYS if original.get(key)}
        source = Path(reference["processed_dir"])
        source_reactions = pd.read_parquet(source / "reactions.parquet", columns=["reaction_id", "official_molecule_set_smiles"])
        source_proteins = pd.read_parquet(source / "enzymes.parquet", columns=["enzyme_id", "sequence"])
        for name, frame, key in (("reaction", source_reactions, "reaction_id"), ("enzyme", source_proteins, "enzyme_id")):
            if frame[key].duplicated().any():
                raise ValueError(f"Duplicate {name} IDs in the frozen inventory")
        unimol = args.unimol_dir.resolve() if args.unimol_dir else reference_path.parent / "unimol"
        if args.residue_cache:
            residue_manifest = args.residue_cache.resolve()
            if residue_manifest.is_dir():
                residue_manifest = residue_manifest / "manifest.json"
            verify_feature_files(reference, unimol, residue_manifest)
            missing = []
        else:
            missing = ["--residue-cache (global manifest.json); complete cache validation has not run"]
    elif args.residue_cache or args.unimol_dir:
        raise ValueError("--residue-cache and --unimol-dir require --feature-reference")
    entries = accession_lookup(raw)
    prepared = {split: build_split(raw, split, entries) for split in args.splits}
    inventory: dict[str, str] = {}
    for _, _, reactions, _ in prepared.values():
        inventory.update(reactions)
    if source_reactions is None:
        reactions = pd.DataFrame(sorted(inventory.items()), columns=["reaction_id", "official_molecule_set_smiles"])
    else:
        source_by_id = dict(zip(source_reactions.reaction_id, source_reactions.official_molecule_set_smiles))
        if any(source_by_id.get(rid) != molecules for rid, molecules in inventory.items()):
            raise ValueError("Official reaction inputs are absent from or differ from the frozen reaction inventory")
        reactions = source_reactions
    source_ids, source_row, sequence_by_id, id_by_sequence = [], {}, {}, {}
    if reference:
        source_ids = read_json(Path(reference["protein_global_ids"]))["enzyme_ids"]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Duplicate IDs in the frozen protein cache")
        source_row = {eid: row for row, eid in enumerate(source_ids)}
        sequence_by_id = dict(zip(source_proteins.enzyme_id, source_proteins.sequence))
        for eid in sorted(source_ids):
            id_by_sequence.setdefault(sequence_by_id[eid], eid)
        # Complete every identity check before creating output files.
        for _, enzymes, _, _ in prepared.values():
            cached_ids = []
            for eid, sequence in enzymes[["enzyme_id", "sequence"]].itertuples(index=False, name=None):
                if eid in sequence_by_id and sequence_by_id[eid] != sequence:
                    raise ValueError(f"Frozen source has a different sequence for {eid}")
                cached = eid if eid in source_row else id_by_sequence.get(sequence)
                if cached is None or sequence_by_id[cached] != sequence:
                    raise ValueError(f"No exactly matching frozen sequence feature for {eid}")
                cached_ids.append(cached)
            enzymes["frozen_feature_source_id"] = cached_ids
            enzymes["frozen_feature_source_row"] = [source_row[eid] for eid in cached_ids]
    combined = pd.concat([value[1][["enzyme_id", "uniprot_id", "sequence"]] for value in prepared.values()])
    if combined.groupby("enzyme_id").sequence.nunique().max() > 1:
        raise ValueError("An accession has different sequences across official splits")
    output.mkdir(parents=True)
    if reference:
        write_json(output / "frozen_feature_reference.json", reference)
    # The inventories are also useful when building new caches outside this adapter.
    common = output / "inventory"
    common.mkdir()
    reactions.to_parquet(common / "reactions.parquet", index=False)
    combined.drop_duplicates("enzyme_id").sort_values("enzyme_id").to_parquet(common / "enzymes.parquet", index=False)
    summaries = []
    for split, (pairs, enzymes, _, metadata) in prepared.items():
        processed = output / split
        processed.mkdir()
        pairs.to_parquet(processed / "pairs.parquet", index=False)
        enzymes.to_parquet(processed / "enzymes.parquet", index=False)
        reactions.to_parquet(processed / "reactions.parquet", index=False)
        for phase, filename in (("val", "validation_contract.json"), ("test", "test_contract.json")):
            write_json(processed / filename, evaluation_contract(pairs, enzymes, processed, phase))
        write_json(processed / "evaluation_contract.json", {
            "adapter": "ReactZymeAdapter", "split_type": split, "source_archives": metadata["source_archives"],
            "direction_protocol": "same score matrix transposed; exact sequence and molecule-set identities",
            "metric": "query mean of all-positive ranks and reciprocal ranks; any-positive Hit@K",
        })
        aliases = (enzymes.loc[enzymes.enzyme_id != enzymes.frozen_feature_source_id,
                             ["enzyme_id", "frozen_feature_source_id"]].to_dict("records") if reference else [])
        audit = {
            "status": "passed", "split_type": split, "original_archive_rows": len(pairs),
            "counts": pairs.split.value_counts().to_dict(), **metadata,
            "original_rows_preserved": True, "missing_reactions": 0, "missing_sequences": 0,
            "exact_sequence_checked_rows": len(enzymes), "exact_molecule_checked_rows": len(reactions),
            "feature_aliases": aliases, "feature_identity_checked": reference is not None,
            "exact_pair_train_test_overlap": 0, "official_test_membership_unchanged": True,
            "validation_fraction": 0.1, "validation_seed": 42, "explicit_negative_pairs_loaded": 0,
            "source_feature_reference": str(output / "frozen_feature_reference.json") if reference else None,
            "source_processed_dir": reference["processed_dir"] if reference else None,
            "train_only_statistics_required": True,
        }
        write_json(processed / "identity_audit.json", audit)
        write_json(processed / "ingestion_report.json", {
            "valid": True, "counts": {"pairs": len(pairs), "enzymes": len(enzymes), "reactions": len(reactions)},
            "metadata": {"input_format": "official_positive_pt", **{key: audit[key] for key in (
                "split_type", "official_train_val_records", "official_test_records",
                "validation_fraction", "validation_seed", "explicit_negative_pairs_loaded")}},
        })
        config = {
            **(reference or {}), "processed_dir": str(processed), "benchmark_split": split,
            "official_uniprot_source": str(raw / "cleaned_uniprot_rhea.tsv"),
            "validation_contract": str(processed / "validation_contract.json"),
            "test_contract": str(processed / "test_contract.json"),
            "training_splits": ["train"], "refit_train_val": False,
            "official_split_view": reference is not None,
            "features_ready": not missing, "missing_features": missing,
        }
        if reference:
            write_json(processed / "protein_ids.json", {"enzyme_ids": enzymes.enzyme_id.tolist()})
            np.save(processed / "protein_source_rows.npy", enzymes.frozen_feature_source_row.to_numpy(dtype=np.int64))
            config.update({
                "frozen_feature_reference": str(output / "frozen_feature_reference.json"),
                "protein_row_mapping": str(processed / "protein_source_rows.npy"),
                "protein_view_ids": str(processed / "protein_ids.json"), "source_dir": str(unimol),
            })
            if residue_manifest:
                write_json(processed / "residue_view.json", {
                    "status": "completed", "source_cache_manifest": str(residue_manifest),
                    "source_mean": reference["protein_global_embeddings"],
                    "ids_path": config["protein_view_ids"], "row_mapping_path": config["protein_row_mapping"],
                    "identity_audit": str(processed / "identity_audit.json"), "exact_sequence_match_required": True,
                })
                config["residue_view"] = str(processed / "residue_view.json")
        write_json(processed / "config.json", config)
        summaries.append({"split": split, "counts": audit["counts"], "feature_aliases": len(aliases),
                          "features_ready": config["features_ready"], "config": str(processed / "config.json")})
    write_json(output / "preparation_summary.json", summaries)
    print(json.dumps(summaries, indent=2))
    if missing:
        print("Data/contracts are ready; training is NOT ready. Missing: " + "; ".join(missing))


if __name__ == "__main__":
    main()
