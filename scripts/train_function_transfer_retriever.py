#!/usr/bin/env python3
"""Train the independent Uni-Mol/ESM expert on official train only.

There is deliberately no test loader or train+validation refit option. The
reference run supplies audited feature paths and comparison metrics, not weights.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ec_retrieval.function_transfer import FunctionTransferRetriever
from ec_retrieval import (
    multi_positive_tail_risk_loss, hierarchical_multi_positive_loss,
    hierarchy_consistency_loss, cross_modal_ec_alignment_loss,
)
from scripts.train_global_anchored_retriever import (
    atomic_torch_save, validate_global_feature_provenance, validate_train_only_protocol,
)
from scripts.train_structured_retriever import (
    atomic_json, append_jsonl, seed_everything, amp_policy, load_contract,
    validate_official_inputs, load_official_ec_annotations, infer_ec_depth,
    build_ec_vocab, target_tokens_for_ids, protein_target_tokens, dense_target_masks,
    build_train_adjacency, sample_query_balanced_batch, rank_metrics, ec_diagnostics,
)

DEFAULT_UNIMOL = ROOT / "data/features/unimol/reaction_atomic_mean.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--reaction-features", type=Path, default=DEFAULT_UNIMOL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--positives-per-query", type=int, default=16)
    p.add_argument("--candidate-pool-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--projection-hidden-dim", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--temperature", type=float, default=.07)
    p.add_argument("--ec-conditioning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ec-router-normalization", choices=("parent_prior", "sibling_conditional"),
                   default="parent_prior", help="Optional marginal-consistent EC path; historical policy is the default.")
    p.add_argument("--all-positive-weight", type=float, default=.2)
    p.add_argument("--tail-risk-weight", type=float, default=.1)
    p.add_argument("--tail-risk-fraction", type=float, default=.2)
    p.add_argument("--ec-loss-weight", type=float, default=.1)
    p.add_argument("--consistency-loss-weight", type=float, default=.02)
    p.add_argument("--cross-modal-ec-weight", type=float, default=.02)
    p.add_argument("--aux-warmup-epochs", type=int, default=3)
    p.add_argument("--gradient-clip-norm", type=float, default=5.)
    p.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--eval-query-batch-size", type=int, default=128)
    p.add_argument("--eval-protein-batch-size", type=int, default=2048)
    p.add_argument("--max-steps-per-epoch", type=int, default=0, help="Smoke testing only; recorded in config.")
    p.add_argument("--fusion-selection-reference-study", type=Path,
                   help="Optional validation-only selection using the deployed frozen-anchor fusion")
    p.add_argument("--fusion-selection-support-study", type=Path)
    p.add_argument("--fusion-selection-guard-study", type=Path,
                   help="Previously selected 30-epoch system supplies validation head-metric floors")
    p.add_argument("--fusion-selection-tail-quantile", type=float, default=.95)
    p.add_argument("--fusion-selection-hit1-tolerance", type=float, default=.003)
    p.add_argument("--fusion-selection-mrr-tolerance", type=float, default=.002)
    return p.parse_args()


def load_frozen_features(path, reference, device):
    manifest_path = path.parent / "reaction_feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Uni-Mol extraction is not complete")
    if manifest.get("label_use") != {"reads_pair_labels": False, "reads_test_labels": False}:
        raise ValueError("Uni-Mol features do not certify label-independent extraction")
    idf_kinds = {"reaction_atomic_idf.pt": "train_idf_atomic_mean",
                 "reaction_atomic_idf_moments.pt": "train_idf_atomic_mean_std",
                 "reaction_atomic_idf_duplicate.pt": "train_idf_atomic_mean_duplicate"}
    if path.name not in ("reaction_atomic_mean.pt", *idf_kinds) or manifest.get("primary_output") != path.name:
        raise ValueError("This experiment requires declared original or TRAIN-IDF atomic-mean features")
    expected = Path(reference["processed_dir"]) / "reactions.parquet"
    if Path(manifest["inputs"]["reactions"]["path"]).resolve() != expected.resolve():
        raise ValueError("Uni-Mol source differs from official reaction inventory")
    if path.name in idf_kinds:
        aggregation = manifest.get("aggregation", {})
        if aggregation.get("kind") != idf_kinds[path.name] or aggregation.get("training_splits") != ["train"]:
            raise ValueError("IDF features must use TRAIN-only frequency fitting")
        statistics = json.loads(Path(aggregation["statistics"]).read_text())
        train_ids = sorted(set(pd.read_parquet(Path(reference["processed_dir"])/"pairs.parquet",
            columns=["reaction_id"], filters=[("split", "==", "train")]).reaction_id.astype(str)))
        if statistics.get("training_splits") != ["train"] or statistics.get("train_ids") != train_ids:
            raise ValueError("IDF statistics were not fitted on the exact official TRAIN reactions")
        # Reconstruct the actual counts, rather than trusting metadata alone.
        from ec_retrieval.molecule_pooling import fit_molecule_document_frequency
        from scripts.extract_tiger_unimol_features import split_components
        rows = pd.read_parquet(expected, columns=["reaction_id", "official_molecule_set_smiles"])
        train_id_set = set(train_ids)
        components = {str(r.reaction_id): split_components(str(r.official_molecule_set_smiles))[0]
                      for r in rows.itertuples(index=False) if str(r.reaction_id) in train_id_set}
        counts, n_train = fit_molecule_document_frequency(components, train_ids)
        if statistics.get("document_frequency") != counts or statistics.get("train_query_count") != n_train:
            raise ValueError("Stored IDF counts differ from direct official TRAIN reconstruction")
    data = torch.load(path, map_location="cpu", weights_only=False)
    qids = list(map(str, data["entries"]))
    q = data["embeddings"].float()
    if path.name in idf_kinds:
        if data.get("feature_kind") != idf_kinds[path.name]:
            raise ValueError("Feature tensor and pooling manifest disagree")
        dimension = 512 if path.name == "reaction_atomic_idf.pt" else 1024
        if q.ndim != 2 or q.shape[1] != dimension:
            raise ValueError("Pooling feature dimension mismatch")
        if path.name == "reaction_atomic_idf_duplicate.pt" and not torch.equal(q[:, :512], q[:, 512:]):
            raise ValueError("Equal-capacity control contains non-mean information")
        if path.name == "reaction_atomic_idf_moments.pt" and (q[:, 512:] < 0).any():
            raise ValueError("Population std must be nonnegative")
    official_ids = set(pd.read_parquet(expected, columns=["reaction_id"]).reaction_id.astype(str))
    if len(set(qids)) != len(qids) or set(qids) != official_ids:
        raise ValueError("Uni-Mol IDs do not exactly cover the official inventory")
    if q.ndim != 2 or q.shape[0] != len(qids) or not torch.isfinite(q).all():
        raise ValueError("Invalid reaction features")
    pids = list(map(str, json.loads(Path(reference["protein_global_ids"]).read_text())["enzyme_ids"]))
    with safe_open(reference["protein_global_embeddings"], framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["embeddings"]:
            raise ValueError("Unexpected ESM tensor schema")
        p = handle.get_tensor("embeddings").float()
    if len(set(pids)) != len(pids) or p.ndim != 2 or p.shape[0] != len(pids) or not torch.isfinite(p).all():
        raise ValueError("Invalid protein features")
    provenance = validate_global_feature_provenance(
        Path(reference["global_feature_bundle"]), Path(reference["protein_global_ids"]),
        Path(reference["protein_global_embeddings"]), Path(reference["processed_dir"]), len(qids), len(pids))
    audit = {"status": "passed", "reaction_count": len(qids), "protein_count": len(pids),
        "reaction_shape": list(q.shape), "protein_shape": list(p.shape),
        "reaction_feature_manifest": str(manifest_path.resolve()), "label_use": manifest["label_use"],
        "unimol_provenance": manifest["provenance"], "feature_fallback_counts": manifest["counts"],
        "all_official_reaction_rows_present": True, "ESM_provenance": provenance,
        "reaction_pooling": manifest.get("aggregation", {"kind": "official_atomic_mean"})}
    return qids, pids, q.to(device).detach(), p.to(device).detach(), audit


def cohort_metrics(predictions, train_queries):
    result = {}
    for name, selected in (
        ("train_seen", [p for p in predictions if p["query_id"] in train_queries]),
        ("train_unseen", [p for p in predictions if p["query_id"] not in train_queries]),
    ):
        ranks = [p["positive_ranks"] for p in selected]
        if not ranks:
            continue
        result.update({
            f"{name}_query_count": len(ranks),
            f"{name}_mean_rank": float(np.mean([np.mean(r) for r in ranks])),
            f"{name}_hit1": float(np.mean([min(r) == 1 for r in ranks])),
            f"{name}_all_mrr": float(np.mean([np.mean(1. / np.asarray(r)) for r in ranks])),
        })
    means = np.asarray([np.mean(p["positive_ranks"]) for p in predictions])
    n = math.ceil(.05 * len(means))
    result["worst5pct_mean_rank_sum_share"] = float(np.sort(means)[-n:].sum() / means.sum())
    return result


@torch.inference_mode()
def evaluate(model, qfeatures, pfeatures, qrows, prows, qids, pids, relevance,
             train_queries, val_labels, token_to_index, device, args):
    model.eval()
    enabled, dtype = amp_policy(device, args.amp_dtype)
    qvectors, pvectors = [], []
    predictions = [[] for _ in token_to_index]
    for start in range(0, len(qrows), args.eval_query_batch_size):
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            result = model.encode_reaction(qfeatures[qrows[start:start+args.eval_query_batch_size]])
        qvectors.append(result["embedding"].float().cpu())
        for level, value in enumerate(result["logits"]):
            predictions[level].append(value.float().argmax(1).cpu())
    for start in range(0, len(prows), args.eval_protein_batch_size):
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            result = model.encode_protein(pfeatures[prows[start:start+args.eval_protein_batch_size]])
        pvectors.append(result["embedding"].float().cpu())
    qv, pv = torch.cat(qvectors), torch.cat(pvectors)
    metrics, ranks = rank_metrics(qv, pv, qids, pids, relevance, device, args.eval_query_batch_size, 20,
                                  score_function=model.score_embeddings)
    metrics.update(cohort_metrics(ranks, train_queries))
    metrics.update(ec_diagnostics(qids, [torch.cat(p) for p in predictions], val_labels, token_to_index))
    return metrics, ranks, qv, pv


def main():
    args = parse_args()
    selection_paths = (args.fusion_selection_reference_study, args.fusion_selection_support_study,
                       args.fusion_selection_guard_study)
    if any(p is not None for p in selection_paths) and not all(p is not None for p in selection_paths):
        raise ValueError("Fusion selection requires all three reference/support/guard study paths")
    for name in ("epochs", "query_batch_size", "candidate_pool_size", "positives_per_query",
                 "projection_hidden_dim", "embedding_dim", "eval_query_batch_size", "eval_protein_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.aux_warmup_epochs < 0:
        raise ValueError("Invalid optimizer configuration")
    if not 0 <= args.dropout < 1 or not 0 < args.tail_risk_fraction <= 1:
        raise ValueError("Invalid dropout or tail fraction")
    for name in ("all_positive_weight", "tail_risk_weight"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
    output = args.output_dir.resolve()
    if (output / "config.json").exists() or (output / "best.pt").exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    reference = json.loads((args.reference_run / "config.json").read_text())
    validate_train_only_protocol(reference.get("refit_train_val", False))
    if reference.get("training_splits", ["train"]) != ["train"] or reference["seed"] != args.seed:
        raise ValueError("Reference must be a matching seed, train-only run")
    processed = Path(reference["processed_dir"])
    official = validate_official_inputs(processed, Path(reference["graph_bundle"]),
        Path(reference["protein_token_manifest"]), Path(reference["validation_contract"]),
        Path(reference["official_uniprot_source"]))
    atomic_json(output / "official_input_audit.json", official)
    qids, pids, qfeatures, pfeatures, feature_audit = load_frozen_features(args.reaction_features.resolve(), reference, device)
    atomic_json(output / "feature_audit.json", feature_audit)
    qrow, prow = {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)}
    train = pd.read_parquet(processed / "pairs.parquet", columns=["reaction_id", "enzyme_id", "split"],
                           filters=[("split", "==", "train")])
    if set(train.split.astype(str)) != {"train"} or train.duplicated(["reaction_id", "enzyme_id"]).any():
        raise ValueError("Invalid official train positive pairs")
    train_queries = set(train.reaction_id.astype(str))
    labels, label_audit = load_official_ec_annotations(processed, Path(reference["official_uniprot_source"]), ["train"])
    depth = infer_ec_depth(labels["train"])
    vocab, token_to_index, parents = build_ec_vocab(labels["train"], depth)
    vocab_sizes = [len(v) for v in vocab]
    # Train supervision is fully materialized before validation diagnostics load.
    qtargets = target_tokens_for_ids(qids, labels["train"], token_to_index)
    ptargets = protein_target_tokens(pids, train, labels["train"], token_to_index)
    adjacency = build_train_adjacency(train, qrow, prow)
    universe = np.asarray(sorted(set(train.enzyme_id.astype(str).map(prow))), dtype=np.int64)
    atomic_json(output / "train_ec_audit.json", label_audit)
    atomic_json(output / "ec_vocabulary.json", {"source": "official_train_only", "levels": vocab, "depth": depth})
    val_labels, val_label_audit = load_official_ec_annotations(processed, Path(reference["official_uniprot_source"]), ["val"])
    atomic_json(output / "validation_ec_diagnostic_audit.json", val_label_audit)
    vqids, vpids, relevance = load_contract(Path(reference["validation_contract"]), "val")
    vqrows, vprows = [qrow[q] for q in vqids], [prow[p] for p in vpids]
    selector = None
    selection_policy = "minimum_full_validation_mean_rank"
    if args.fusion_selection_reference_study is not None:
        from scripts.fusion_checkpoint_selection import FrozenFusionSelection
        selector = FrozenFusionSelection.from_artifacts(args, reference, train_queries, qrow, vqids, vpids, device)
        selection_policy = "minimum_full_validation_fused_mean_rank_with_head_guards"
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"architecture": "standalone_unimol_esm_function_transfer_with_predicted_ec",
        "training_splits": ["train"], "checkpoint_selection": selection_policy,
        "ground_truth_ec_is_model_input": False, "EC_role": "predicted_intermediate_and_train_auxiliary",
        "ec_depth": depth, "ec_vocab_sizes": vocab_sizes, "frozen_feature_encoders": ["Uni-Mol", "ESM-2"],
        "ec_router_group_reduction": "deterministic_csr" if args.ec_router_normalization == "sibling_conditional" else "global_softmax",
        "reference_weights_loaded": False, "initialization": "new_random_trainable_projections_and_router",
        "tail_risk_score": "raw", "positive_sampling": "coverage", "test_relevance_opened": False,
        "evaluation_split": "val", "processed_dir": str(processed),
        "validation_contract": reference["validation_contract"],
        "reaction_pooling": feature_audit["reaction_pooling"]["kind"],
        "official_uniprot_source": reference["official_uniprot_source"],
        "protein_global_ids": reference["protein_global_ids"],
        "protein_global_embeddings": reference["protein_global_embeddings"],
        "dataset_query_input": "official_unordered_molecule_inventory", "is_smoke": args.max_steps_per_epoch > 0})
    atomic_json(output / "config.json", config)
    atomic_json(output / "data_audit.json", {"training_splits": ["train"], "train_edges": len(train),
        "train_queries": len(adjacency), "train_enzymes": len(universe),
        "validation_queries": len(vqids), "validation_candidates": len(vpids),
        "strict_train_ec_labeled_queries": len(labels["train"]), "unobserved_pairs_are_confirmed_negatives": False,
        "validation_supervises_training": False, "test_relevance_opened": False})
    model_kwargs = dict(reaction_input_dim=qfeatures.shape[1], protein_input_dim=pfeatures.shape[1],
        ec_vocab_sizes=vocab_sizes, child_to_parent=parents, projection_hidden_dim=args.projection_hidden_dim,
        embedding_dim=args.embedding_dim, dropout=args.dropout, temperature=args.temperature,
        ec_conditioning=args.ec_conditioning, ec_router_normalization=args.ec_router_normalization)
    model = FunctionTransferRetriever(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate*.05)
    enabled, dtype = amp_policy(device, args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16")
    rng = np.random.default_rng(args.seed)
    negative_rng = np.random.default_rng(args.seed + 1_000_003)
    exposure = {q: np.zeros(len(v), dtype=np.int64) for q, v in adjacency.items()}
    best_mr = math.inf
    best_standalone_mr = math.inf
    for epoch in range(args.epochs):
        began = time.perf_counter()
        model.train()
        totals = defaultdict(float)
        order = rng.permutation(np.asarray(sorted(adjacency), dtype=np.int64))
        steps = 0
        for start in range(0, len(order), args.query_batch_size):
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
            queries = list(map(int, order[start:start+args.query_batch_size]))
            candidates, mask_cpu = sample_query_balanced_batch(queries, adjacency, args.positives_per_query,
                rng, candidate_universe=universe, target_candidate_count=args.candidate_pool_size,
                negative_rng=negative_rng, positive_exposure_counts=exposure)
            mask = mask_cpu.to(device)
            qt = dense_target_masks(queries, qtargets, vocab_sizes, device)
            pt = dense_target_masks(candidates, ptargets, vocab_sizes, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                scores, reaction, protein = model(qfeatures[queries], pfeatures[candidates])
                retrieval, components = multi_positive_tail_risk_loss(scores, mask,
                    all_positive_weight=args.all_positive_weight, tail_risk_weight=args.tail_risk_weight,
                    tail_fraction=args.tail_risk_fraction, tail_risk_score="raw")
                ec_loss = .5 * (hierarchical_multi_positive_loss(reaction["logits"], qt)
                                + hierarchical_multi_positive_loss(protein["logits"], pt))
                consistency = .5 * (hierarchy_consistency_loss(reaction["logits"], model.ec_router.parent_maps())
                                    + hierarchy_consistency_loss(protein["logits"], model.ec_router.parent_maps()))
                edges = mask.nonzero()
                alignment = cross_modal_ec_alignment_loss(reaction["logits"],
                    [v.index_select(0, edges[:, 1]) for v in protein["logits"]], edges[:, 0])
                warmup = min(1., (epoch+1)/args.aux_warmup_epochs) if args.aux_warmup_epochs else 1.
                loss = retrieval + warmup * (args.ec_loss_weight * ec_loss
                    + args.consistency_loss_weight * consistency + args.cross_modal_ec_weight * alignment)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {steps}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            for key, value in {"loss": loss, "retrieval": retrieval, "ec": ec_loss,
                               "consistency": consistency, "alignment": alignment, "gradient_norm": norm,
                               **components}.items():
                totals[key] += float(value.detach())
            steps += 1
        scheduler.step()
        train_metrics = {key: value/steps for key, value in totals.items()}
        train_metrics.update(epoch=epoch, steps=steps, seconds=time.perf_counter()-began)
        append_jsonl(output / "train_metrics.jsonl", train_metrics)
        metrics, _, qv, pv = evaluate(model, qfeatures, pfeatures, vqrows, vprows, vqids, vpids, relevance,
            train_queries, val_labels.get("val", {}), token_to_index, device, args)
        metrics["epoch"] = epoch
        append_jsonl(output / "validation_metrics.jsonl", metrics)
        checkpoint = {"model_state": model.state_dict(), "model_kwargs": model_kwargs,
                      "epoch": epoch, "validation_metrics": metrics, "config": config}
        chosen_metrics, eligible = metrics, True
        if selector is not None:
            if metrics["official_reactzyme_mean_rank"] < best_standalone_mr:
                best_standalone_mr = metrics["official_reactzyme_mean_rank"]
                atomic_torch_save(output / "best_standalone.pt", checkpoint)
            chosen_metrics, fused_ranks, eligible, support_payload = selector.evaluate(model, qfeatures, qv, pv, relevance)
            chosen_metrics.update(cohort_metrics(fused_ranks, train_queries))
            chosen_metrics.update(epoch=epoch, head_guards_pass=eligible)
            append_jsonl(output / "fused_validation_metrics.jsonl", chosen_metrics)
            checkpoint["selection_metrics"] = chosen_metrics
        improved = eligible and chosen_metrics["official_reactzyme_mean_rank"] < best_mr
        if improved:
            best_mr = chosen_metrics["official_reactzyme_mean_rank"]
            atomic_torch_save(output / "best.pt", checkpoint)
            if selector is not None:
                atomic_torch_save(output / "best_support_bank.pt", support_payload)
        print(json.dumps({"epoch": epoch, "train_loss": train_metrics["loss"],
            "val_mean_rank": metrics["official_reactzyme_mean_rank"], "val_hit1": metrics["hit@1"],
            "val_unseen_mean_rank": metrics["train_unseen_mean_rank"], "improved": improved,
            "selection_mean_rank": chosen_metrics["official_reactzyme_mean_rank"], "head_guards_pass": eligible}), flush=True)
    if not (output / "best.pt").exists():
        raise RuntimeError("No checkpoint passed the predeclared head guards; preserve the prior selected model")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    final, ranks, qv, pv = evaluate(model, qfeatures, pfeatures, vqrows, vprows, vqids, vpids, relevance,
        train_queries, val_labels.get("val", {}), token_to_index, device, args)
    for key in ("official_reactzyme_mean_rank", "hit@1", "official_reactzyme_mrr"):
        if abs(final[key]-best["validation_metrics"][key]) > 1e-6:
            raise ValueError("Best checkpoint validation did not reproduce on reload")
    for row in ranks:
        append_jsonl(output / "validation_topk.jsonl", row)
    atomic_torch_save(output / "validation_embeddings.pt", {"query_ids": vqids, "candidate_ids": vpids,
        "query_embeddings": qv, "candidate_embeddings": pv, "split": "val", "epoch": best["epoch"]})
    exposures = np.concatenate(list(exposure.values()))
    summary = {"status": "completed", "epochs_completed": args.epochs, "best_epoch": best["epoch"],
        "best_validation_metrics": final, "training_splits": ["train"], "test_relevance_opened": False,
        "ground_truth_ec_is_inference_input": False, "checkpoint_selection": selection_policy,
        "direct_positive_sampling_coverage": float((exposures > 0).mean()),
        "coverage_note": "Counts explicit positive draws only; all known positives entering each pool were masked as positive.",
        "is_smoke": args.max_steps_per_epoch > 0}
    if selector is not None:
        fused, fused_ranks, eligible, support_payload = selector.evaluate(model, qfeatures, qv, pv, relevance)
        fused.update(cohort_metrics(fused_ranks, train_queries))
        for key in ("official_reactzyme_mean_rank", "hit@1", "official_reactzyme_mrr"):
            if abs(fused[key]-best["selection_metrics"][key]) > 1e-6:
                raise ValueError("Fused best checkpoint failed strict reload verification")
        if not eligible:
            raise ValueError("Reloaded fused checkpoint no longer passes head guards")
        retained = torch.load(output / "best_support_bank.pt", map_location="cpu", weights_only=False)
        if not torch.allclose(retained["query_weights"], support_payload["query_weights"], rtol=0, atol=1e-7):
            raise ValueError("TRAIN support-derived weights changed on checkpoint reload")
        for row in fused_ranks:
            append_jsonl(output / "fused_validation_topk.jsonl", row)
        atomic_json(output / "selected_fusion_metrics.json", fused)
        standalone_best = torch.load(output / "best_standalone.pt", map_location="cpu", weights_only=False)
        summary.update(selected_fusion_metrics=fused, standalone_selected_epoch=standalone_best["epoch"],
                       standalone_selected_metrics=standalone_best["validation_metrics"],
                       fusion_selection_updates_parameters=False)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
