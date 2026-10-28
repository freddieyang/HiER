#!/usr/bin/env python3
"""Controlled TRAIN-only candidate coverage with optional current-model mining.

Only TRAIN associations update parameters or fit molecule statistics. VAL is
reserved for checkpoint selection. This script has no test or refit option.
Original trainers and completed studies are preserved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.morgan_dropout_schedule import morgan_probability_for_epoch
from ec_retrieval.coarse_ec_targets import load_train_consensus, supplement_reaction_targets
from ec_retrieval.rank_selected_tail import rank_selected_tail_risk_loss
from ec_retrieval.training_candidates import (TrainCandidateDomain,
    mine_current_train_candidates, sample_training_candidates)
from ec_retrieval.shared_functional_set_retriever import SharedFusionFunctionalSetRetriever
from ec_retrieval.molecule_store import MoleculeFeatureStore
from ec_retrieval.target_store import ECTargetStore
from ec_retrieval.molecule_pooling import fit_molecule_document_frequency, smooth_molecule_idf, pool_molecule_features
from ec_retrieval import (multi_positive_tail_risk_loss, hierarchical_multi_positive_loss,
                         hierarchy_consistency_loss, cross_modal_ec_alignment_loss)
from scripts.extract_tiger_unimol_features import split_components
from scripts.train_function_transfer_retriever import load_frozen_features, evaluate
from scripts.train_global_anchored_retriever import atomic_torch_save, validate_train_only_protocol
from scripts.train_structured_retriever import (
    atomic_json, append_jsonl, seed_everything, amp_policy, load_contract,
    validate_official_inputs, load_official_ec_annotations, infer_ec_depth,
    build_ec_vocab, target_tokens_for_ids, protein_target_tokens, dense_target_masks,
    build_train_adjacency, sample_query_balanced_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True, help="Existing audited TRAIN-only config, not its weights")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/features/unimol")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pooling", choices=("moments", "attention", "ec_attention"), default="ec_attention")
    parser.add_argument("--ec-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion-sharing", choices=("independent_matched", "shared"), required=True,
                        help="Both variants start with the same fusion weights; only shared ties them throughout training")
    parser.add_argument("--drop-morgan-probability", type=float, default=0., help="TRAIN-only whole-view masking; never mask both views")
    parser.add_argument("--drop-unimol-probability", type=float, default=0.)
    parser.add_argument("--morgan-dropout-schedule", choices=("constant", "linear_decay"), required=True)
    parser.add_argument("--reaction-ec-supervision", choices=("strict",), required=True)
    parser.add_argument("--tail-selector", choices=("raw",), required=True,
                        help="Choose difficult queries by original CE or sampled excess rank, preserving CE gradients")
    parser.add_argument("--candidate-sampling", choices=("random", "dynamic"), required=True)
    parser.add_argument("--mining-warmup-epochs", type=int, default=5)
    parser.add_argument("--mining-refresh-epochs", type=int, default=5)
    parser.add_argument("--mining-bank-size", type=int, default=64)
    parser.add_argument("--mined-per-query", type=int, default=4)
    parser.add_argument("--mining-query-batch-size", type=int, default=64)
    parser.add_argument("--mining-protein-batch-size", type=int, default=2048)
    parser.add_argument("--random-reserve-fraction", type=float, default=.25,
                        help="Reserve up to this pool fraction after preserving direct positives")
    parser.add_argument("--cache-ec-targets", action=argparse.BooleanOptionalAction, default=False,
                        help="Preload TRAIN boolean EC targets once to avoid per-step Python target assembly")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--positives-per-query", type=int, default=16)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--temperature", type=float, default=.07)
    parser.add_argument("--all-positive-weight", type=float, default=.2)
    parser.add_argument("--tail-risk-weight", type=float, default=.1)
    parser.add_argument("--tail-risk-fraction", type=float, default=.2)
    parser.add_argument("--ec-loss-weight", type=float, default=.1)
    parser.add_argument("--consistency-loss-weight", type=float, default=.02)
    parser.add_argument("--cross-modal-ec-weight", type=float, default=.02)
    parser.add_argument("--aux-warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--eval-query-batch-size", type=int, default=128)
    parser.add_argument("--eval-protein-batch-size", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Smoke only; 0 processes all TRAIN queries")
    return parser.parse_args()


def load_component_features(reference, source, device):
    """Reconstruct the audited official feature means before exposing tokens."""
    qids, pids, original_mean, proteins, audit = load_frozen_features(
        source / "reaction_atomic_mean.pt", reference, torch.device("cpu"))
    manifest = json.loads((source / "reaction_feature_manifest.json").read_text())
    molecules, vectors = [], []
    for entry in manifest["shards"]:
        shard = torch.load(entry["path"], map_location="cpu", weights_only=False)
        molecules.extend(shard["smiles"])
        vectors.append(shard["atomic_mean"].float())
    vectors = torch.cat(vectors)
    if len(set(molecules)) != len(molecules) or vectors.shape != (len(molecules), 512):
        raise ValueError("Invalid audited per-molecule cache")
    processed = Path(reference["processed_dir"])
    frame = pd.read_parquet(processed / "reactions.parquet", columns=["reaction_id", "official_molecule_set_smiles"])
    if frame.reaction_id.duplicated().any() or set(frame.reaction_id.astype(str)) != set(qids):
        raise ValueError("Official reaction metadata differs from cached rows")
    components = {str(r.reaction_id): split_components(str(r.official_molecule_set_smiles))[0]
                  for r in frame.itertuples(index=False)}
    train_ids = sorted(set(pd.read_parquet(processed / "pairs.parquet", columns=["reaction_id"],
        filters=[("split", "==", "train")]).reaction_id.astype(str)))
    counts, n_train = fit_molecule_document_frequency(components, train_ids)
    weights = smooth_molecule_idf(molecules, counts, n_train)
    lookup = {m: i for i, m in enumerate(molecules)}
    component_rows = [[lookup[m] for m in components[q]] for q in qids]
    lengths = [len(rows) for rows in component_rows]
    reconstructed = pool_molecule_features(vectors, [v for row in component_rows for v in row], lengths)
    error = float((reconstructed - original_mean).abs().max())
    if error > 2e-6:
        raise ValueError(f"Token cache fails original mean reconstruction: {error}")
    bundle = torch.load(reference["global_feature_bundle"], map_location="cpu", weights_only=False)
    if list(map(str, bundle["reaction_ids"])) != qids:
        raise ValueError("Morgan rows differ from the audited molecule rows")
    contract = bundle["feature_contract"]
    if any(contract.get(k, True) for k in ("uses_pair_associations", "uses_split_labels", "uses_enzyme_or_ec_labels", "uses_reaction_direction")):
        raise ValueError("Morgan bundle violates the reaction-only input contract")
    store = MoleculeFeatureStore(vectors, component_rows, weights, bundle["reaction_fingerprints"], device)
    audit.update(original_mean_reconstruction_max_abs=error, unique_molecules=len(molecules),
        total_components=sum(lengths), maximum_components=max(lengths),
        molecule_fitting_splits=["train"], train_frequency_queries=n_train,
        official_component_multiset_preserved=True, pretrained_encoders_frozen=True,
        feature_statistics_source="TRAIN reactions only; no pair labels in model input")
    frequency = {"training_splits": ["train"], "train_ids": train_ids,
                 "document_frequency": counts, "train_query_count": n_train}
    return qids, pids, store, proteins.to(device), audit, frequency


def validate_args(args):
    for name in ("epochs", "query_batch_size", "positives_per_query", "candidate_pool_size",
                 "projection_hidden_dim", "embedding_dim", "num_threads", "eval_query_batch_size", "eval_protein_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_steps_per_epoch < 0 or args.aux_warmup_epochs < 0:
        raise ValueError("Step cap and warmup must be nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 <= args.dropout < 1 or args.temperature <= 0 or not 0 < args.tail_risk_fraction <= 1:
        raise ValueError("Invalid dropout, temperature or tail fraction")
    if any(not 0 <= getattr(args, key) <= 1 for key in ("all_positive_weight", "tail_risk_weight")):
        raise ValueError("Retrieval mixture weights must lie in [0,1]")
    if any(getattr(args, key) < 0 for key in ("ec_loss_weight", "consistency_loss_weight", "cross_modal_ec_weight")):
        raise ValueError("Auxiliary weights must be nonnegative")
    if not (0 <= args.drop_morgan_probability <= 1 and 0 <= args.drop_unimol_probability <= 1
            and args.drop_morgan_probability + args.drop_unimol_probability <= 1):
        raise ValueError("Invalid reaction view-drop probabilities")
    if args.pooling != "moments" and args.drop_morgan_probability + args.drop_unimol_probability > 0:
        raise ValueError("View masking is defined for moments pooling only")
    if args.mining_warmup_epochs < 0 or min(args.mining_refresh_epochs, args.mining_bank_size,
            args.mined_per_query, args.mining_query_batch_size, args.mining_protein_batch_size) < 1:
        raise ValueError("Invalid candidate mining schedule or batch size")
    if not 0 <= args.random_reserve_fraction <= 1:
        raise ValueError("Invalid random reserve")
    morgan_probability_for_epoch(args.morgan_dropout_schedule, args.drop_morgan_probability, 0, args.epochs)
    if args.fusion_sharing != "independent_matched" or args.drop_unimol_probability != 0:
        raise ValueError("Schedule study requires independent fusion and unmasked Uni-Mol")


def main():
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no formal training was started. Use --device cpu only explicitly.")
        torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an experiment directory")
    torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    reference = json.loads((args.reference_run / "config.json").read_text())
    validate_train_only_protocol(reference.get("refit_train_val", False))
    if reference.get("training_splits", ["train"]) != ["train"]:
        raise ValueError("Reference config must certify TRAIN-only optimization")
    processed = Path(reference["processed_dir"])
    official = validate_official_inputs(processed, Path(reference["graph_bundle"]),
        Path(reference["protein_token_manifest"]), Path(reference["validation_contract"]),
        Path(reference["official_uniprot_source"]))
    qids, pids, qfeatures, pfeatures, audit, frequency = load_component_features(reference, args.source_dir.resolve(), device)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "official_input_audit.json", official)
    atomic_json(output / "feature_audit.json", audit)
    atomic_json(output / "train_frequency.json", frequency)
    qrow, prow = {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)}
    train = pd.read_parquet(processed / "pairs.parquet", columns=["reaction_id", "enzyme_id", "split"],
                           filters=[("split", "==", "train")])
    if set(train.split.astype(str)) != {"train"} or train.duplicated(["reaction_id", "enzyme_id"]).any():
        raise ValueError("Invalid TRAIN associations")
    train_queries = set(train.reaction_id.astype(str))
    labels, ec_audit = load_official_ec_annotations(processed, Path(reference["official_uniprot_source"]), ["train"])
    depth = infer_ec_depth(labels["train"])
    vocab, token_to_index, parents = build_ec_vocab(labels["train"], depth)
    vocab_sizes = [len(v) for v in vocab]
    qtargets = target_tokens_for_ids(qids, labels["train"], token_to_index)
    ptargets = protein_target_tokens(pids, train, labels["train"], token_to_index)
    # The protein targets are constructed from the unchanged strict labels.
    # Supplement only reaction-side auxiliary targets before VAL diagnostics.
    if args.reaction_ec_supervision == "coarse_consensus_l1_l3":
        consensus = load_train_consensus(processed, Path(reference["official_uniprot_source"]), train)
        qtargets, target_audit, supplement = supplement_reaction_targets(
            qids, qtargets, token_to_index, consensus, set(labels["train"]))
    else:
        counts = [sum(bool(v) for v in level) for level in qtargets]
        target_audit = dict(policy="strict", training_splits=["train"], supervision_before=counts,
            supervision_after=counts, added_by_level=[0]*depth, added_query_count=0,
            original_targets_preserved=True, l4_unchanged=True, vocabulary_unchanged=True,
            nontrain_targets_unchanged=True, protein_targets_unchanged=True,
            missing_children_imputed=False, later_rhea_snapshot_read=False, test_labels_read=False)
        supplement = []
    atomic_json(output / "reaction_ec_target_audit.json", target_audit)
    pd.DataFrame(supplement, columns=["reaction_id", "level", "prefix"]).to_csv(
        output / "reaction_ec_supplement.csv", index=False)
    atomic_torch_save(output / "training_ec_targets.pt", dict(query_ids=qids, protein_ids=pids,
        reaction_targets=qtargets, protein_targets=ptargets, training_splits=["train"]))

    qtarget_store = ECTargetStore(qtargets, vocab_sizes, device) if args.cache_ec_targets else None
    ptarget_store = ECTargetStore(ptargets, vocab_sizes, device) if args.cache_ec_targets else None
    adjacency = build_train_adjacency(train, qrow, prow)
    universe = np.asarray(sorted(set(train.enzyme_id.astype(str).map(prow))), dtype=np.int64)
    domain = TrainCandidateDomain(adjacency, universe)
    atomic_torch_save(output / "candidate_domain.pt", dict(query_rows=torch.from_numpy(domain.query_rows),
        candidate_rows=torch.from_numpy(universe), training_splits=["train"]))
    atomic_json(output / "train_ec_audit.json", ec_audit)
    atomic_json(output / "ec_vocabulary.json", {"source": "official_train_only", "levels": vocab, "depth": depth})
    # Materialize all optimization targets before any validation diagnostics.
    val_labels, val_audit = load_official_ec_annotations(processed, Path(reference["official_uniprot_source"]), ["val"])
    vqids, vpids, relevance = load_contract(Path(reference["validation_contract"]), "val")
    vqrows, vprows = [qrow[q] for q in vqids], [prow[p] for p in vpids]
    atomic_json(output / "validation_ec_diagnostic_audit.json", val_audit)
    kwargs = dict(molecule_input_dim=qfeatures.molecule_input_dim, global_input_dim=qfeatures.global_input_dim,
        protein_input_dim=pfeatures.shape[1], ec_vocab_sizes=vocab_sizes, child_to_parent=parents,
        projection_hidden_dim=args.projection_hidden_dim, embedding_dim=args.embedding_dim,
        dropout=args.dropout, temperature=args.temperature, pooling=args.pooling,
        ec_conditioning=args.ec_conditioning, ec_router_normalization="parent_prior",
        drop_morgan_probability=args.drop_morgan_probability, drop_unimol_probability=args.drop_unimol_probability,
        fusion_sharing=args.fusion_sharing)
    if args.fusion_sharing != "independent_matched" or args.drop_unimol_probability != 0:
        raise ValueError("This schedule study fixes independent fusion and preserves the Uni-Mol view")
    schedule = [morgan_probability_for_epoch(args.morgan_dropout_schedule,
        args.drop_morgan_probability, epoch, args.epochs) for epoch in range(args.epochs)]
    atomic_json(output / "masking_schedule.json", dict(policy=args.morgan_dropout_schedule,
        probabilities=schedule, mean_probability=sum(schedule)/len(schedule),
        source="declared_epoch_index_only", validation_used=False))
    model = SharedFusionFunctionalSetRetriever(**kwargs).to(device)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture="shared_fusion_functional_set_retriever", training_splits=["train"],
        pretrained_encoders_frozen=["Uni-Mol", "ESM-2"], initialization="random_trainable_modules_no_teacher_or_anchor",
        fusion_initialization="identical random reaction/protein fusion weights in both paired controls",
        ground_truth_ec_is_model_input=False, checkpoint_selection="minimum_full_validation_mean_rank",
        reference_weights_loaded=False, parameter_count=sum(p.numel() for p in model.parameters()),
        ec_depth=depth, ec_vocab_sizes=vocab_sizes, processed_dir=str(processed),
        validation_contract=reference["validation_contract"], test_relevance_opened=False,
        historical_time_test_previously_observed=True, is_smoke=args.max_steps_per_epoch > 0,
        candidate_index_dimension=args.embedding_dim, ensemble_members=1,
        external_support_bank=False, posthoc_density_correction=False,
        optimization_scope="all retrieval parameters jointly; pretrained feature extractors frozen",
        ec_target_cache_bytes=(qtarget_store.nbytes + ptarget_store.nbytes) if args.cache_ec_targets else 0,
        parallelism="vectorized padded batches; DDP not implemented; do not row-shard both score axes with DataParallel")
    config.update(candidate_source="enzymes_with_TRAIN_associations_only",
        mining_active=args.candidate_sampling == "dynamic", inference_requires_mining_bank=False,
        known_positive_policy="all_TRAIN_associations_in_pool_are_positive")
    atomic_json(output / "config.json", config)
    atomic_json(output / "data_audit.json", dict(training_splits=["train"], train_edges=len(train),
        train_queries=len(adjacency), train_enzymes=len(universe), validation_queries=len(vqids),
        validation_candidates=len(vpids), test_relevance_opened=False,
        validation_supervises_training=False, unobserved_pairs_are_confirmed_negatives=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.05*args.learning_rate)
    enabled, dtype = amp_policy(device, args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16")
    rng, negative_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1_000_003)
    exposure = {q: np.zeros(len(v), dtype=np.int64) for q, v in adjacency.items()}
    hard_rng = np.random.default_rng(args.seed + 2_000_033)
    bank, mining_events, query_orders = None, [], []
    planned_mining_epochs = (list(range(args.mining_warmup_epochs, args.epochs, args.mining_refresh_epochs))
        if args.candidate_sampling == "dynamic" else [])
    def save_mining_audit():
        atomic_json(output / "candidate_mining_audit.json", dict(training_splits=["train"],
            mode=args.candidate_sampling, planned_epochs_zero_based=planned_mining_epochs,
            events=mining_events, validation_or_test_associations_used=False,
            ground_truth_ec_input=False, inference_dependency=False))
    save_mining_audit()
    best_mr = math.inf
    for epoch in range(args.epochs):
        began = time.perf_counter()
        if epoch in planned_mining_epochs:
            bank, artifact = mine_current_train_candidates(model, qfeatures, pfeatures, domain,
                bank_size=args.mining_bank_size, query_batch_size=args.mining_query_batch_size,
                protein_batch_size=args.mining_protein_batch_size, amp_enabled=enabled, amp_dtype=dtype)
            artifact.update(before_epoch_zero_based=epoch, completed_training_epochs=epoch)
            path = output / "mining" / f"before_epoch{epoch:03d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(path, artifact)
            mining_events.append(dict(before_epoch_zero_based=epoch, path=str(path),
                seconds=artifact["seconds"], query_count=artifact["query_count"],
                candidate_count=artifact["candidate_count"], known_positive_overlap=0))
            save_mining_audit()
            print(json.dumps(dict(event="candidate_refresh", epoch=epoch,
                seconds=artifact["seconds"], training_splits=["train"])), flush=True)
            del artifact
        model.train()
        model.view_dropout.morgan_probability = schedule[epoch]
        totals, steps = defaultdict(float), 0
        order = rng.permutation(np.asarray(sorted(adjacency), dtype=np.int64))
        query_orders.append(torch.from_numpy(order.copy()))
        sampling_totals = defaultdict(float)
        for start in range(0, len(order), args.query_batch_size):
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
            queries = list(map(int, order[start:start+args.query_batch_size]))
            candidates, mask, sampling = sample_training_candidates(queries, domain, args.positives_per_query, rng,
                target_candidate_count=args.candidate_pool_size, negative_rng=negative_rng,
                positive_exposure_counts=exposure, bank=bank, hard_rng=hard_rng,
                hard_per_query=args.mined_per_query, random_reserve_fraction=args.random_reserve_fraction)
            for key, value in sampling.items():
                sampling_totals[key] += value
            mask = mask.to(device)
            qt = qtarget_store[queries] if qtarget_store is not None else dense_target_masks(queries, qtargets, vocab_sizes, device)
            pt = ptarget_store[candidates] if ptarget_store is not None else dense_target_masks(candidates, ptargets, vocab_sizes, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                scores, reaction, protein = model(qfeatures[queries], pfeatures[candidates])
                retrieval, components = rank_selected_tail_risk_loss(scores, mask,
                    all_positive_weight=args.all_positive_weight, tail_risk_weight=args.tail_risk_weight,
                    tail_fraction=args.tail_risk_fraction, tail_selector=args.tail_selector)
                ec = .5 * (hierarchical_multi_positive_loss(reaction["logits"], qt)
                           + hierarchical_multi_positive_loss(protein["logits"], pt))
                consistency = .5 * (hierarchy_consistency_loss(reaction["logits"], model.ec_router.parent_maps())
                                    + hierarchy_consistency_loss(protein["logits"], model.ec_router.parent_maps()))
                edges = mask.nonzero()
                alignment = cross_modal_ec_alignment_loss(reaction["logits"],
                    [v.index_select(0, edges[:, 1]) for v in protein["logits"]], edges[:, 0])
                warmup = min(1., (epoch+1)/args.aux_warmup_epochs) if args.aux_warmup_epochs else 1.
                loss = retrieval + warmup * (args.ec_loss_weight*ec + args.consistency_loss_weight*consistency
                                             + args.cross_modal_ec_weight*alignment)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {steps}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            for key, value in dict(loss=loss, retrieval=retrieval, ec=ec, consistency=consistency,
                                   alignment=alignment, gradient_norm=norm, **components).items():
                # Keep detached FP64 scalar sums on device; one transfer per epoch.
                totals[key] = totals[key] + value.detach().double()
            steps += 1
        scheduler.step()
        metric_values = torch.stack(list(totals.values())).div(steps).cpu().tolist()
        train_metrics = dict(zip(totals, metric_values))
        train_metrics.update(epoch=epoch, steps=steps, seconds=time.perf_counter()-began,
            morgan_drop_probability=schedule[epoch])
        train_metrics.update({"sampling_" + k: v / steps for k, v in sampling_totals.items()})
        append_jsonl(output / "train_metrics.jsonl", train_metrics)
        metrics, _, _, _ = evaluate(model, qfeatures, pfeatures, vqrows, vprows, vqids, vpids, relevance,
            train_queries, val_labels.get("val", {}), token_to_index, device, args)
        metrics["epoch"] = epoch
        append_jsonl(output / "validation_metrics.jsonl", metrics)
        improved = metrics["official_reactzyme_mean_rank"] < best_mr
        if improved:
            best_mr = metrics["official_reactzyme_mean_rank"]
            atomic_torch_save(output / "best.pt", dict(model_state=model.state_dict(), model_kwargs=kwargs,
                epoch=epoch, validation_metrics=metrics, config=config,
                training_morgan_drop_probability=schedule[epoch]))
        print(json.dumps(dict(epoch=epoch, train_loss=train_metrics["loss"],
            val_mean_rank=metrics["official_reactzyme_mean_rank"], val_hit1=metrics["hit@1"],
            val_all_mrr=metrics["official_reactzyme_mrr"], improved=improved, is_smoke=config["is_smoke"])), flush=True)
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    final, ranks, qv, pv = evaluate(model, qfeatures, pfeatures, vqrows, vprows, vqids, vpids, relevance,
        train_queries, val_labels.get("val", {}), token_to_index, device, args)
    for key in ("official_reactzyme_mean_rank", "hit@1", "official_reactzyme_mrr"):
        if abs(final[key] - best["validation_metrics"][key]) > 1e-6:
            raise ValueError("Reloaded checkpoint failed validation reproduction")
    for row in ranks:
        append_jsonl(output / "validation_topk.jsonl", row)
    atomic_torch_save(output / "validation_embeddings.pt", dict(query_ids=vqids, candidate_ids=vpids,
        query_embeddings=qv, candidate_embeddings=pv, split="val", epoch=best["epoch"]))
    atomic_torch_save(output / "sampling_trace.pt", dict(query_orders=torch.stack(query_orders),
        query_rows=torch.from_numpy(domain.query_rows),
        direct_positive_exposures=[torch.from_numpy(exposure[int(q)].copy()) for q in domain.query_rows],
        training_splits=["train"]))
    coverage = np.concatenate(list(exposure.values()))
    summary = dict(status="smoke_completed" if config["is_smoke"] else "completed", epochs_completed=args.epochs,
        best_epoch=best["epoch"], best_validation_metrics=final, training_splits=["train"],
        is_smoke=config["is_smoke"], checkpoint_reload_verified=True, test_relevance_opened=False,
        direct_positive_sampling_coverage=float((coverage > 0).mean()), parameter_count=config["parameter_count"],
        index_dimension=args.embedding_dim, trainable_backbones=False, ensemble_members=1,
        candidate_sampling=args.candidate_sampling, mining_refreshes_completed=len(mining_events))
    if device.type == "cuda":
        summary.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
