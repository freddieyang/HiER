#!/usr/bin/env python3
"""Freeze the predeclared residue-readout comparison, audit TEST, encode and score.

This evaluator never trains, selects a checkpoint, or promotes an arm on TEST.
Encoding accepts entity IDs and label-independent features; relevance is used
only by the protocol audit and scoring phases. Run the frozen script after the
freeze phase so imported model and evaluation source is fixed as well.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime
import filecmp
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("mean", "residue_attention")
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def tensor(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def now():
    return datetime.datetime.now().astimezone().isoformat()


def file_state(path):
    path = Path(path).resolve()
    st = path.stat()
    return dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)


def verify_freeze(study):
    inventory = read(study / "freeze_inventory.json")
    filecmp.clear_cache()
    for item in inventory:
        if not filecmp.cmp(item["source"], item["copy"], shallow=False):
            raise ValueError(f"Frozen artifact changed: {item['source']}")
    inputs = read(study / "input_inventory.json")
    for state in inputs:
        if file_state(state["path"]) != state:
            raise ValueError(f"Data/feature metadata changed: {state['path']}")
    return dict(copied_artifacts_byte_equal=len(inventory),
                external_input_size_mtime_unchanged=len(inputs))


def freeze(study, training_study):
    if study.exists():
        raise FileExistsError(study)
    plan = read(training_study / "plan.json")
    selection = read(training_study / "analysis/selection.json")
    assert selection["status"] == "frozen_from_validation" and selection["test_used"] is False
    assert selection["primary"] in ARMS
    assert selection["new_test_authorized_by_frozen_plan"] is True
    assert read(training_study / "analysis/audit.json")["status"] == "passed"
    assert plan["status"] == "completed"
    assert {(j["cell"], j["seed"]) for j in plan["jobs"]} == {(a, s) for a in ARMS for s in SEEDS}
    assert read(training_study / "independent_reload.json")["status"] == "passed"
    # The source used for the training experiment must still match before use.
    for source in (training_study / "source_snapshot").rglob("*.py"):
        live = ROOT / source.relative_to(training_study / "source_snapshot")
        if not filecmp.cmp(source, live, shallow=False):
            raise ValueError(f"Training source drift: {live}")
    study.mkdir(parents=True)
    inventory = []

    def copy(source, target):
        source, target = Path(source).resolve(), Path(target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory.append(dict(source=str(source), copy=str(target), size=source.stat().st_size))

    for directory in ("scripts", "ec_retrieval"):
        for source in sorted((ROOT / directory).rglob("*.py")):
            copy(source, study / "frozen/source" / source.relative_to(ROOT))
    copy(ROOT / "external/ReactZyme_official/retrieval.py",
         study / "frozen/source/external/ReactZyme_official/retrieval.py")
    copy(training_study / "plan.json", study / "frozen/training_plan.json")
    copy(training_study / "analysis/selection.json", study / "frozen/selection.json")
    copy(training_study / "analysis/audit.json", study / "frozen/training_analysis_audit.json")
    copy(training_study / "independent_reload.json", study / "frozen/training_reload_audit.json")
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            run = training_study / arm / f"seed{seed}"
            cfg, summary, checkpoint = read(run / "config.json"), read(run / "summary.json"), tensor(run / "best.pt")
            assert summary["status"] == "completed" and summary["epochs_completed"] == 60
            assert cfg["training_splits"] == ["train"] and not cfg["test_relevance_opened"]
            assert not cfg["is_smoke"] and not cfg["ground_truth_ec_is_model_input"]
            assert cfg["scoring"] == "explicit" and cfg["positive_coverage"] == "legacy"
            assert cfg["ec_posterior"] == "leaf_marginal" and cfg["protein_readout"] == arm
            assert cfg["ec_score_depth"] == 4
            assert cfg["ensemble_members"] == 1 and cfg["candidate_index_dimension"] == 4260
            assert not cfg["external_support_bank"] and not cfg["posthoc_density_correction"]
            assert not cfg["inference_requires_mining_bank"]
            assert cfg["checkpoint_selection"] == "minimum_full_validation_mean_rank"
            assert checkpoint["epoch"] == summary["best_epoch"]
            assert checkpoint["config"] == cfg
            dest = study / "frozen/models" / arm / f"seed{seed}"
            for name in ("best.pt", "config.json", "summary.json", "ec_vocabulary.json",
                         "train_frequency.json", "data_audit.json", "reaction_ec_target_audit.json",
                         "validation_embeddings.pt"):
                copy(run / name, dest / name)
            jobs.append(dict(arm=arm, seed=seed, source_run=str(run), frozen_run=str(dest),
                             selected_epoch_zero_based=checkpoint["epoch"]))
    cfg = read(Path(jobs[0]["frozen_run"]) / "config.json")
    ref_path = Path(cfg["reference_run"]) / "config.json"
    ref = read(ref_path)
    copy(ref_path, study / "frozen/reference_config.json")
    copy(ref["test_contract"], study / "frozen/test_contract.json")
    copy(ref["validation_contract"], study / "frozen/validation_contract.json")
    contract = read(study / "frozen/test_contract.json")
    assert contract["split"] == "test" and contract["copy_only"] is True
    write(study / "frozen/test_entities.json", dict(split="test", query_ids=contract["query_ids"],
                                                   candidate_ids=contract["candidate_ids"]))
    feature_dir = Path(cfg["source_dir"])
    paths = [feature_dir / "reaction_atomic_mean.pt", feature_dir / "reaction_feature_manifest.json",
             Path(ref["global_feature_bundle"]), Path(ref["global_feature_bundle"]).parent / "manifest.json",
             Path(ref["protein_global_ids"]), Path(ref["protein_global_embeddings"]),
             Path(ref["protein_global_embeddings"]).with_name("protein_mean_embeddings.report.json"),
             ROOT / "data/ReactZyme/time_split/positive_test_time.pt"]
    paths += [Path(ref["processed_dir"]) / name for name in ("pairs.parquet", "reactions.parquet", "enzymes.parquet")]
    paths += [Path(s["path"]) for s in read(feature_dir / "reaction_feature_manifest.json")["shards"]]
    residue = read(cfg["residue_cache"])
    assert residue["status"] == "completed"
    paths += [Path(cfg["residue_cache"])] + [Path(residue[k]) for k in ("cache_path","offsets_path","ids_path")]
    write(study / "input_inventory.json", [file_state(p) for p in sorted(set(paths))])
    write(study / "freeze_inventory.json", inventory)
    write(study / "plan.json", dict(status="frozen_before_new_test_scoring", created_at=now(),
        project_root=str(ROOT), training_study=str(training_study), jobs=jobs, seeds=list(SEEDS), arms=list(ARMS),
        primary=selection["primary"], secondary=selection["secondary"], negative_ablation=None,
        model_count_per_prediction=1, new_training=False, test_hyperparameter_search=False,
        checkpoint_selection="Existing minimum full VAL MR per seed; no re-selection",
        reporting="All six runs and two means plus sample SD; no seed ensemble or best-seed selection",
        encoding="Same cached frozen features and BF16 batching as training validation",
        scoring="CPU FP32, model logit scale times inner product, stable descending candidate order",
        score_query_batch_size=128, metric="official all-relevant MRR and query-macro all-positive MR",
        historical_test_previously_observed=True, test_does_not_change_primary=True,
        raw_test=str(ROOT / "data/ReactZyme/time_split/positive_test_time.pt"),
        future_reporting="TEST-only main result tables; VAL used internally for selection and diagnostics"))
    write(study / "freeze_audit.json", verify_freeze(study))
    print(json.dumps(dict(phase="freeze", models=len(jobs), artifacts=len(inventory))), flush=True)


def audit(study):
    from scripts.train_structured_retriever import load_contract
    verify_freeze(study)
    plan, ref = read(study / "plan.json"), read(study / "frozen/reference_config.json")
    processed = Path(ref["processed_dir"])
    pairs = pd.read_parquet(processed / "pairs.parquet")
    test = pairs[pairs.split == "test"]
    enzymes = pd.read_parquet(processed / "enzymes.parquet", columns=["enzyme_id", "sequence"])
    reactions = pd.read_parquet(processed / "reactions.parquet", columns=["reaction_id", "official_molecule_set_smiles"])
    seq = enzymes.set_index("enzyme_id").sequence.astype(str).to_dict()
    mol = reactions.set_index("reaction_id").official_molecule_set_smiles.astype(str).to_dict()
    raw = tensor(plan["raw_test"])
    qids, pids, relevance = load_contract(study / "frozen/test_contract.json", "test")
    sealed = read(study / "frozen/test_contract.json")
    assert qids == sealed["query_ids"] and pids == sealed["candidate_ids"]
    assert len(qids) == len(set(qids)) == 2634
    assert len(pids) == len(set(pids)) == 12277
    assert len(set(seq[p] for p in pids)) == len(pids)
    expected = {(mol[q], seq[p]) for q in qids for p in relevance[q]}
    assert expected == set(map(tuple, raw.values()))
    assert len(test) == len(raw) == len(expected) == 12287
    observed = set()
    for row in test.itertuples(index=False):
        prefix, key = row.source_pair_id.split(":")
        assert prefix == "positive_test_time"
        assert tuple(raw[int(key)]) == (mol[row.reaction_id], seq[row.enzyme_id])
        observed.add(int(key))
    assert observed == set(raw)
    train_ids = sorted(set(pairs.loc[pairs.split == "train", "reaction_id"].astype(str)))
    assert len(train_ids) == 7246 and int((pairs.split == "train").sum()) == 149557
    write(study / "protocol/train_ids.json", train_ids)
    result = dict(status="passed", completed_at=now(), queries=len(qids), candidates=len(pids),
        official_test_pairs=len(raw), missing_pairs=0, extra_pairs=0, no_query_filtering=True,
        candidate_identity="exact full sequence, official ReactZyme convention",
        duplicate_uniprot_ids_collapsed=int(test.enzyme_id.nunique()) - len(pids),
        train_pairs=int((pairs.split == "train").sum()), train_queries=len(train_ids),
        test_ec_targets_used=False, model_scores_computed=False, freeze=verify_freeze(study))
    write(study / "protocol/audit.json", result)
    print(json.dumps(result), flush=True)


@torch.inference_mode()
def encode_vectors(model, qfeatures, pfeatures, qrows, prows, device, cfg):
    from scripts.train_structured_retriever import amp_policy
    model.eval()
    enabled, dtype = amp_policy(device, cfg["amp_dtype"])
    vectors = []
    for rows, features, fn, batch_size in (
        (qrows, qfeatures, model.encode_reaction, cfg["eval_query_batch_size"]),
        (prows, pfeatures, model.encode_protein, cfg["eval_protein_batch_size"]),
    ):
        batches = []
        for start in range(0, len(rows), batch_size):
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                result = fn(features[rows[start:start + batch_size]])
            batches.append(result["embedding"].float().cpu())
        value = torch.cat(batches)
        assert value.shape == (len(rows), cfg["candidate_index_dimension"]) and torch.isfinite(value).all()
        vectors.append(value)
    return vectors


@torch.inference_mode()
def encode(study, device):
    from ec_retrieval.residue_readout import ResidueReadoutRetriever
    from ec_retrieval.residue_store import ResidueFeatureStore
    from scripts.train_candidate_coverage_retriever import load_component_features
    assert read(study / "protocol/audit.json")["status"] == "passed"
    verify_freeze(study)
    plan = read(study / "plan.json")
    cfg0 = read(Path(plan["jobs"][0]["frozen_run"]) / "config.json")
    reference = read(study / "frozen/reference_config.json")
    qids_all, pids_all, qfeatures, pfeatures, audit_data, frequency = load_component_features(
        reference, Path(cfg0["source_dir"]), device)
    qr, pr = {q:i for i,q in enumerate(qids_all)}, {p:i for i,p in enumerate(pids_all)}
    entities = read(study / "frozen/test_entities.json")
    vcontract = read(study / "frozen/validation_contract.json")
    qids, pids = entities["query_ids"], entities["candidate_ids"]
    vqids, vpids = vcontract["query_ids"], vcontract["candidate_ids"]
    del vcontract
    rows = []
    residue_store = ResidueFeatureStore(pfeatures,pids_all,Path(cfg0["residue_cache"]))
    for job in plan["jobs"]:
        run = Path(job["frozen_run"])
        dest = study / "encoded" / job["arm"] / f"seed{job['seed']}"
        dest.mkdir(parents=True, exist_ok=False)
        cfg, checkpoint = read(run / "config.json"), tensor(run / "best.pt")
        assert frequency == read(run / "train_frequency.json")
        model = ResidueReadoutRetriever(**checkpoint["model_kwargs"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == cfg["parameter_count"] == (4012966 if job["arm"] == "mean" else 4014246)
        # Reproduce stored validation embeddings, without evaluating or selecting on labels.
        vq, vp = encode_vectors(model, qfeatures, residue_store if cfg["protein_readout"] == "residue_attention" else pfeatures, [qr[q] for q in vqids], [pr[p] for p in vpids], device, cfg)
        cached = tensor(run / "validation_embeddings.pt")
        assert cached["query_ids"] == vqids and cached["candidate_ids"] == vpids
        qerr = float((vq - cached["query_embeddings"]).abs().max())
        perr = float((vp - cached["candidate_embeddings"]).abs().max())
        check = dict(query_max_abs=qerr, candidate_max_abs=perr, no_validation_scoring=True)
        write(dest / "validation_reload_check.json", check)
        if max(qerr, perr) > 1e-7:
            raise ValueError(f"Reload encoding drift: {job['arm']}/{job['seed']}: {check}")
        began = time.perf_counter()
        qv, pv = encode_vectors(model, qfeatures, residue_store if cfg["protein_readout"] == "residue_attention" else pfeatures, [qr[q] for q in qids], [pr[p] for p in pids], device, cfg)
        seconds = time.perf_counter() - began
        scale = model.logit_scale.detach().float().cpu().clamp(max=math.log(100.)).exp()
        torch.save(dict(query_ids=qids, candidate_ids=pids, query_embeddings=qv,
            candidate_embeddings=pv, logit_scale_factor=scale, split="test",
            seed=job["seed"], arm=job["arm"], checkpoint_epoch=checkpoint["epoch"],
            ground_truth_ec_input=False, mining_bank_loaded=False), dest / "embeddings.pt")
        row = dict(arm=job["arm"], seed=job["seed"], epoch=checkpoint["epoch"],
                   encoding_seconds=seconds, validation_reload=check,
                   query_shape=list(qv.shape), candidate_shape=list(pv.shape))
        rows.append(row)
        print(json.dumps(dict(phase="encode", **row)), flush=True)
        del model, checkpoint, cached, vq, vp
    write(study / "encoded/audit.json", dict(status="completed", completed_at=now(), models=rows,
        optimizer_steps=0, test_relevance_used_by_encoder=False, true_ec_labels_loaded=False,
        mining_bank_loaded=False, training_frequency_exactly_reproduced=True,
        feature_audit=audit_data, device=str(device), torch_version=torch.__version__,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
        freeze=verify_freeze(study)))


def independent_verify(q, p, scale, qids, pids, relevance, metrics, predictions):
    path = ROOT / "external/ReactZyme_official/retrieval.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "topk_accuracy")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    official = namespace["topk_accuracy"]
    total = np.zeros(18, dtype=np.float64)
    index = {v:i for i,v in enumerate(pids)}
    for start in range(0, len(qids), 128):
        scores = scale * (q[start:start+128] @ p.T)
        assert torch.isfinite(scores).all()
        labels = torch.zeros_like(scores, dtype=torch.bool)
        for i, qid in enumerate(qids[start:start+128]):
            labels[i, [index[v] for v in relevance[qid]]] = True
        total += np.asarray([float(v) for v in official(scores, labels)]) * len(scores)
        order = np.argsort(-scores.numpy(), axis=1, kind="stable")
        inverse = np.empty_like(order)
        np.put_along_axis(inverse, order, np.broadcast_to(np.arange(len(pids)), order.shape), axis=1)
        for i, qid in enumerate(qids[start:start+128]):
            ranks = sorted((inverse[i, labels[i].numpy()] + 1).tolist())
            assert predictions[start+i]["query_id"] == qid
            assert predictions[start+i]["positive_ranks"] == ranks
    names = [f"precision@{k}" for k in (1,2,3,4,5,10,20,50)] + [f"hit@{k}" for k in (1,2,3,4,5,10,20,50)] + ["mean_rank", "all_mrr"]
    results = dict(zip(names, (total / len(qids)).tolist()))
    errors = {}
    for name, key in (("mean_rank", "official_reactzyme_mean_rank"), ("all_mrr", "official_reactzyme_mrr_all_relevant")):
        errors[name] = abs(results[name] - metrics[key])
        assert errors[name] < (2e-5 if name == "mean_rank" else 2e-7), errors
    for k in (1,5,10,20,50):
        assert abs(results[f"hit@{k}"] - metrics[f"hit@{k}"]) < 2e-7
        assert abs(results[f"precision@{k}"] - metrics[f"official_reactzyme_acc_n_at_{k}"]) < 2e-7
    return dict(status="passed", official_metrics=results, numeric_aggregation_difference=errors,
                independent_numpy_positive_rank_rows=len(qids), scale_included=True)


@torch.inference_mode()
def evaluate(study):
    from scripts.train_structured_retriever import load_contract, rank_metrics
    from scripts.train_function_transfer_retriever import cohort_metrics
    assert read(study / "encoded/audit.json")["status"] == "completed"
    verify_freeze(study)
    plan = read(study / "plan.json")
    qids, pids, relevance = load_contract(study / "frozen/test_contract.json", "test")
    train_ids = set(read(study / "protocol/train_ids.json"))
    rows, query_rows = [], []
    for job in plan["jobs"]:
        dest = study / "test" / job["arm"] / f"seed{job['seed']}"
        dest.mkdir(parents=True, exist_ok=False)
        cached = tensor(study / "encoded" / job["arm"] / f"seed{job['seed']}" / "embeddings.pt")
        assert cached["split"] == "test" and cached["query_ids"] == qids and cached["candidate_ids"] == pids
        q, p, scale = cached["query_embeddings"], cached["candidate_embeddings"], cached["logit_scale_factor"]
        metrics, ranks = rank_metrics(q, p, qids, pids, relevance, torch.device("cpu"), 128, 20,
                                     score_function=lambda a, b: scale * (a @ b.T))
        assert metrics["query_count"] == 2634 and metrics["candidate_count"] == 12277
        metrics.update(cohort_metrics(ranks, train_ids))
        write(dest / "metrics.json", metrics)
        with (dest / "test_topk.jsonl").open("w") as f:
            for r in ranks:
                f.write(json.dumps(r) + "\n")
        verified = independent_verify(q, p, scale, qids, pids, relevance, metrics, ranks)
        write(dest / "independent_verification.json", verified)
        row = dict(arm=job["arm"], seed=job["seed"], epoch_one_based=cached["checkpoint_epoch"]+1,
                   mean_rank=metrics["official_reactzyme_mean_rank"], all_mrr=metrics["official_reactzyme_mrr_all_relevant"],
                   seen_mean_rank=metrics["train_seen_mean_rank"], unseen_mean_rank=metrics["train_unseen_mean_rank"])
        row.update({f"hit{k}":metrics[f"hit@{k}"] for k in (1,5,10,20)})
        rows.append(row)
        for item in ranks:
            r = np.asarray(item["positive_ranks"], dtype=float)
            query_rows.append(dict(arm=job["arm"], seed=job["seed"], query_id=item["query_id"],
                mean_rank=float(r.mean()), all_mrr=float((1/r).mean()), hit1=float(r.min()==1),
                positive_count=len(r), train_seen=item["query_id"] in train_ids))
        print(json.dumps(dict(phase="evaluate", verified=True, **row)), flush=True)
    pd.DataFrame(rows).to_csv(study / "test/per_seed_results.csv", index=False)
    pd.DataFrame(query_rows).to_parquet(study / "test/per_query_metrics.parquet", index=False)
    means = []
    keys = ("hit1", "hit5", "hit10", "hit20", "all_mrr", "mean_rank", "seen_mean_rank", "unseen_mean_rank")
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        result = dict(arm=arm, seeds=len(group))
        for key in keys:
            values = [r[key] for r in group]
            result[key+"_mean"] = statistics.mean(values)
            result[key+"_sd"] = statistics.stdev(values)
        means.append(result)
    pd.DataFrame(means).to_csv(study / "test/mean_sd.csv", index=False)
    write(study / "test/summary.json", dict(status="completed", completed_at=now(), split="test",
        primary=plan["primary"], secondary=plan["secondary"], negative_ablation=plan["negative_ablation"],
        means=means, runs=rows, query_count=len(qids), candidate_count=len(pids),
        all_six_official_metrics_and_numpy_ranks_verified=True, new_training=False,
        test_used_for_selection=False, historical_test_previously_observed=True,
        freeze_after_evaluation=verify_freeze(study)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "audit", "encode", "evaluate"), required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--training-study", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    study = args.study_dir.resolve()
    if args.phase == "freeze":
        assert args.training_study is not None
        freeze(study, args.training_study.resolve())
        return
    assert ROOT == study / "frozen/source", "Run the frozen copy for evaluation phases"
    assert read(study / "plan.json")["status"] == "frozen_before_new_test_scoring"
    if args.phase == "audit":
        audit(study)
    elif args.phase == "encode":
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        encode(study, device)
    else:
        evaluate(study)


if __name__ == "__main__":
    main()
