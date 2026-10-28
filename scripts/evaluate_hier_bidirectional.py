#!/usr/bin/env python3
"""Freeze a VAL-selected HiER checkpoint and evaluate both official directions.

Each direction ranks the same full score matrix (or its transpose). No model
parameter, checkpoint, score weight, candidate filter or EC label is selected
using TEST. Published baselines are separate from these actual model runs.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from datetime import datetime
import filecmp
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_structured_retriever import atomic_json, load_contract, rank_metrics
from scripts.evaluate_residue_readout_test import encode_vectors
from ec_retrieval.candidate_background import make_residue_retriever, validation_selection_key


def read(path):
    return json.loads(Path(path).read_text())


def official_function():
    path = ROOT / 'external/ReactZyme_official/retrieval.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'topk_accuracy')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['topk_accuracy']


def score_direction(scores, qids, pids, relevance, output):
    output.mkdir(parents=True, exist_ok=False)
    # Pass row numbers to the existing audited ranker so both directions use
    # EXACTLY the same matrix, including all floating-point ties.
    metrics, ranks = rank_metrics(torch.arange(len(qids))[:,None], torch.arange(len(pids))[:,None],
        qids, pids, relevance, torch.device('cpu'), 128, 20,
        score_function=lambda a, b: scores[a[:,0].long()])
    official = official_function()
    total = np.zeros(18, dtype=np.float64)
    column = {p:i for i,p in enumerate(pids)}
    for start in range(0, len(qids), 128):
        block = scores[start:start+128]
        labels = torch.zeros_like(block, dtype=torch.bool)
        for i, q in enumerate(qids[start:start+128]):
            labels[i, [column[p] for p in relevance[q]]] = True
        values = np.asarray([float(x) for x in official(block, labels)])
        total += values * len(block)
        order = np.argsort(-block.numpy(), axis=1, kind='stable')
        inverse = np.empty_like(order)
        np.put_along_axis(inverse, order, np.broadcast_to(np.arange(len(pids)), order.shape), axis=1)
        for i in range(len(block)):
            assert ranks[start+i]['positive_ranks'] == sorted((inverse[i, labels[i].numpy()]+1).tolist())
    values = total/len(qids)
    mr_error = abs(values[-2]-metrics['official_reactzyme_mean_rank'])
    mrr_error = abs(values[-1]-metrics['official_reactzyme_mrr_all_relevant'])
    # The official function averages float32 ranks. Allow its float32 rounding
    # at large MR, while independently requiring EVERY NumPy rank to be exact.
    mr_tolerance = max(2e-5, 2e-7 * abs(metrics['official_reactzyme_mean_rank']))
    assert mr_error < mr_tolerance and mrr_error < 2e-7, (mr_error, mrr_error)
    for index, k in enumerate((1,2,3,4,5,10,20,50)):
        if k in (1,5,10,20,50):
            assert abs(values[index+8]-metrics[f'hit@{k}']) < 2e-7
    atomic_json(output / 'metrics.json', metrics)
    with (output / 'per_query_ranks.jsonl').open('w') as handle:
        for record in ranks:
            # Direction-neutral names avoid labeling a reaction as an enzyme.
            record['positive_candidate_ids'] = record.pop('positive_enzyme_ids')
            record['top_candidate_ids'] = record.pop('top_enzyme_ids')
            handle.write(json.dumps(record)+'\n')
    atomic_json(output / 'verification.json', dict(status='passed', official_source=str(
        ROOT / 'external/ReactZyme_official/retrieval.py'), official_mr_error=mr_error,
        official_mrr_error=mrr_error, official_mr_float32_tolerance=mr_tolerance,
        numpy_verified_queries=len(qids),
        same_matrix_transposed=True, score_tie_break='stable candidate-ID order'))
    return dict(hit1=metrics['hit@1'], mrr=metrics['official_reactzyme_mrr_all_relevant'],
        mr=metrics['official_reactzyme_mean_rank'], query_count=len(qids), candidate_count=len(pids),
        **{f'hit{k}':metrics[f'hit@{k}'] for k in (5,10,20)})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=('time','seq_smi','mol_smi'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-contract', type=Path, help='Defaults to the prepared split referenced by the run')
    parser.add_argument('--save-scores', action='store_true', help='Save the full score matrix as scores.npy')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected-epochs', type=int, choices=(100,200), default=100,
                        help='Declared complete training budget; does not change scoring')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.run = args.run.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = read(args.run / 'config.json')
    if cfg['seed'] != args.seed or cfg['training_splits'] != ['train'] or cfg['is_smoke']:
        raise ValueError('Only completed TRAIN-only formal seeds may be scored')
    summary = read(args.run / 'summary.json')
    if summary['status'] != 'completed' or summary['epochs_completed'] != args.expected_epochs or cfg['epochs'] != args.expected_epochs or not summary['checkpoint_reload_verified']:
        raise ValueError('Declared full training budget and reload verification required')
    frozen = out / 'frozen'
    frozen.mkdir()
    for name in ('best.pt','config.json','summary.json','ec_vocabulary.json','train_frequency.json'):
        shutil.copyfile(args.run / name, frozen / name)
        assert filecmp.cmp(args.run / name, frozen / name, shallow=False)
    cp = torch.load(frozen / 'best.pt', map_location='cpu', weights_only=False)
    vals = [json.loads(line) for line in (args.run / 'validation_metrics.jsonl').read_text().splitlines()]
    policy = cfg.get('checkpoint_selection_policy', 'mean_rank')
    assert len(vals) == args.expected_epochs and cp['epoch'] == min(vals, key=lambda x:validation_selection_key(x, policy))['epoch']
    atomic_json(out / 'freeze.json', dict(status='frozen_before_test_scoring',
        timestamp=datetime.now().astimezone().isoformat(), run=str(args.run), seed=args.seed, split=args.split,
        selection=cfg['checkpoint_selection'] + '; same checkpoint both directions',
        checkpoint_epoch_zero_based=cp['epoch'], test_used_for_selection=False,
        file_copy_verified=True))
    ref = read(Path(cfg['reference_run']) / 'config.json')
    if ref.get('benchmark_split') != args.split:
        raise ValueError('The requested split must match the prepared run reference')
    contract_path = args.test_contract or Path(ref['processed_dir']) / 'test_contract.json'
    # Open TEST relevance only after freezing the validation-selected weights.
    qids, pids, relevance = load_contract(contract_path, 'test')
    from scripts.train_cached_residue_readout import load_component_features
    from ec_retrieval.residue_store import ResidueFeatureStore
    from ec_retrieval.residue_readout import ResidueReadoutRetriever
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    all_q, all_p, qfeatures, mean, audit, frequency = load_component_features(ref, Path(cfg['source_dir']), device)
    assert frequency == read(args.run / 'train_frequency.json')
    pfeatures = ResidueFeatureStore(mean, all_p, Path(cfg['residue_cache']))
    qr, pr = {v:i for i,v in enumerate(all_q)}, {v:i for i,v in enumerate(all_p)}
    model = make_residue_retriever(cp['model_kwargs']).to(device)
    model.load_state_dict(cp['model_state'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == cfg['parameter_count']
    qv, pv = encode_vectors(model, qfeatures, pfeatures, [qr[q] for q in qids],
        [pr[p] for p in pids], device, cfg)
    scale = model.logit_scale.detach().float().cpu().clamp(max=math.log(100.)).exp()
    torch.save(dict(query_ids=qids,candidate_ids=pids,query_embeddings=qv,candidate_embeddings=pv,
        logit_scale_factor=scale,split='test',seed=args.seed,checkpoint_epoch=cp['epoch'],
        ground_truth_ec_input=False), out / 'embeddings.pt')
    atomic_json(out / 'encoding.json', dict(feature_audit=audit, encoder_input_ec=False,
        train_statistics_match_frozen_checkpoint=True, trainable_parameters=cfg['parameter_count']))
    del model, qfeatures, pfeatures, mean
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Use the same float32 CPU scoring and batch size for both directions.
    scores = torch.cat([scale*(qv[s:s+128] @ pv.T) for s in range(0, len(qids), 128)])
    assert scores.shape == (len(qids), len(pids)) and torch.isfinite(scores).all()
    if args.save_scores:
        np.save(out / 'scores.npy', scores.numpy())
    reverse = defaultdict(set)
    for q, ps in relevance.items():
        for p in ps:
            reverse[p].add(q)
    assert set(reverse) == set(pids)
    results = []
    for direction, matrix, query, candidates, truth in (
        ('R-E',scores,qids,pids,relevance), ('E-R',scores.T,pids,qids,reverse)):
        values = score_direction(matrix,query,candidates,truth,out/direction)
        results.append(dict(split=args.split, direction=direction, seed=args.seed,
            epoch_one_based=cp['epoch']+1, **values))
        print(json.dumps(results[-1]), flush=True)
    assert filecmp.cmp(args.run/'best.pt',frozen/'best.pt',shallow=False)
    atomic_json(out/'summary.json',dict(status='completed', reporting_split='test', runs=results,
        test_used_for_selection=False, directions_share_checkpoint=True, directions_share_score_matrix=True,
        all_positive_ranks_verified=True, checkpoint_unchanged=True))


if __name__ == '__main__':
    main()
