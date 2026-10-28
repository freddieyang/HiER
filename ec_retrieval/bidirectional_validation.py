"""VAL-only, equal-direction checkpoint objectives; no model updates or TEST."""
from collections import defaultdict
import math

import torch


def joint_head_values(metrics):
    """Geometric means give the two retrieval directions equal log weights."""
    keys = ('hit@1', 'e2r_hit@1', 'official_reactzyme_mrr', 'e2r_official_reactzyme_mrr')
    values = [float(metrics[k]) for k in keys]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Both directions require finite Hit1/MRR in [0,1]')
    return dict(joint_hit1_geomean=math.sqrt(values[0]*values[1]),
                joint_mrr_geomean=math.sqrt(values[2]*values[3]))


@torch.inference_mode()
def add_reverse_validation(metrics, model, qv, pv, qids, pids, relevance, device, batch_size=128):
    """Reuse encoded VAL entities; transpose the complete forward score matrix.

    The trainer's original R-E metrics remain unchanged. Matrix products use
    its original query batching and FP32 scoring. No encoder is rerun and no
    per-enzyme train-time labels enter the model.
    """
    from scripts.train_structured_retriever import rank_metrics

    if model.training:
        raise ValueError('Validation requires model.eval()')
    proteins = pv.to(device)  # [P,D]
    scores = torch.cat([model.score_embeddings(qv[s:s+batch_size].to(device), proteins)
                        for s in range(0, len(qids), batch_size)])  # [Q,P]
    if scores.shape != (len(qids),len(pids)) or not torch.isfinite(scores).all():
        raise ValueError('Invalid full validation scores')
    reverse = defaultdict(set)
    for q in qids:
        for p in relevance[q]:
            reverse[p].add(q)
    if set(reverse) != set(pids):
        raise ValueError('VAL candidate domain must match the reverse query domain')
    values, _ = rank_metrics(torch.arange(len(pids))[:,None],torch.arange(len(qids))[:,None],
        pids,qids,reverse,device,batch_size,0,
        score_function=lambda query, _:scores[:,query[:,0].long()].T)
    result = dict(metrics)
    for key in ('hit@1','official_reactzyme_mrr','official_reactzyme_mean_rank','query_count','candidate_count'):
        result['e2r_'+key] = values[key]
    result.update(joint_head_values(result))
    return result
