"""Keep the original training prefix, then hold its terminal learning rate."""
import json
import random
from pathlib import Path
import numpy as np
import torch


def step_prefix_scheduler(scheduler, epoch, schedule_epochs):
    if epoch < schedule_epochs:
        scheduler.step()


def save_training_state(path, *, model, optimizer, scheduler, scaler, epoch,
        best_epoch, best_metrics, rng, negative_rng, hard_rng, reverse_pool,
        exposure, query_orders, config):
    """Full training state before reloading the VAL-selected best checkpoint."""
    from scripts.train_global_anchored_retriever import atomic_torch_save
    atomic_torch_save(path,dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),scaler_state=scaler.state_dict(),epoch=epoch,
        best_epoch=best_epoch,best_validation_metrics=best_metrics,config=config,
        random_state=random.getstate(),numpy_state=np.random.get_state(),torch_rng_state=torch.get_rng_state(),
        cuda_rng_states=torch.cuda.get_rng_state_all(),query_rng_state=rng.bit_generator.state,
        negative_rng_state=negative_rng.bit_generator.state,hard_rng_state=hard_rng.bit_generator.state,
        reverse_rng_state=reverse_pool.rng.bit_generator.state if reverse_pool else None,
        direct_positive_exposures=exposure,query_orders=query_orders,training_splits=['train']))


def verify_prefix(output, control, epochs, exposure, query_orders, domain):
    """Require exact replay of an already-completed control before continuation."""
    from scripts.train_structured_retriever import atomic_json
    read=lambda p:json.loads(p.read_text())
    rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    checks={}
    for filename in ('train_metrics.jsonl','validation_metrics.jsonl'):
        a,b=rows(output/filename),rows(control/filename)
        keys=('loss','retrieval','r2e_retrieval','e2r_retrieval','ec','alignment','consistency','morgan_drop_probability') if filename.startswith('train') else (
            'hit@1','official_reactzyme_mrr','official_reactzyme_mean_rank','e2r_hit@1',
            'e2r_official_reactzyme_mrr','e2r_official_reactzyme_mean_rank','joint_hit1_geomean','joint_mrr_geomean')
        checks[filename]=len(a)==epochs and len(b)>=epochs and all(x[k]==y[k] for x,y in zip(a,b[:epochs]) for k in keys)
    for filename in ('initial_state.pt','best.pt'):
        a,b=load(output/filename),load(control/filename)
        if filename=='best.pt':
            checks['best_epoch']=a['epoch']==b['epoch'];a,b=a['model_state'],b['model_state']
        checks[filename]=a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    old=load(control/'sampling_trace.pt')
    checks['query_orders']=torch.equal(torch.stack(query_orders),old['query_orders'][:epochs])
    checks['positive_exposures']=all(np.array_equal(exposure[int(q)],v.numpy()) for q,v in zip(domain.query_rows,old['direct_positive_exposures']))
    for filename in ('train_frequency.json','ec_vocabulary.json','data_audit.json'):
        checks[filename]=read(output/filename)==read(control/filename)
    checks['ec_targets']=load(output/'training_ec_targets.pt')==load(control/'training_ec_targets.pt')
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,epochs=epochs,
                control=str(control),test_used=False)
    atomic_json(output/'prefix_equivalence.json',result)
    if not all(checks.values()):
        raise ValueError(f'Original training prefix changed: {result}')
