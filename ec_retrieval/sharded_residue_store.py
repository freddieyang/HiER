"""Immutable TRAIN-only residue cache spread over explicitly budgeted GPUs.

Only loading changes: original FP16 values, candidate order and model inputs
are preserved. A bounded producer gathers cache hits on their owning GPU and
uploads misses from pinned CPU memory. The original synchronous VAL path is
inherited. No peer-access capability is assumed and no RNG is consumed.
"""
from __future__ import annotations

from queue import Full
import time

import numpy as np
import torch

from .prefetch_residue_store import PrefetchResidueFeatureStore, _ResiduePrefetch


def training_cache_priority(adjacency, row_count, positives_per_query):
    """Expected direct-positive access, fitted exclusively from TRAIN sampling.

    Every query is visited once per epoch. Random denominator filling adds an
    approximately common background, so ranking by direct exposure is simple
    and reproducible. This policy does not change the sampler or model inputs.
    """
    priority=np.zeros(row_count,dtype=np.float64)
    for values in adjacency.values():
        if not values:raise ValueError('TRAIN queries need positive candidates')
        np.add.at(priority,np.asarray(values,dtype=np.int64),
                  min(positives_per_query,len(values))/len(values))
    return priority


class ShardedResidueFeatureStore(PrefetchResidueFeatureStore):
    def __init__(self, mean_features, protein_ids, manifest_path, *, train_rows,
                 priority, cache_budgets, main_headroom_bytes=10*2**30,
                 helper_headroom_bytes=2*2**30, load_chunk_residues=65536):
        super().__init__(mean_features,protein_ids,manifest_path)
        if self.device.type!='cuda':raise ValueError('Resident cache requires CUDA')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        rows=np.asarray(train_rows,dtype=np.int64)
        priority=np.asarray(priority,dtype=np.float64)
        if rows.ndim!=1 or not len(rows) or not np.array_equal(rows,np.unique(rows)):
            raise ValueError('Sorted unique TRAIN candidate rows required')
        if rows.min()<0 or rows.max()>=self.shape[0] or priority.shape!=(self.shape[0],):
            raise ValueError('Cache rows and priorities must match the feature table')
        if not np.isfinite(priority).all() or (priority<0).any():
            raise ValueError('Finite nonnegative TRAIN cache priorities required')
        if not cache_budgets or len({int(d) for d,_ in cache_budgets})!=len(cache_budgets):
            raise ValueError('Explicit distinct CUDA devices and byte budgets required')
        if min(main_headroom_bytes,helper_headroom_bytes)<0 or load_chunk_residues<1:
            raise ValueError('Invalid cache headroom or preload chunk size')
        self.owners=np.full(self.shape[0],-1,dtype=np.int32)
        self.resident_starts=np.full(self.shape[0],-1,dtype=np.int64)
        self.shards={}
        self.counts=dict(batches=0,candidate_rows=0,residue_rows=0,
                         cached_candidate_rows=0,cached_residue_rows=0,
                         cpu_residue_rows=0,local_residue_rows=0,remote_residue_rows=0)
        self.device_counts={}
        lengths=self.lengths
        remaining=rows[np.lexsort((rows,-priority[rows]))]
        plans=[]
        for raw_device,raw_budget in cache_budgets:
            device=int(raw_device);budget=int(raw_budget)
            if not 0<=device<torch.cuda.device_count() or budget<=0:
                raise ValueError('Positive byte budgets on available CUDA devices required')
            free,total=torch.cuda.mem_get_info(device)
            headroom=main_headroom_bytes if device==self.device.index else helper_headroom_bytes
            if budget+headroom>free:
                raise MemoryError(f'cuda:{device} needs {budget/2**30:.2f} GiB cache plus '
                                  f'{headroom/2**30:.2f} GiB headroom; free={free/2**30:.2f} GiB')
            capacity=budget//(self.shape[1]*2)
            cumulative=np.cumsum(lengths[remaining],dtype=np.int64)
            number=int(np.searchsorted(cumulative,capacity,side='right'))
            chosen=np.sort(remaining[:number]);remaining=remaining[number:]
            pointer=np.r_[0,np.cumsum(lengths[chosen],dtype=np.int64)]
            plans.append(dict(device=device,rows=chosen,pointer=pointer,
                              budget_bytes=budget,free_before_bytes=free,total_bytes=total))
        if not any(len(p['rows']) for p in plans):
            raise ValueError('Cache budgets cannot hold any TRAIN protein')
        began=time.perf_counter()
        for plan in plans:
            device=plan['device'];chosen=plan['rows'];pointer=plan['pointer']
            if not len(chosen):continue
            with torch.cuda.device(device),torch.inference_mode(False),torch.no_grad():
                resident=torch.empty((int(pointer[-1]),self.shape[1]),device=f'cuda:{device}',dtype=torch.float16)
                host=torch.empty((load_chunk_residues,self.shape[1]),dtype=torch.float16,pin_memory=True)
                # Preload bounded chunks once, outside forward/backward.
                for start in range(0,len(resident),load_chunk_residues):
                    stop=min(start+load_chunk_residues,len(resident))
                    target=np.arange(start,stop,dtype=np.int64)
                    protein=np.searchsorted(pointer[1:],target,side='right')
                    source=self.starts[chosen[protein]]+target-pointer[protein]
                    torch.index_select(self.values,0,torch.from_numpy(source),out=host[:len(target)])
                    resident[start:stop].copy_(host[:len(target)],non_blocking=True)
                    # Keep the single preload host buffer alive until its copy completes.
                    torch.cuda.current_stream(device).synchronize()
                self.shards[device]=resident
            self.owners[chosen]=device
            self.resident_starts[chosen]=pointer[:-1]
            self.device_counts[str(device)]=0
        self.cache_plan=dict(policy='TRAIN expected direct-positive exposure; row ID breaks ties',
            labels_as_model_inputs=False,training_splits=['train'],
            cached_candidates=int((self.owners>=0).sum()),
            cached_bytes=sum(v.numel()*v.element_size() for v in self.shards.values()),
            total_train_candidates=len(rows),
            total_train_bytes=int(lengths[rows].sum())*self.shape[1]*2,
            preload_seconds=time.perf_counter()-began,
            devices=[dict(device=p['device'],budget_bytes=p['budget_bytes'],
                cached_candidates=len(p['rows']),cached_residues=int(p['pointer'][-1]),
                cached_bytes=int(p['pointer'][-1])*self.shape[1]*2,
                free_before_bytes=p['free_before_bytes'],total_bytes=p['total_bytes'],
                peer_access_to_training_device=(p['device']==self.device.index or
                    torch.cuda.can_device_access_peer(p['device'],self.device.index))) for p in plans])
        if not np.isin(np.flatnonzero(self.owners>=0),rows).all():
            raise AssertionError('Non-TRAIN candidate entered the resident cache')

    def snapshot(self):
        return dict(plan=self.cache_plan,counts=dict(self.counts),
                    cached_residue_rows_by_device=dict(self.device_counts))

    def prefetch(self,row_batches,*,depth=2):
        if depth not in (1,2):raise ValueError('Prefetch depth must be 1 or 2')
        return _ShardedPrefetch(self,row_batches,depth)


class _ShardedPrefetch(_ResiduePrefetch):
    def _produce(self):
        try:
            store=self.store
            with torch.cuda.device(store.device),torch.inference_mode(False),torch.no_grad():
                # Each stream is created and used by this producer only. Preload
                # completed before construction; shard tensors stay immutable.
                streams={d:torch.cuda.Stream(device=d) for d in store.shards if d!=store.device.index}
                for index,values in enumerate(self.rows):
                    while not self.stop.is_set():
                        if self.capacity.acquire(timeout=.05):break
                    if self.stop.is_set():return
                    if isinstance(values,torch.Tensor):
                        if values.device.type!='cpu':raise ValueError('CPU row metadata required')
                        values=values.detach().numpy()
                    rows=np.array(values,dtype=np.int64,copy=True)
                    if rows.ndim!=1 or not len(rows) or rows.min()<0 or rows.max()>=store.shape[0]:
                        raise ValueError('Nonempty valid one-dimensional candidate rows required')
                    lengths=store.lengths[rows]
                    pointer=np.r_[0,np.cumsum(lengths)];owners=store.owners[rows]
                    slot=index%2
                    if self.host_events[slot] is not None:self.host_events[slot].synchronize()
                    groups=[]
                    # At most one group per cache GPU plus the CPU misses; no
                    # per-protein or per-residue Python loop in the hot path.
                    for owner in [-1,*store.shards]:
                        take=np.flatnonzero(owners==owner)
                        if not len(take):continue
                        count=int(lengths[take].sum())
                        local_ptr=np.r_[0,np.cumsum(lengths[take])]
                        local=np.arange(count,dtype=np.int64)-np.repeat(local_ptr[:-1],lengths[take])
                        destination=local+np.repeat(pointer[take],lengths[take])
                        starts=(store.starts[rows[take]] if owner==-1 else store.resident_starts[rows[take]])
                        positions=local+np.repeat(starts,lengths[take])
                        groups.append((owner,count,destination,positions))
                        if owner==-1:
                            host=self.host_buffers[slot]
                            if host is None or len(host)<count:
                                host=torch.empty((count,store.shape[1]),dtype=torch.float16,pin_memory=True)
                                self.host_buffers[slot]=host
                            torch.index_select(store.values,0,torch.from_numpy(positions),out=host[:count])
                    if self.stop.is_set():return
                    with torch.cuda.stream(self.stream):
                        packed=torch.empty((int(pointer[-1]),store.shape[1]),dtype=torch.float16,device=store.device)
                        for owner,count,destination,positions in groups:
                            target=torch.from_numpy(destination).to(store.device)
                            if owner==-1:
                                gathered=self.host_buffers[slot][:count].to(store.device,non_blocking=True)
                            elif owner==store.device.index:
                                source=torch.from_numpy(positions).to(store.device)
                                gathered=store.shards[owner].index_select(0,source)
                            else:
                                with torch.cuda.stream(streams[owner]):
                                    source=torch.from_numpy(positions).to(f'cuda:{owner}')
                                    gathered=store.shards[owner].index_select(0,source).to(store.device,non_blocking=True)
                            packed.index_copy_(0,target,gathered)
                        mean=store.mean[rows.tolist()]
                        event=torch.cuda.Event();event.record(self.stream)
                    self.host_events[slot]=event
                    hits=owners>=0
                    store.counts['batches']+=1
                    store.counts['candidate_rows']+=len(rows)
                    store.counts['residue_rows']+=int(pointer[-1])
                    store.counts['cached_candidate_rows']+=int(hits.sum())
                    store.counts['cached_residue_rows']+=int(lengths[hits].sum())
                    store.counts['cpu_residue_rows']+=int(lengths[~hits].sum())
                    store.counts['local_residue_rows']+=int(lengths[owners==store.device.index].sum())
                    store.counts['remote_residue_rows']+=int(lengths[hits & (owners!=store.device.index)].sum())
                    for owner in store.shards:
                        store.device_counts[str(owner)]+=int(lengths[owners==owner].sum())
                    payload=({'mean':mean,'values':packed,'offsets_cpu':pointer.tolist()},event)
                    while not self.stop.is_set():
                        try:self.ready.put(payload,timeout=.05);break
                        except Full:pass
                    if self.stop.is_set():return
        except BaseException as error:
            self.error=error
        finally:
            self.finished.set()
