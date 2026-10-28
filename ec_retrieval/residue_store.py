"""Vectorized shared-mmap residue gather; original tensors remain immutable."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


class ResidueFeatureStore:
    def __init__(self,mean_features,protein_ids,manifest_path):
        self.mean=mean_features;self.device=mean_features.device;self.shape=mean_features.shape
        self.manifest_path=Path(manifest_path).resolve()
        m=json.loads(self.manifest_path.read_text());self.manifest=m
        view = m if 'source_cache_manifest' in m else None
        if view:
            m=json.loads(Path(view['source_cache_manifest']).read_text())
        ids=json.loads(Path(m['ids_path']).read_text())['enzyme_ids']
        if m['status']!='completed' or m['mean_reconstruction_checked_rows']!=len(ids):
            raise ValueError('Residue cache was not completely verified')
        contract=m['feature_contract']
        if any(contract[k] for k in ['reads_pair_labels','reads_ec_labels','reads_split_labels']):
            raise ValueError('Residue cache violates the label-independent feature contract')
        if not contract['retains_baseline_residue_window']:
            raise ValueError('Residue input window differs from the mean control')
        if view:
            target_ids=json.loads(Path(view['ids_path']).read_text())['enzyme_ids']
            row_map=np.load(view['row_mapping_path'])
            identity=json.loads(Path(view['identity_audit']).read_text())
            if (target_ids!=list(protein_ids) or row_map.shape!=(len(target_ids),) or
                    row_map.min()<0 or row_map.max()>=len(ids) or identity['status']!='passed' or
                    identity['exact_sequence_checked_rows']!=len(target_ids)):
                raise ValueError('Invalid exact-sequence residue view')
        else:
            if ids!=list(protein_ids):raise ValueError('Residue cache candidate IDs differ')
            row_map=np.arange(len(ids))
        self.offsets=np.load(m['offsets_path']);shape=tuple(m['shape'])
        if len(self.offsets)!=len(ids)+1 or self.offsets[0]!=0 or self.offsets[-1]!=shape[0]:
            raise ValueError('Residue cache boundary mismatch')
        if np.any(np.diff(self.offsets)<=0) or np.max(np.diff(self.offsets))>m['maximum_residues']:
            raise ValueError('Residue cache has an invalid length')
        if Path(m['cache_path']).stat().st_size!=int(np.prod(shape))*2:
            raise ValueError('Residue cache file size mismatch')
        self.values=torch.from_file(m['cache_path'],shared=False,size=int(np.prod(shape)),dtype=torch.float16).view(shape)
        self.starts=self.offsets[row_map]
        self.lengths=np.diff(self.offsets)[row_map]
        self.buffer=None;self.copy_event=None

    def __getitem__(self,rows):
        if isinstance(rows,torch.Tensor):rows=rows.detach().cpu().numpy()
        rows=np.asarray(rows,dtype=np.int64)
        starts=self.starts[rows];lengths=self.lengths[rows]
        ptr=np.r_[0,np.cumsum(lengths)];n=int(ptr[-1])
        positions=np.arange(n,dtype=np.int64)+np.repeat(starts-ptr[:-1],lengths)
        take=torch.from_numpy(positions)
        if self.device.type=='cuda':
            # A persistent pinned staging buffer removes per-step pin/copy cost.
            if self.copy_event is not None:self.copy_event.synchronize()
            if self.buffer is None or self.buffer.shape[0]<n:
                # This persistent buffer crosses TRAIN -> VAL -> TRAIN. VAL
                # can grow it inside inference_mode, but the next TRAIN batch
                # must still be allowed to fill it in place.
                with torch.inference_mode(False):
                    self.buffer=torch.empty((n,self.shape[1]),dtype=torch.float16,pin_memory=True)
            torch.index_select(self.values,0,take,out=self.buffer[:n])
            packed=self.buffer[:n].to(self.device,non_blocking=True)
            self.copy_event=torch.cuda.Event();self.copy_event.record()
        else:
            packed=self.values.index_select(0,take).to(self.device)
        return {'mean':self.mean[rows.tolist()],'values':packed,'offsets_cpu':ptr.tolist()}
