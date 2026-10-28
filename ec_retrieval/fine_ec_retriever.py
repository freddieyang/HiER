"""Fine EC evidence with one coherent posterior and a valid MAP path.

The existing top-down predictor proposes a leaf posterior. Ancestor evidence
is obtained by marginalizing that SAME posterior, so retrieval cannot combine
unrelated coarse and fine class distributions. This adds no learned module.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch_scatter import segment_csr

from .explicit_ec_retriever import ExplicitECFunctionalRetriever, ProbabilityOnlyRouter


def grouped_log_marginal(log_probability, parent_index, sort_index, pointer):
    """[B,K_child] -> [B,K_parent], stable deterministic CSR logsumexp."""
    values=log_probability.float().index_select(1,sort_index)
    maximum=segment_csr(values.detach(),pointer.unsqueeze(0),reduce='max')
    shifted=log_probability.float()-maximum.index_select(1,parent_index)
    total=segment_csr(shifted.index_select(1,sort_index).exp(),pointer.unsqueeze(0),reduce='sum')
    return maximum+total.clamp_min(torch.finfo(total.dtype).tiny).log()


class LeafMarginalRouter(ProbabilityOnlyRouter):
    """Reuse every predictor weight; expose marginals of its final posterior."""
    def __init__(self,full):
        nn.Module.__init__(self)
        if full.normalization!='parent_prior':
            raise ValueError('The controlled proposal predictor uses parent_prior')
        self.vocab_sizes=full.vocab_sizes
        self.normalization=full.normalization
        self.state_norm=full.state_norm
        self.heads=full.heads
        self.prefix_embeddings=full.prefix_embeddings
        for level,mapping in enumerate(full.parent_maps(),1):
            self.register_buffer(f'parent_index_{level}',mapping.clone())
            counts=torch.bincount(mapping,minlength=self.vocab_sizes[level-1])
            if (counts==0).any():
                raise ValueError('Every represented parent must have a TRAIN-supported leaf descendant')
            self.register_buffer(f'child_order_{level}',torch.argsort(mapping,stable=True),persistent=False)
            self.register_buffer(f'group_pointer_{level}',torch.cat([counts.new_zeros(1),counts.cumsum(0)]),persistent=False)

    def forward(self,state):
        proposal=super().forward(state)
        logits=[None]*self.depth
        logits[-1]=proposal['logits'][-1].float().log_softmax(-1)
        for level in range(self.depth-1,0,-1):
            logits[level-1]=grouped_log_marginal(logits[level],getattr(self,f'parent_index_{level}'),
                getattr(self,f'child_order_{level}'),getattr(self,f'group_pointer_{level}'))
        probabilities=[value.exp() for value in logits]
        confidences=[]
        for size,prob in zip(self.vocab_sizes,probabilities):
            certainty=(torch.ones_like(prob[:,0]) if size==1 else
                1+(prob*prob.clamp_min(1e-8).log()).sum(-1)/math.log(size))
            confidences.append(certainty.clamp(0,1))
        # Already-computed L1..L(depth-1) proposals can receive optional TRAIN
        # supervision. They do not replace the coherent retrieval marginals.
        return dict(logits=logits,probabilities=probabilities,confidence=torch.stack(confidences,1),
                    proposal_logits=proposal['logits'][:-1])


def decode_leaf_map(probabilities,parent_maps):
    """One valid full EC path per item; marginal argmaxes need not align."""
    predictions=[probabilities[-1].argmax(-1)]
    for mapping in reversed(parent_maps):
        predictions.append(mapping.index_select(0,predictions[-1]))
    return torch.stack(list(reversed(predictions)),1)  # [B,L]


class FineECFunctionalRetriever(ExplicitECFunctionalRetriever):
    """Score one latent vector plus coarse/fine internally predicted EC."""
    def __init__(self,*,ec_posterior='legacy',**kwargs):
        if ec_posterior not in {'legacy','leaf_marginal'}:
            raise ValueError('Unknown EC posterior policy')
        if kwargs.get('scoring','explicit')!='explicit':
            raise ValueError('This controlled study fixes explicit scoring')
        super().__init__(**kwargs)
        self.ec_posterior=ec_posterior
        if ec_posterior=='leaf_marginal':
            self.ec_router=LeafMarginalRouter(self.ec_router)

    def _pack(self,out):
        result=super()._pack(out)
        # This is a diagnostic/output decode, not a hard retrieval filter.
        # Soft probabilities still determine every candidate's score.
        result['ec_map_path']=decode_leaf_map(result['probabilities'],self.ec_router.parent_maps())
        return result
