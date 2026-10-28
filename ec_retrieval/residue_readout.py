"""Mean-preserving residue readout inside the existing single enzyme encoder."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import segment_csr
from .fine_ec_retriever import FineECFunctionalRetriever


def residue_attention_delta(values, pointer, weight):
    """Ragged frozen [R,F] -> [B,F]; zero logits reproduce the original mean.

    Only attention scores are learned. No position/EC/ID/pair labels are added.
    The uniform contribution is subtracted explicitly, making zero-initialized
    readout exactly neutral despite cached-mean FP16 rounding.
    """
    if values.requires_grad:
        raise ValueError('This controlled experiment freezes residue features')
    with torch.autocast(device_type=values.device.type,enabled=False):
        x=values.float()
        keys=F.layer_norm(x,(x.shape[-1],),eps=1e-5)
        logits=keys@weight.float()  # [R]; learned linear scorer, no extra temperature
        lengths=pointer[1:]-pointer[:-1]
        owner=torch.repeat_interleave(torch.arange(len(lengths),device=x.device),lengths,
                                      output_size=len(values))
        maximum=segment_csr(logits.detach(),pointer,reduce='max')
        unnormalized=(logits-maximum[owner]).exp()
        denominator=segment_csr(unnormalized,pointer,reduce='sum')
        attention=unnormalized/denominator[owner].clamp_min(1e-12)
        uniform=lengths.float().reciprocal()[owner]
        delta=segment_csr((attention-uniform).unsqueeze(-1)*x,pointer,reduce='sum')
        entropy=segment_csr(-attention*attention.clamp_min(1e-12).log(),pointer,reduce='sum')
    return delta,entropy.exp()/lengths.float()


class ResidueReadoutRetriever(FineECFunctionalRetriever):
    """Keep the global mean; refine its readout without a second retrieval expert.

    mean + 0.5 * (attention_mean - residue_mean) enters the original protein MLP.
    EC predictions, scoring, index dimension and all reaction modules are intact.
    """
    def __init__(self,*,protein_readout='mean',residue_mix=.5,residue_block_size=128,**kwargs):
        super().__init__(**kwargs)
        if protein_readout not in {'mean','residue_attention'}:
            raise ValueError('Unknown protein readout')
        if residue_mix!=.5 or residue_block_size<1:
            raise ValueError('Fix half global/half attentive pooling and positive block size')
        self.protein_readout=protein_readout
        self.residue_mix=float(residue_mix)
        self.residue_block_size=int(residue_block_size)
        if protein_readout=='residue_attention':
            # Created after all original modules, without consuming RNG state.
            self.residue_weight=nn.Parameter(torch.zeros(kwargs['protein_input_dim']))

    def encode_protein(self,features):
        if self.protein_readout=='mean':
            if not isinstance(features,torch.Tensor):
                raise ValueError('Mean control requires the unchanged feature tensor')
            return super().encode_protein(features)
        if set(features)!={'mean','values','offsets_cpu'}:
            raise ValueError('Residue readout accepts frozen means and ragged features only')
        means,values,offsets=features['mean'],features['values'],features['offsets_cpu']
        if len(offsets)!=len(means)+1 or offsets[0]!=0 or offsets[-1]!=len(values):
            raise ValueError('Invalid residue boundaries')
        if any(b<=a for a,b in zip(offsets,offsets[1:])):
            raise ValueError('Empty protein sequences are not allowed')
        deltas,fractions=[],[]
        # Blocks contain up to 128 proteins, never a per-protein GPU loop. This
        # limits temporary FP32 residue allocations without changing the loss
        # or splitting the candidate-score matrix / encoder dropout operation.
        for first in range(0,len(means),self.residue_block_size):
            last=min(first+self.residue_block_size,len(means));start,stop=offsets[first],offsets[last]
            ptr=torch.tensor(offsets[first:last+1],device=values.device,dtype=torch.long)-start
            delta,fraction=residue_attention_delta(values[start:stop],ptr,self.residue_weight)
            deltas.append(delta);fractions.append(fraction)
        refined=means.float()+self.residue_mix*torch.cat(deltas)
        result=super().encode_protein(refined)
        result['residue_effective_fraction']=torch.cat(fractions)
        result['residue_refinement_norm']=torch.cat(deltas).float().norm(dim=1)*self.residue_mix
        return result
