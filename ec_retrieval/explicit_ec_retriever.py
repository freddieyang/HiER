"""One dual encoder with an explicitly decomposable predicted-EC score.

The explicit branch removes both residual fusion networks. Retrieval uses
one concatenated vector and one inner product; it needs no query-dependent
candidate encoder, external EC label, stored TRAIN support or second expert.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .shared_functional_set_retriever import SharedFusionFunctionalSetRetriever
from .retrieval_scoring import (ec_root_blocks, joint_index_vector,
                                positive_score_scale, score_components)


class ProbabilityOnlyRouter(nn.Module):
    """Keep the existing parent-conditioned predictors, remove unused context."""
    def __init__(self, full):
        super().__init__()
        if full.normalization != 'parent_prior':
            raise ValueError('This controlled router uses the existing parent-prior normalization')
        self.vocab_sizes=full.vocab_sizes
        self.normalization=full.normalization
        self.state_norm=full.state_norm
        self.heads=full.heads
        # Level l embeddings are needed to predict l+1. The last embedding,
        # context norm and context mixing weights have no role without fusion.
        self.prefix_embeddings=nn.ModuleList(list(full.prefix_embeddings)[:-1])
        for level,mapping in enumerate(full.parent_maps(),1):
            self.register_buffer(f'parent_index_{level}',mapping.clone())

    @property
    def depth(self):
        return len(self.vocab_sizes)

    def parent_maps(self):
        return [getattr(self,f'parent_index_{i}') for i in range(1,self.depth)]

    def forward(self,state):
        if state.ndim!=2:
            raise ValueError('EC router state must be [B,D]')
        state=self.state_norm(state)
        logits, probabilities, confidence=[],[],[]
        for level,head in enumerate(self.heads):
            if level==0:
                raw=head(state)
            else:
                parent=probabilities[-1]@self.prefix_embeddings[level-1].weight.float()
                raw=head(torch.cat([state,parent],-1))
                prior=probabilities[-1].index_select(1,getattr(self,f'parent_index_{level}'))
                raw=raw.float()+prior.clamp_min(1e-8).log()
            prob=raw.float().softmax(-1)
            size=self.vocab_sizes[level]
            certainty=(torch.ones_like(prob[:,0]) if size==1 else
                1+(prob*prob.clamp_min(1e-8).log()).sum(-1)/math.log(size))
            logits.append(raw);probabilities.append(prob);confidence.append(certainty.clamp(0,1))
        return dict(logits=logits,probabilities=probabilities,confidence=torch.stack(confidence,1))


class ExplicitECFunctionalRetriever(SharedFusionFunctionalSetRetriever):
    """s(q,p) = scale [(1-a) <zq,zp> + a/L sum_l BC(ECq_l,ECp_l)].

    Both EC distributions are predictions. BC is the inner product of their
    square roots. The shared scalar a is learned on TRAIN, never calibrated on
    VAL/TEST. Returned vectors make the score exactly indexable and separable.
    """
    def __init__(self, *, scoring='explicit', ec_score_depth=3,
                 initial_ec_weight=.25, **kwargs):
        if scoring not in {'gated','explicit'}:
            raise ValueError('Unknown retrieval scoring policy')
        if kwargs.get('pooling')!='moments':
            raise ValueError('This comparison fixes moments pooling')
        if not 0<initial_ec_weight<1:
            raise ValueError('Initial EC weight must be in (0,1)')
        super().__init__(**kwargs)
        if not 1<=ec_score_depth<=self.ec_router.depth:
            raise ValueError('EC score depth exceeds the TRAIN vocabulary')
        self.scoring=scoring
        self.ec_score_depth=ec_score_depth
        self.index_dimension=self.embedding_dim
        if scoring=='explicit':
            if self.fusion_sharing!='independent_matched':
                raise ValueError('Explicit comparison removes the two independent fusion modules')
            self.ec_conditioning=False  # No implicit residual context path.
            del self.reaction_fusion,self.protein_fusion
            self.ec_router=ProbabilityOnlyRouter(self.ec_router)
            # The inherited moments branch never reads these attention layers.
            for name in ('molecule_projection','pool_query','pool_key','pool_value','pool_output','readout_norm'):
                delattr(self,name)
            self.ec_weight_logit=nn.Parameter(torch.tensor(math.log(initial_ec_weight/(1-initial_ec_weight))))
            self.index_dimension+=sum(self.ec_router.vocab_sizes[:ec_score_depth])

    def ec_weight(self):
        if self.scoring!='explicit':
            raise ValueError('Only the explicit branch has an EC score weight')
        return self.ec_weight_logit.float().sigmoid()

    def _pack(self,out):
        if self.scoring=='gated':
            return out
        a=self.ec_weight()
        roots=ec_root_blocks(out['probabilities'],self.ec_score_depth)
        base=out['embedding']
        # [B,D + sum_l V_l]; each EC block has unit norm before weighting.
        out['embedding']=joint_index_vector(base,roots,a,self.ec_score_depth)
        out['latent_embedding']=base
        out['ec_sqrt_probabilities']=roots
        return out

    def encode_reaction(self,features):
        return self._pack(super().encode_reaction(features))

    def encode_protein(self,features):
        return self._pack(super().encode_protein(features))

    def explain_scores(self,reaction,protein):
        """Exact score terms [Bq,Bp]; these are not mechanistic explanations."""
        if self.scoring!='explicit':
            raise ValueError('Score decomposition requires explicit scoring')
        a=self.ec_weight()
        scale=positive_score_scale(self.logit_scale)
        return score_components(reaction,protein,a,scale,self.ec_score_depth)
