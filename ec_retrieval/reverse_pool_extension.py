"""TRAIN-only reverse candidates, without changing forward anchors or inference."""
from contextlib import contextmanager
import numpy as np
import torch
from scipy.sparse import csr_matrix

from .positive_coverage import positive_coverage_loss
from .reaction_balanced_reverse import reaction_macro_reverse_loss


class ReverseReactionPool:
    """Sparse complete TRAIN truth; IDs are alignment metadata, never features."""
    def __init__(self, domain, size, seed):
        self.rows=domain.query_rows.copy()
        if size < 1 or size > len(self.rows):
            raise ValueError('Reverse pool must fit the unique TRAIN reaction domain')
        self.size=int(size)
        self.rng=np.random.default_rng(seed)
        qs=np.repeat(self.rows,[len(domain.adjacency[int(q)]) for q in self.rows])
        ps=np.concatenate([domain.adjacency[int(q)] for q in self.rows])
        self.truth=csr_matrix((np.ones(len(qs),dtype=bool),(qs,ps)),
            shape=(int(self.rows.max())+1,int(domain.universe.max())+1))
        self.domain=domain

    def sample(self, forward_queries, candidates, original_mask):
        query=np.asarray(forward_queries,dtype=np.int64)
        if len(np.unique(query))!=len(query) or not np.isin(query,self.rows).all():
            raise ValueError('Forward anchors must be unique TRAIN reactions')
        if len(query)>self.size:
            raise ValueError('Reverse pool cannot truncate forward anchors')
        self.domain.check_candidates(candidates)
        remaining=self.rows[~np.isin(self.rows,query)]
        extra=self.rng.choice(remaining,size=self.size-len(query),replace=False)
        combined=np.r_[query,extra]
        mask=torch.from_numpy(self.truth[combined][:,candidates].toarray())
        if not torch.equal(mask[:len(query)],original_mask.cpu().bool()):
            raise ValueError('Forward mask must include ALL known TRAIN positives')
        return extra.tolist(),mask,combined


@contextmanager
def isolated_extra_rng(device, seed):
    """Extra encoder dropout must not consume the baseline's random stream."""
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        yield


def expanded_bidirectional_loss(scores, mask, extra_scores, extended_mask, *,
        reverse_weight=.5, mode='legacy', all_positive_weight=.2,
        tail_risk_weight=.1, tail_fraction=.2, reverse_reduction='enzyme_mean'):
    """[B,C] forward, [C,B+E] reverse; one unchanged protein encoding.

    Extra reactions receive retrieval gradients only. Known-positive masks are
    built from complete TRAIN associations. Missing positives do not turn into
    diagonal targets and unobserved pairs are not verified biological negatives.
    """
    if reverse_weight!=.5 or extra_scores.ndim!=2 or extra_scores.shape[1]!=scores.shape[1]:
        raise ValueError('Expected a half-weighted expanded reverse objective')
    if extended_mask.shape!=(len(scores)+len(extra_scores),scores.shape[1]):
        raise ValueError('Reverse score/mask dimensions differ')
    settings=dict(mode=mode,all_positive_weight=all_positive_weight,
                  tail_risk_weight=tail_risk_weight,tail_fraction=tail_fraction)
    forward,rdiag=positive_coverage_loss(scores,mask,**settings)
    all_scores=torch.cat((scores,extra_scores),dim=0)
    if reverse_reduction=='enzyme_mean':
        reverse,ediag=positive_coverage_loss(all_scores.T,extended_mask.T,**settings)
    elif reverse_reduction=='reaction_macro' and mode=='legacy':
        reverse,ediag=reaction_macro_reverse_loss(all_scores.T,extended_mask.T,
            all_positive_weight=all_positive_weight,tail_risk_weight=tail_risk_weight,
            tail_fraction=tail_fraction)
    else:
        raise ValueError('Unknown reverse reduction or unsupported combined loss change')
    diagnostic={k:.5*(rdiag[k]+ediag[k]) for k in ('any_positive','all_positive','tail_positive')}
    diagnostic.update({f'r2e_{k}':v for k,v in rdiag.items()})
    diagnostic.update({f'e2r_{k}':v for k,v in ediag.items()})
    with torch.no_grad():
        diagnostic.update(r2e_retrieval=forward.detach(),e2r_retrieval=reverse.detach(),
            r2e_valid_anchor_count=mask.any(1).sum().float(),
            e2r_valid_anchor_count=extended_mask.any(0).sum().float(),
            e2r_skipped_anchor_count=(~extended_mask.any(0)).sum().float(),
            observed_positive_count=mask.sum().float(),
            e2r_observed_positive_count=extended_mask.sum().float(),
            e2r_reaction_pool_size=scores.new_tensor(len(all_scores),dtype=torch.float32))
    return .5*(forward+reverse),diagnostic
