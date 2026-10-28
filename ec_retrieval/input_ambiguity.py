"""Neutral TRAIN comparisons for indistinguishable model inputs.

Observed pairs remain the only positives. Other observed associations of an
input-identical reaction, or of an identical full enzyme sequence, identify
uncertain comparisons to exclude from TRAIN denominators. This does not assert
biochemical equivalence of reactions and never modifies evaluation candidates.
"""
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.sparse import csr_matrix


class TrainInputAmbiguity:
    """Build sparse comparison masks once from explicitly selected TRAIN rows."""
    def __init__(self, query_ids, protein_ids, observed_pairs, components, sequences, fingerprints):
        if len(set(query_ids))!=len(query_ids) or len(set(protein_ids))!=len(protein_ids):
            raise ValueError('Unique feature identities required')
        qi,pi={q:i for i,q in enumerate(query_ids)},{p:i for i,p in enumerate(protein_ids)}
        observed=defaultdict(set)
        for q,p in observed_pairs:
            if q not in qi or p not in pi:raise ValueError('TRAIN pair missing from feature rows')
            observed[q].add(p)
        tq,tp=set(observed),set().union(*observed.values())
        if not tq or set(components)!=tq or set(sequences)!=tp:
            raise ValueError('Metadata must contain exactly selected TRAIN reaction/enzyme IDs')
        fingerprints=np.asarray(fingerprints)
        if fingerprints.ndim!=2 or len(fingerprints)!=len(query_ids) or not np.isfinite(fingerprints).all():
            raise ValueError('Finite aligned Morgan inputs required')
        raw_groups=defaultdict(list)
        for q in sorted(tq):
            if not components[q]:raise ValueError('Empty reaction input')
            raw_groups[tuple(sorted(Counter(components[q]).items()))].append(q)
        reaction_groups=[]
        for group in raw_groups.values():
            # Direct equality of cached model inputs, not a chemistry hash.
            same=defaultdict(list)
            for q in group:same[fingerprints[qi[q]].tobytes()].append(q)
            reaction_groups.extend(same.values())
        qgroup={q:g for g in reaction_groups for q in g}
        sequence_groups=defaultdict(list)
        for p in sorted(tp):
            if not sequences[p]:raise ValueError('Empty full enzyme sequence')
            sequence_groups[sequences[p]].append(p)
        pgroup={p:g for g in sequence_groups.values() for p in g}
        reaction_neutral={q:set().union(*(observed[r] for r in qgroup[q]))-observed[q] for q in tq}
        protein_neutral={q:set().union(*(set(pgroup[p]) for p in observed[q]))-observed[q] for q in tq}
        shape=(len(query_ids),len(protein_ids))
        def sparse(mapping):
            edges=sorted((qi[q],pi[p]) for q,ps in mapping.items() for p in ps)
            if not edges:return csr_matrix(shape,dtype=bool)
            rr,pp=np.asarray(edges,dtype=np.int64).T
            return csr_matrix((np.ones(len(edges),dtype=bool),(rr,pp)),shape=shape)
        self.truth=sparse(observed)
        reaction_mask,protein_mask=sparse(reaction_neutral),sparse(protein_neutral)
        self.neutral=reaction_mask.maximum(protein_mask)
        if self.truth.multiply(self.neutral).nnz:
            raise ValueError('Observed positives must never be neutralized')
        self.query_rows=np.array(sorted(qi[q] for q in tq),dtype=np.int64)
        self.protein_rows=np.array(sorted(pi[p] for p in tp),dtype=np.int64)
        self.audit=dict(policy='input_ambiguity_neutral',training_splits=['train'],
            train_reactions=len(tq),train_enzyme_ids=len(tp),observed_positive_pairs=int(self.truth.nnz),
            identical_reaction_input_groups=sum(len(g)>1 for g in reaction_groups),
            input_ambiguous_reactions=sum(len(g) for g in reaction_groups if len(g)>1),
            identical_full_sequence_groups=sum(len(g)>1 for g in sequence_groups.values()),
            sequence_alias_enzyme_ids=sum(len(g) for g in sequence_groups.values() if len(g)>1),
            reaction_neutral_edges=int(reaction_mask.nnz),protein_neutral_edges=int(protein_mask.nnz),
            union_neutral_edges=int(self.neutral.nnz),positive_neutral_overlap=0,
            observed_positive_pairs_unchanged=True,ec_targets_unchanged=True,
            true_ec_used_for_identity=False,official_val_test_associations_used=False,
            biological_identity_of_reactions_assumed=False,inference_unchanged=True)

    def batch_mask(self, queries, proteins, observed_mask):
        """Return a CPU [Q,P] mask; validate the original complete TRAIN truth."""
        q,p=np.asarray(queries,dtype=np.int64),np.asarray(proteins,dtype=np.int64)
        if (q.ndim!=1 or p.ndim!=1 or not len(q) or not len(p)
                or not np.isin(q,self.query_rows).all() or not np.isin(p,self.protein_rows).all()):
            raise ValueError('Comparison masks are available only within the TRAIN domain')
        if observed_mask.device.type!='cpu' or observed_mask.dtype!=torch.bool:
            raise ValueError('Validate CPU boolean TRAIN truth before device transfer')
        actual=torch.from_numpy(self.truth[q][:,p].toarray())
        if not torch.equal(actual,observed_mask):
            raise ValueError('Observed mask differs from complete original TRAIN truth')
        neutral=torch.from_numpy(self.neutral[q][:,p].toarray())
        if bool((neutral&actual).any()):raise ValueError('Cannot suppress an observed positive')
        return neutral

    def export(self):
        rows,cols=self.neutral.nonzero()
        return dict(query_rows=torch.from_numpy(self.query_rows.copy()),
            protein_rows=torch.from_numpy(self.protein_rows.copy()),
            neutral_query_rows=torch.from_numpy(rows.astype(np.int64)),
            neutral_protein_rows=torch.from_numpy(cols.astype(np.int64)),
            training_splits=['train'],observed_positive_pairs_unchanged=True)


def exclude_neutral_comparisons(scores, neutral_mask=None):
    """Change only TRAIN softmax comparison entries; None is an exact bypass."""
    if neutral_mask is None:return scores
    if scores.ndim!=2 or neutral_mask.shape!=scores.shape or neutral_mask.dtype!=torch.bool:
        raise ValueError('Aligned boolean neutral mask required')
    if neutral_mask.device!=scores.device:raise ValueError('Score/mask devices must match')
    return scores.masked_fill(neutral_mask,-torch.inf)
