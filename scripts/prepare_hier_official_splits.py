"""Compatibility helper for evaluation contracts used by cross-fit utilities.

For public data preparation use scripts/prepare_data.py. This function preserves
contract construction for the optional cross-fit code imported by the trainer.
"""
from collections import defaultdict


def evaluation_contract(pairs, enzymes, processed, split):
    selected = pairs.loc[pairs.split == split]
    sequence = dict(zip(enzymes.enzyme_id, enzymes.sequence))
    groups = defaultdict(list)
    for eid in sorted(set(selected.enzyme_id)):
        groups[sequence[eid]].append(eid)
    representative = {eid: min(ids) for ids in groups.values() for eid in ids}
    relevance = defaultdict(set)
    for q, p in selected[['reaction_id', 'enzyme_id']].itertuples(index=False, name=None):
        relevance[q].add(representative[p])
    qids = sorted(relevance)
    pids = sorted(set(representative.values()))
    return dict(adapter='ReactZymeAdapter', direction='reaction_to_enzyme', split=split,
        copy_only=True, source_contract=str(processed / 'evaluation_contract.json'),
        query_ids=qids, candidate_ids=pids,
        relevance={q: sorted(relevance[q]) for q in qids},
        candidate_identity='exact_full_amino_acid_sequence',
        source_positive_rows=len(selected), unique_positive_pairs=sum(map(len, relevance.values())),
        identical_sequence_alias_count=len(representative)-len(pids))
