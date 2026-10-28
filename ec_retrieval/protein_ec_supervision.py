"""Conservative TRAIN-only coarse EC supervision from enzyme annotations.

Reaction consensus and enzyme annotation are different supervision sources.
Only entirely unsupervised TRAIN enzyme IDs may receive a target here. Each
added prefix must be known for every source token, identical across those
tokens, and present in the unchanged TRAIN vocabulary. L4 is never changed.
"""
from collections import Counter
from pathlib import Path
import pandas as pd


def known_numeric_path(token):
    parts=str(token).strip().split('.')
    if len(parts)!=4:
        return ()
    path=[]
    for piece in parts:
        if not piece.isdigit():
            break
        path.append(piece)
    return tuple(path)


def load_train_enzyme_annotations(processed, official_source, train_ids):
    """Join only already selected TRAIN IDs to the frozen official source."""
    table=pd.read_parquet(Path(processed)/'enzymes.parquet',columns=['enzyme_id','uniprot_id'])
    selected=table.loc[table.enzyme_id.astype(str).isin(train_ids)].copy()
    if selected.enzyme_id.duplicated().any() or set(selected.enzyme_id.astype(str))!=set(train_ids):
        raise ValueError('TRAIN enzyme identities do not match annotation metadata')
    mapping=dict(zip(selected.enzyme_id.astype(str),selected.uniprot_id.astype(str)))
    raw=pd.read_csv(official_source,sep='\t',dtype=str,usecols=['Entry','EC number']).fillna('')
    raw=raw.loc[raw.Entry.astype(str).isin(set(mapping.values()))]
    if raw.Entry.duplicated().any():
        raise ValueError('Selected official UniProt entries are duplicated')
    values=dict(zip(raw.Entry.astype(str),raw['EC number'].astype(str)))
    return {enzyme:values.get(entry,'') for enzyme,entry in mapping.items()}


def fill_missing_protein_coarse_targets(protein_ids, original, token_to_index, train_ids, annotations):
    """Return new level-first target tuples plus a complete provenance audit.

    New coarse targets are supervision only, never protein/reaction features.
    This neither transfers an enzyme's other activities to a reaction nor
    changes positive retrieval pairs, EC vocabularies, or the four-level head.
    """
    if len(original)!=4 or len(token_to_index)!=4 or len(set(protein_ids))!=len(protein_ids):
        raise ValueError('Expected unique protein IDs and four aligned EC levels')
    if any(len(level)!=len(protein_ids) for level in original):
        raise ValueError('Target rows must match protein IDs')
    train_ids=set(train_ids)
    if not train_ids<=set(protein_ids) or set(annotations)!=train_ids:
        raise ValueError('Annotations must contain exactly the selected TRAIN enzyme IDs')
    output=[list(level) for level in original]
    reasons=[Counter() for _ in range(3)];additions=[]
    before=[0]*4;after=[0]*4
    for row,enzyme in enumerate(protein_ids):
        if enzyme not in train_ids:
            continue
        for level in range(4):
            before[level]+=bool(original[level][row])
        already_supervised=any(original[level][row] for level in range(4))
        tokens=sorted({p.strip() for p in str(annotations[enzyme]).split(';') if p.strip()})
        paths=[known_numeric_path(token) for token in tokens]
        for level in range(3):
            depth=level+1
            if already_supervised:
                reasons[level]['existing_targets_preserved']+=1
                continue
            if not paths or any(len(path)<depth for path in paths):
                reasons[level]['at_least_one_source_prefix_unknown']+=1
                continue
            prefixes={'.'.join(path[:depth]) for path in paths}
            if len(prefixes)!=1:
                reasons[level]['multiple_known_prefixes_not_collapsed']+=1
                continue
            prefix=next(iter(prefixes))
            if prefix not in token_to_index[level]:
                reasons[level]['prefix_not_in_unchanged_train_vocabulary']+=1
                continue
            output[level][row]=(token_to_index[level][prefix],)
            reasons[level]['added_unique_fully_observed_prefix']+=1
            additions.append(dict(enzyme_id=enzyme,level=depth,prefix=prefix,raw_ec=annotations[enzyme]))
        for level in range(4):
            after[level]+=bool(output[level][row])
    assert output[3]==original[3]
    assert all(output[level][i]==original[level][i] for i,e in enumerate(protein_ids)
               if e not in train_ids for level in range(4))
    assert all(not original[level][i] or output[level][i]==original[level][i]
               for i in range(len(protein_ids)) for level in range(4))
    audit=dict(policy='fill_unsupervised_TRAIN_enzyme_unique_prefix_L1_L3',training_splits=['train'],
        train_enzyme_ids=len(train_ids),supervision_before=before,supervision_after=after,
        added_by_level=[a-b for a,b in zip(after,before)],
        added_enzyme_ids=len({row['enzyme_id'] for row in additions}),
        reasons_by_level=[dict(r) for r in reasons],original_targets_preserved=True,
        nontrain_targets_unchanged=True,l4_unchanged=True,vocabulary_unchanged=True,
        reaction_targets_unchanged=True,retrieval_positive_pairs_unchanged=True,
        partial_prefixes_not_imputed=True,multiple_prefixes_not_forced_to_one=True,
        ground_truth_ec_is_model_input=False,official_val_test_annotations_used=False,
        later_rhea_snapshot_read=False)
    return output,audit,additions
