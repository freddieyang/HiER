"""Supplement reaction-side TRAIN EC targets without inferring missing children."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def known_numeric_prefixes(annotation):
    """Conservatively parse each annotation; malformed tokens remain unknown."""
    if pd.isna(annotation) or not str(annotation).strip():
        return [()]
    paths = []
    for token in str(annotation).split(';'):
        parts, prefix, unknown, malformed = token.strip().split('.'), [], False, False
        if len(parts) != 4:
            paths.append(())
            continue
        for part in parts:
            if part == '-':
                unknown = True
            elif not part.isdigit() or unknown:
                malformed = True
                break
            else:
                prefix.append(part)
        paths.append(tuple(prefix) if not malformed else ())
    return paths or [()]


def consensus_for_train_pairs(train_pairs, enzymes, annotations):
    """Read only annotations linked to explicitly TRAIN associations."""
    if set(train_pairs['split'].astype(str)) != {'train'}:
        raise ValueError('Consensus targets require nonempty TRAIN-only pairs')
    if train_pairs.duplicated(['reaction_id', 'enzyme_id']).any():
        raise ValueError('Duplicate TRAIN association')
    enzymes = enzymes[['enzyme_id', 'uniprot_id']].astype(str)
    if enzymes.enzyme_id.duplicated().any():
        raise ValueError('Duplicate enzyme metadata')
    lookup = dict(enzymes.itertuples(index=False, name=None))
    required = {lookup[str(e)] for e in train_pairs.enzyme_id}
    raw = annotations.loc[annotations.Entry.astype(str).isin(required), ['Entry', 'EC number']]
    if raw.Entry.duplicated().any() or set(raw.Entry.astype(str)) != required:
        raise ValueError('Missing or duplicated TRAIN source annotations')
    parsed = {str(entry): known_numeric_prefixes(value) for entry, value in raw.itertuples(index=False, name=None)}
    result = {}
    for query, group in train_pairs.groupby('reaction_id', sort=True):
        paths = [path for enzyme in group.enzyme_id for path in parsed[lookup[str(enzyme)]]]
        levels = []
        for level in range(3):
            candidates = {'.'.join(p[:level + 1]) for p in paths} if all(len(p) > level for p in paths) else set()
            levels.append(next(iter(candidates)) if len(candidates) == 1 else None)
        result[str(query)] = tuple(levels)
    return result


def supplement_reaction_targets(query_ids, legacy_targets, vocabulary_maps, consensus, strict_query_ids):
    """Add only L1-L3 labels; preserve all historical rows and every L4 target."""
    if len(legacy_targets) != 4 or len(vocabulary_maps) != 4:
        raise ValueError('This controlled ablation requires the existing four-level vocabulary')
    query_ids = list(map(str, query_ids))
    if len(set(query_ids)) != len(query_ids) or any(len(level) != len(query_ids) for level in legacy_targets):
        raise ValueError('Query IDs and target rows must align')
    if not set(consensus).issubset(query_ids) or not set(strict_query_ids).issubset(consensus):
        raise ValueError('TRAIN target IDs are missing from the declared query inventory')
    result = [[tuple(values) for values in level] for level in legacy_targets]
    before = [sum(bool(v) for v in level) for level in legacy_targets]
    records = []
    for row, query in enumerate(query_ids):
        if query not in consensus or query in strict_query_ids:
            continue
        if any(level[row] for level in legacy_targets):
            raise ValueError('Supplement attempted on an unexpected historical target')
        for level, prefix in enumerate(consensus[query]):
            if prefix is None or prefix not in vocabulary_maps[level]:
                continue
            result[level][row] = (vocabulary_maps[level][prefix],)
            records.append(dict(reaction_id=query, level=level + 1, prefix=prefix))
    if result[3] != legacy_targets[3]:
        raise AssertionError('L4 targets changed')
    for row, query in enumerate(query_ids):
        if query in strict_query_ids or query not in consensus:
            if any(result[l][row] != tuple(legacy_targets[l][row]) for l in range(4)):
                raise AssertionError('Historical or non-TRAIN target changed')
    after = [sum(bool(v) for v in level) for level in result]
    audit = dict(policy='coarse_consensus_l1_l3', training_splits=['train'],
        supervision_before=before, supervision_after=after,
        added_by_level=[b - a for a, b in zip(before, after)],
        added_query_count=len({r['reaction_id'] for r in records}),
        original_targets_preserved=True, l4_unchanged=True, vocabulary_unchanged=True,
        nontrain_targets_unchanged=True, protein_targets_unchanged=True,
        source_policy='all annotations known and unanimous at each added level; existing vocabulary only',
        missing_children_imputed=False, later_rhea_snapshot_read=False, test_labels_read=False)
    return result, audit, records


def load_train_consensus(processed: Path, source: Path, train_pairs):
    enzymes = pd.read_parquet(processed / 'enzymes.parquet', columns=['enzyme_id', 'uniprot_id'])
    annotations = pd.read_csv(source, sep='\t', dtype=str, usecols=['Entry', 'EC number'])
    return consensus_for_train_pairs(train_pairs, enzymes, annotations)
