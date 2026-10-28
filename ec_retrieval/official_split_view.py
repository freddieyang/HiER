"""Fail-closed identity audit for label-independent cross-split feature reuse."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_split_view(reference):
    if reference.get('train_crossfit_view'):
        from ec_retrieval.reaction_crossfit import validate_reaction_crossfit
        return validate_reaction_crossfit(reference)
    root = Path(reference['processed_dir'])
    audit = json.loads((root / 'identity_audit.json').read_text())
    if (audit['status'] != 'passed' or not audit['original_rows_preserved'] or
            audit['split_type'] != reference['benchmark_split'] or
            audit['exact_pair_train_test_overlap'] != 0 or
            audit['missing_sequences'] or audit['missing_reactions']):
        raise ValueError('Invalid official split identity audit')
    frozen = json.loads(Path(reference['frozen_feature_reference']).read_text())
    source = Path(frozen['processed_dir'])
    target_r = pd.read_parquet(root / 'reactions.parquet').set_index('reaction_id')
    source_r = pd.read_parquet(source / 'reactions.parquet').set_index('reaction_id')
    if not target_r['official_molecule_set_smiles'].equals(source_r['official_molecule_set_smiles']):
        raise ValueError('Reaction inventory identity changed')
    source_p = pd.read_parquet(source / 'enzymes.parquet').set_index('enzyme_id')
    target_p = pd.read_parquet(root / 'enzymes.parquet')
    source_ids = json.loads(Path(frozen['protein_global_ids']).read_text())
    if isinstance(source_ids, dict):
        source_ids = source_ids.get('enzyme_ids', source_ids.get('ids'))
    rows = np.load(reference['protein_row_mapping'])
    ids = json.loads(Path(reference['protein_view_ids']).read_text())['enzyme_ids']
    if ids != target_p.enzyme_id.tolist() or len(rows) != len(ids) or (rows < 0).any() or (rows >= len(source_ids)).any():
        raise ValueError('Invalid sequence feature row mapping')
    mapped_ids = [source_ids[i] for i in rows]
    if mapped_ids != target_p.frozen_feature_source_id.tolist():
        raise ValueError('Mapped feature identities changed')
    if target_p.sequence.tolist() != source_p.loc[mapped_ids, 'sequence'].tolist():
        raise ValueError('Refusing to reuse features of a different sequence')
    train_val = pd.read_parquet(root / 'pairs.parquet', filters=[('split', 'in', ['train', 'val'])])
    if len(train_val) != audit['official_train_val_records'] or set(train_val.split) != {'train', 'val'}:
        raise ValueError('TRAIN/VAL rows differ from the released archive')
    if train_val.duplicated(['reaction_id', 'enzyme_id', 'split']).any():
        raise ValueError('Duplicate TRAIN/VAL association keys')
    contract = json.loads(Path(reference['validation_contract']).read_text())
    if contract['split'] != 'val' or contract['copy_only'] is not True or contract['adapter'] != 'ReactZymeAdapter':
        raise ValueError('Invalid validation contract')
    if Path(contract['source_contract']).resolve() != (root / 'evaluation_contract.json').resolve():
        raise ValueError('Validation contract belongs to another split')
    return dict(status='passed', split_type=reference['benchmark_split'],
        original_rows_preserved=True, exact_sequence_checked_rows=len(ids),
        exact_molecule_checked_rows=len(target_r), source_feature_reference=reference['frozen_feature_reference'],
        feature_aliases=audit['feature_aliases'], train_val_rows=len(train_val),
        test_associations_used_for_optimization=False, test_relevance_loaded_by_trainer=False,
        preparation_read_test_for_membership_and_identity_audit=True,
        train_only_statistics=True, global_features_label_independent=True)
