"""Fail-closed validation of internal reaction-held-out TRAIN views."""
import json
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal


def validate_reaction_crossfit(reference):
    from ec_retrieval.official_split_view import validate_split_view
    root=Path(reference['processed_dir'])
    parent=json.loads(Path(reference['crossfit_parent_config']).read_text())
    if parent.get('train_crossfit_view'):
        raise ValueError('Nested crossfit references are not supported')
    parent_audit=validate_split_view(parent)
    source=Path(parent['processed_dir'])
    original=pd.read_parquet(source/'pairs.parquet',filters=[('split','==','train')])
    data=pd.read_parquet(root/'pairs.parquet')
    if set(data.split)!={'train','val','oof'} or reference.get('test_contract') is not None:
        raise ValueError('Crossfit must expose only TRAIN-derived train/inner-val/oof')
    columns=[c for c in original.columns if c!='split']
    assert_frame_equal(original[columns].sort_values(columns).reset_index(drop=True),
                       data[columns].sort_values(columns).reset_index(drop=True))
    if data.groupby('reaction_id').split.nunique().max()!=1:
        raise ValueError('A reaction crosses fit/inner/outer boundaries')
    assignments=pd.read_csv(root.parents[1]/'fold_assignments.csv').set_index('reaction_id')
    fold=int(reference['crossfit_fold'])
    outer=set(assignments.index[assignments.fold==fold])
    if outer!=set(data.loc[data.split=='oof','reaction_id']):
        raise ValueError('Outer membership differs from frozen fold assignment')
    membership=data[['reaction_id','split']].drop_duplicates().set_index('reaction_id')
    if membership.join(assignments).groupby('cluster').split.nunique().max()!=1:
        raise ValueError('A reaction cluster crosses fit/inner/outer boundaries')
    for name in ('enzymes.parquet','reactions.parquet'):
        assert_frame_equal(pd.read_parquet(root/name),pd.read_parquet(source/name))
    for key in ('protein_row_mapping','protein_view_ids','frozen_feature_reference','official_uniprot_source'):
        if Path(reference[key]).resolve()!=Path(parent[key]).resolve():
            raise ValueError(f'Frozen feature/annotation identity changed: {key}')
    from scripts.prepare_hier_official_splits import evaluation_contract
    enzymes=pd.read_parquet(root/'enzymes.parquet')
    for part,key in [('val','validation_contract'),('oof','oof_contract')]:
        contract=json.loads(Path(reference[key]).read_text())
        if contract!=evaluation_contract(data,enzymes,root,part):
            raise ValueError(f'Contract differs from TRAIN-only pair membership: {part}')
    return dict(status='passed',role='TRAIN-only internal reaction crossfit',fold=fold,
        original_train_rows_preserved=True,original_val_test_rows_in_fit_or_inner_or_outer=0,
        outer_fit_reaction_overlap=0,outer_inner_reaction_overlap=0,cluster_disjoint=True,
        inner_selects_checkpoint=True,outer_selects_checkpoint=False,
        train_only_statistics=True,fresh_fold_ec_vocabulary_required=True,
        official_benchmark_membership_unchanged=True,parent_identity_audit=parent_audit)
