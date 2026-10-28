#!/usr/bin/env python3
"""Run HiER or one of the three paper ablations using explicit paper settings."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'no_ec_branch', 'no_latent_similarity', 'no_ec_agreement')


def build_command(args):
    parameters = json.loads((ROOT / 'configs/paper.json').read_text())
    for option, value in (('all-positive-weight', args.beta),
                          ('ec-loss-weight', args.lambda_ec),
                          ('cross-modal-ec-weight', args.lambda_align)):
        if value is not None:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{option} must be finite and nonnegative')
            parameters[option] = value
    data_root = args.data_root.resolve()
    reference = (args.reference_run or data_root / 'processed' / args.split).resolve()
    source = (args.source_dir or data_root / 'features' / 'unimol').resolve()
    residue = (args.residue_cache or reference / 'residue_view.json').resolve()
    config_path = reference / 'config.json'
    reference_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if args.source_dir is None and reference_config.get('source_dir'):
        source = Path(reference_config['source_dir']).resolve()
    output = args.output.resolve()
    if not args.dry_run:
        if not config_path.is_file():
            raise FileNotFoundError(f'Prepare data and feature references first: {config_path}')
        if reference_config.get('features_ready') is False:
            raise ValueError('Feature cache is incomplete: ' + str(reference_config.get('missing_features', [])))
        if reference_config.get('benchmark_split') != args.split:
            raise ValueError('Prepared reference split does not match --split')
        if output.exists():
            raise FileExistsError(f'Use a new run directory: {output}')
        if not residue.is_file() or not source.is_dir():
            raise FileNotFoundError('Both the residue cache view and Uni-Mol cache are required')
    if sum(v is not None for v in (args.beta, args.lambda_ec, args.lambda_align)) > 1:
        raise ValueError('Sensitivity runs change one loss weight at a time')
    parameters.update({
        'reference-run': reference, 'source-dir': source, 'residue-cache': residue,
        'output-dir': output, 'seed': args.seed, 'device': args.device,
        'matched-ablation': 'none' if args.variant == 'full' else args.variant,
        'num-threads': args.num_threads, 'residue-prefetch-depth': args.prefetch_depth,
        'resident-cache-devices': args.cache_devices,
        'resident-cache-gib': args.cache_gib,
    })
    command = [sys.executable, str(ROOT / 'scripts/run_deterministic_residue_check.py'),
               str(ROOT / 'scripts/train_hier_controlled_extensions.py')]
    for key, value in parameters.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        else:
            command.append('--' + key)
            command.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('time', 'seq_smi', 'mol_smi'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--residue-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--cache-devices', default='0', help='Logical CUDA cache devices')
    parser.add_argument('--cache-gib', default='6', help='GiB budgets corresponding to cache devices')
    parser.add_argument('--prefetch-depth', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--beta', type=float, help='Single-parameter retrieval-loss sensitivity run')
    parser.add_argument('--lambda-ec', type=float, help='Single-parameter EC supervision sensitivity run')
    parser.add_argument('--lambda-align', type=float, help='Single-parameter EC distribution alignment sensitivity run')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without training or reading tensors')
    args = parser.parse_args()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
