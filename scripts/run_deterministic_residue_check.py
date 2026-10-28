#!/usr/bin/env python3
"""Run a local trainer with deterministic PyTorch and cuBLAS settings."""
from pathlib import Path
import os
import runpy
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch


def main():
    if len(sys.argv) < 2:
        raise ValueError('Trainer path and its normal arguments are required')
    script = Path(sys.argv[1]).resolve()
    allowed = {'train_residue_readout.py', 'train_prefetched_residue_readout.py',
               'train_cached_residue_readout.py', 'train_hier_controlled_extensions.py'}
    if script.name not in allowed or script.parent != Path(__file__).resolve().parent:
        raise ValueError('Only supported local trainer scripts are allowed')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
