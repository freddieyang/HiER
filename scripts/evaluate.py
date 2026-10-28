#!/usr/bin/env python3
"""Evaluate a completed HiER run in both retrieval directions."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('evaluate_hier_bidirectional.py')), run_name='__main__')
