# Third-party materials

## ReactZyme

`external/ReactZyme_official/retrieval.py` is the ReactZyme metric implementation from <https://github.com/WillHua127/ReactZyme>. Its upstream CC0-1.0 license is reproduced in `external/ReactZyme_official/LICENSE`. HiER's evaluator extracts and calls the `topk_accuracy` function to verify reported ranks and metrics; it does not invoke the upstream training program.

## Dependencies and pretrained models

PyTorch, PyTorch Geometric, torch-scatter, NumPy, pandas, SciPy, RDKit, Uni-Mol and ESM-2 are separate projects with their own licenses. Their source distributions and pretrained weights are not included here. Refer to the respective upstream projects when obtaining them:

- Uni-Mol: <https://github.com/deepmodeling/Uni-Mol>
- ESM: <https://github.com/facebookresearch/esm>
- ReactZyme data: <https://zenodo.org/records/13635807>

The license of the vendored metric file does not license the rest of this repository.
