# HiER

Code for **HiER: hierarchy-informed functional matching for enzyme–reaction retrieval**.

HiER ranks enzymes for a reaction and reactions for an enzyme. It combines molecular fingerprints and frozen Uni-Mol features with frozen ESM-2 residue features. A shared predictor produces EC distributions for both modalities, with coarse probabilities obtained by summing the finest-level distribution. Retrieval combines latent similarity and multilevel EC agreement in one inner product; candidates can be encoded independently of the query.

## Contents

- `ec_retrieval/`: encoders, hierarchy-consistent prediction, joint scoring, losses and feature stores.
- `configs/paper.json`: the full-model training settings.
- `scripts/train.py`: model training with the paper settings.
- `scripts/evaluate.py`: validation-selected checkpoint evaluation in both directions.
- `scripts/prepare_data.py`: official split conversion and feature-reference checks.
- `analysis/`: figure renderers that take explicit local result inputs.
- `tests/`: CPU checks of scoring, hierarchy and ranking semantics.

The release contains source code. Benchmark data, frozen feature caches, pretrained backbone weights, trained HiER checkpoints and experiment results are separate inputs; they are not included here. See [data preparation](docs/data.md) for the input formats and current cache-based workflow, and [figure inputs](docs/figures.md) for plotting requirements.

## Environment

Use Python 3.10. The core was checked with PyTorch 2.4.0, CUDA 12.4, `torch-scatter` 2.1.2 and `torch-geometric` 2.6.1. Install the PyTorch and scatter wheels for the same CUDA version. For that environment:

```bash
conda create -n hier python=3.10 -y
conda activate hier
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install -r requirements.txt
```

For CPU checks, install the corresponding PyTorch CPU wheel and the scatter wheels from `https://data.pyg.org/whl/torch-2.4.0+cpu.html`. The formal trainer uses CUDA and BF16; CPU tests do not require data or weights. Plotting additionally uses `pip install -r requirements-figures.txt`.

## Prepare the official data

Download the released data from [ReactZyme](https://github.com/WillHua127/ReactZyme) / [Zenodo](https://zenodo.org/records/13635807), then follow the layout in [docs/data.md](docs/data.md).

```bash
python scripts/prepare_data.py \
  --raw-dir data/raw/ReactZyme \
  --output-dir data/processed_preview
```

This previews the tables and retrieval contracts; it does not extract frozen features. With matching Morgan, Uni-Mol and ESM-2 caches available, create the training reference in a new directory:

```bash
python scripts/prepare_data.py \
  --raw-dir data/raw/ReactZyme \
  --output-dir data/processed \
  --feature-reference data/features/reference.json \
  --unimol-dir data/features/unimol \
  --residue-cache data/features/residue_cache/manifest.json
```

The adapter validates identities and marks whether all required features are available. A validated raw-sequence/raw-molecule-to-cache extraction pipeline is not included in this release. If you use the `processed_with_features` output directory shown in the detailed guide, pass `--reference-run data/processed_with_features/<split>` when training.

The official test partitions (`time`, `seq_smi`, `mol_smi`) are retained. Training uses a fixed 90/10 training/validation division of each official train/validation archive. Training IDs retain their own EC targets; exact-sequence deduplication applies to evaluation candidates. EC labels supervise training and are not encoder inputs at inference.

## Train

Once `data/processed/<split>/config.json` and the referenced feature caches are ready:

```bash
# Inspect all resolved arguments without starting training.
python scripts/train.py --split time --seed 42 \
  --output runs/time/full_seed42 --dry-run

# Run the full model (100 epochs).
python scripts/train.py --split time --seed 42 \
  --output runs/time/full_seed42 --device cuda:0
```

Use `--data-root` for another data directory, or pass `--reference-run`, `--source-dir` and `--residue-cache` explicitly. Every run needs a new output directory. `--cache-devices` and `--cache-gib` configure GPU residue caching; their comma-separated entries must have the same length. For another primary GPU, set both `--device` and `--cache-devices` to the intended logical CUDA indices. Training and checkpoint selection use training and validation data; both test directions use the same selected checkpoint.

## Evaluate both directions

```bash
python scripts/evaluate.py \
  --run runs/time/full_seed42 --split time --seed 42 \
  --output results/time/full_seed42 --device cuda:0
```

The evaluator resolves `test_contract.json` from the run's data reference, or accepts `--test-contract`. It requires a completed 100-epoch run and verifies the validation-selected checkpoint. It encodes the full contract candidate library, evaluates the score matrix and its transpose, and writes metrics and per-query ranks under `R-E/` and `E-R/`.

`Hit@k` is the proportion of queries with at least one recorded positive in the top k. The reported ReactZyme MR and MRR average ranks and reciprocal ranks over **all** recorded positives for each query, then over queries. They differ from first-positive MR/MRR. Stable candidate-ID order resolves ties. The evaluator checks ranks independently against NumPy sorting and the vendored official ReactZyme metric implementation.

## Tests and figures

```bash
python -m unittest discover -s tests -v
```

These tests use synthetic tensors and tiny fixtures to check implementation behavior. They do not reproduce benchmark scores or replace a training run. Plotting commands and input schemas are in [docs/figures.md](docs/figures.md). Each renderer requires the corresponding user-generated analysis tables; the figure scripts do not download results.

## Attribution

The ReactZyme metric source and its CC0-1.0 license are retained in `external/ReactZyme_official/`. See [THIRD_PARTY.md](THIRD_PARTY.md). No repository-wide license has been added for the HiER-authored code; the third-party license applies to the attributed files only.
