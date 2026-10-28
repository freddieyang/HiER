# Data preparation and frozen features

The release can convert the official ReactZyme positive archives into the HiER
split tables and retrieval contracts. Training additionally requires existing
frozen feature caches in the formats below. This adapter does **not** download
pretrained weights, extract those caches, or reconstruct them from the archives.
No benchmark data, feature tensors or model checkpoints are included in Git.

## Official inputs

Obtain ReactZyme's released data from the data links in the
[official ReactZyme repository](https://github.com/WillHua127/ReactZyme) and its
[Zenodo data record](https://zenodo.org/records/13635807).
Use the frozen source tables shipped with that release, not a later UniProt/Rhea
snapshot. Extract the three split archives into this layout:

```text
data/
  raw/ReactZyme/
    cleaned_uniprot_rhea.tsv
    uniprot_molecules.tsv
    time_split/
      positive_train_val_time.pt
      positive_test_time.pt
    enzyme_smi_split/
      positive_train_val_seq_smi.pt
      positive_test_seq_smi.pt
    reaction_smi_split/
      positive_train_val_mol_smi.pt
      positive_test_mol_smi.pt
  processed/                       # created by the command below
  features/                        # required separately for training
    reference.json
    unimol/
    residue_cache/
```

Each `.pt` archive must be a dictionary in its released insertion order, with
values `(molecule_set_smiles, amino_acid_sequence)`. The required TSV columns are
`Entry`, `Sequence`, `EC number` in `cleaned_uniprot_rhea.tsv`, and `uniprot_id`,
`molecules` in `uniprot_molecules.tsv`. The accessions are joined one-to-one.
Only positive archives are opened. Negative-pair archives and `rhea_molecules.tsv`
are not needed for this direction-free molecule-set representation.

```bash
python scripts/prepare_data.py \
  --raw-dir data/raw/ReactZyme \
  --output-dir data/processed
```

The output directory must not already exist. Use a different output directory
when attaching caches later; the script never silently replaces a processed set.
`--splits time` prepares just the time split; the default prepares all three.

For every split the command writes `pairs.parquet`, `enzymes.parquet`,
`reactions.parquet`, `validation_contract.json`, `test_contract.json`,
`evaluation_contract.json`, `identity_audit.json`, `ingestion_report.json` and
`config.json`. `inventory/` contains the shared reaction inventory and accession
inventory. Without cache arguments, `config.json` says `features_ready: false`
and lists missing inputs. Successful data conversion alone is not a trainable run.

## Preserved protocol

The official test membership is unchanged. For each train/validation archive,
the validation ordinals are exactly
`np.random.default_rng(42).permutation(n)[:round(0.1 * n)]`; the remaining rows
are training pairs. This is a pair-level split, not a new enzyme- or EC-class
holdout. All released positive rows are retained, including queries without a
usable EC supervision target.

Molecule strings replace `*` with `C` as in the original preprocessing. Reaction
IDs are `RZR_` plus the first 20 hexadecimal characters of the SHA-256 of that
string. Sequences remove spaces and are uppercased. Accessions matching the same
reaction/sequence key are sorted and assigned by occurrence in train/validation
then test archive order. Missing or ambiguous mappings stop conversion instead
of creating synthetic IDs or dropping rows. An exact reaction/sequence overlap
between train/validation and test is an error.

Training preserves separate enzyme IDs even when their sequences are identical.
The existing trainer derives EC supervision from training associations and the
frozen `EC number` column: a reaction target requires exactly one complete EC
and no partial annotation; enzyme targets follow their own training enzyme IDs.
Sequence identity is not used to merge EC supervision across different IDs.
Evaluation alone collapses identical full sequences within each evaluation pool,
using the lexicographically smallest accession as representative. The contract
retains all recorded relevant pairs after this identity deduplication. The same
score matrix can be transposed for enzyme-to-reaction retrieval.

## Attach existing frozen caches

The feature JSON is a data reference, not a previous training run. Its four
required keys are shown here. Relative paths are resolved against this JSON's
directory; the generated configuration records resolved paths on your machine.

```json
{
  "processed_dir": "inventory",
  "global_feature_bundle": "morgan/features.pt",
  "protein_global_embeddings": "esm2/protein_mean_embeddings.safetensors",
  "protein_global_ids": "esm2/protein_mean_embeddings.ids.json"
}
```

Here `features/inventory/` must contain the reaction and enzyme Parquet tables
used to create those caches. Its reactions need columns `reaction_id` and
`official_molecule_set_smiles`; its enzymes need `enzyme_id` and `sequence`.
These source inventories establish tensor identities and row order. They are
not the selected split's training or evaluation labels.

```bash
python scripts/prepare_data.py \
  --raw-dir data/raw/ReactZyme \
  --output-dir data/processed_with_features \
  --feature-reference data/features/reference.json \
  --unimol-dir data/features/unimol \
  --residue-cache data/features/residue_cache/manifest.json
```

`--unimol-dir` defaults to `unimol/` next to the JSON. `--residue-cache` accepts
either the global manifest or its directory. The adapter checks the source
inventories, row identities, tensor shapes, provenance manifests and residue
array size without loading that large array. It does not rerun a residue/mean
reconstruction audit; the source cache must already contain its completed audit.

The adapter also writes `frozen_feature_reference.json`, each split's
`protein_ids.json`, `protein_source_rows.npy` and `residue_view.json`. A missing
accession may reuse a cache row only if its full sequence matches exactly;
this maps feature rows without merging its training identity or labels.
With all checks passed, use the split directory as `--reference-run`, the
Uni-Mol directory as `--source-dir`, and the split's `residue_view.json` as
`--residue-cache`. `config.json` also contains these paths. For example, the
reference directory above is `data/processed_with_features/time`.

### Morgan cache

`features.pt` contains `reaction_ids`, `reaction_fingerprints` (N × 2048) and
`feature_contract`. The original representation is a radius-2 Morgan count
fingerprints summed over the component molecules, with the resulting vector
L2-normalized. The four
`feature_contract` flags `uses_pair_associations`, `uses_split_labels`,
`uses_enzyme_or_ec_labels` and `uses_reaction_direction` must all be false.

The sibling `manifest.json` must have `status: completed`, the actual
`source_reactions`, `feature_bundle`, `reaction_count`, and false values for
`pair_or_split_tables_read`, `enzyme_associations_read`,
`validation_or_test_associations_read`, `ec_labels_read`, and
`reaction_direction_used`. Do not assert these flags for features built from
labels or a different preprocessing recipe.

### Uni-Mol cache

`reaction_atomic_mean.pt` contains `entries` and `embeddings` (N × 512), aligned
to the source reaction inventory. `reaction_feature_manifest.json` must report
`status: completed`, `primary_output: reaction_atomic_mean.pt`,
`inputs.reactions.path`, and
`label_use: {reads_pair_labels: false, reads_test_labels: false}`. Its `shards`
list supplies the molecule-shard paths. Each shard contains `smiles` and
`atomic_mean` (number of molecules × 512). The loader uses those individual
molecule vectors to calculate training-derived IDF-weighted moments; a file of
reaction means alone is insufficient.

The formal cache uses Uni-Mol v1, `unimol_tools==0.1.6`, all hydrogens
retained (`remove_hs=False`), `max_atoms=256`, and seed 42. Its dictionary uses
the published file order; atom means exclude BOS, PAD and EOS and retain UNK
atom positions. The corresponding manifest identifies this as `published_mean`.
For molecules exceeding 256 atoms, the original crop is
`np.random.default_rng(np.random.SeedSequence([42, global_sorted_molecule_index])).choice(atom_count, 256, replace=False)`.
The original code excludes one exact
Fe-containing molecule string from the official ReactZyme notebook, rather
than applying a general element or size filter. The released core's
`extract_tiger_unimol_features.split_components` retains this exact exclusion.
Molecule CLS vectors are not interchangeable with these atomic means. Producing
a new cache requires the matching pretrained checkpoint and dictionary and the
same extraction recipe. The early generic `UniMolRepr` extraction is not a
replacement for this dictionary-compatible atom selection. This release
supplies the cache consumer, not a new validated extraction pipeline.

### ESM-2 mean and residue caches

The mean safetensors file must contain only `embeddings` (P × 1280), with rows
listed in `protein_mean_embeddings.ids.json` as `{"enzyme_ids": [...]}`.
`protein_mean_embeddings.report.json` is required next to it and records
`baseline: CLIPZyme-ESM-Frozen`, `explicit_negative_pairs_loaded: 0`,
`enzyme_count`, `ids_path`, `output`, and an existing `source_manifest`.

The global residue cache uses the frozen `facebook/esm2_t33_650M_UR50D` features
and the first at most 1,022 residues underlying the original mean features.
Its manifest names `cache_path` (raw FP16 matrix), `offsets_path` (NumPy offsets),
`ids_path`, `source_mean`, `shape: [total_retained_residues, 1280]`,
`dtype: float16`, `maximum_residues: 1022`, `status: completed`, and
`mean_reconstruction_checked_rows` equal to the number of protein IDs. The
`feature_contract` flags `reads_pair_labels`, `reads_ec_labels`, and
`reads_split_labels` are false and `retains_baseline_residue_window` is true.
The IDs and their order must match the mean cache. The complete source pipeline
also keeps the source shard manifest and its mean-reconstruction audit.

Do not substitute a CLS embedding, a whole-sequence mean computed with another
truncation rule, or a different ESM layer. The release currently consumes this
existing format; it does not provide an independently validated sequence-to-cache
extractor. The residue array can be large and is accessed through memory mapping.

### Paths and trust

The original runtime validates paths inside the Morgan, Uni-Mol, protein and
residue manifests. If caches are relocated, update their path fields to the
actual corresponding files; changing only `reference.json` is insufficient.
Keep the recorded scientific settings and provenance intact. These files are
not downloaded automatically. Load `.pt` feature bundles only from a trusted
source: the inherited feature loader supports the original PyTorch format and
may deserialize Python objects.
