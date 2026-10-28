# Analysis and figure scripts

The `analysis/` commands render figures from explicitly supplied analysis inputs.
They do not train models, select checkpoints, or evaluate test sets. This source
release contains no experiment outputs, case records, molecular structures,
projection coordinates, checkpoints, or figure assets. The commands below use
paths relative to the repository root; replace the example input paths with your
own prepared files.

All generated images, tables, aggregate CSVs, and diagnostic JSONs go to
`--output-dir`. It must be outside every input directory. Existing files with the
same output names can be overwritten, so use a fresh output directory when
comparing runs. Never run these scripts with Python's `-O` option: their input
checks use assertions.

Dependencies are NumPy, pandas, and Matplotlib. Molecular case rendering also
requires RDKit and CairoSVG, including its Cairo system library. The plotting
commands run on CPU. Each command supports `--help`.

## Figure coverage

Filenames retain the manuscript's existing asset names; their numeric prefixes
do not always equal the current figure numbers.

| Manuscript figure | Command | Output stem |
| --- | --- | --- |
| Figure 2: retrieval depth | `build_figure2.py` | `figure2_retrieval_depth_column` |
| Figure 3: paired query ranks | `build_figure3.py` | `figure3_rank_effect` |
| Figure 4: hierarchical matching | `build_hierarchy_figure.py` | `figure4_hierarchical_matching`, plus panels a–c |
| Figure 5: three loss-weight scans | `build_loss_sensitivity.py` | `figure6_loss_sensitivity` |
| Figure S1: molecular inventories | `build_figureS1.py` | `figureS1_case_inventories` |
| Figure S2: t-SNE display | `build_figureS2_tsne.py` | `figureS2_tsne` |
| Figure S3: additional rank corrections | `build_figure4.py --layout supplement` | Set `--stem figureS3_rank_corrections` |
| Figure 6 and Figure S4: case composition | `compose_case_studies.py` | `figure4_case_studies`, `figureS4_rank_corrections` |

Each renderer writes PDF, SVG, and PNG. Figure 1 was manually edited; this release
does not claim automatic reproduction of the final framework artwork. Figure 6
also needs an externally supplied editable SVG for its two bidirectional panels,
as detailed below. The figure layout code retains the manuscript's axis and
color limits and its case-layout capacities; inspect a rendered figure before
using it for a different dataset or score range.

## Preparing inputs from evaluation outputs

The evaluation command writes `embeddings.pt`, `summary.json`, and
directional `R-E/` and `E-R/` directories containing `metrics.json`,
`per_query_ranks.jsonl`, and `verification.json`. Add `--save-scores` to also
write `scores.npy`; the full matrix is not saved by default. The embedding bundle includes
`query_ids`, `candidate_ids`, `query_embeddings`, `candidate_embeddings`,
`logit_scale_factor`, `split`, `seed`, `checkpoint_epoch`, and
`ground_truth_ec_input=False`. Its query/candidate axes are reaction/enzyme;
enzyme-to-reaction evaluation uses the transposed score matrix.

Directional metric keys include `hit@1`, `hit@5`, `hit@10`, `hit@20`,
`official_reactzyme_mrr_all_relevant`, and `official_reactzyme_mean_rank`.
Per-query records include `query_id`, `positive_candidate_ids`,
`positive_ranks`, and `top_candidate_ids`. Use the saved ordering and recorded
positive sets when preparing analysis inputs. Hit@k denotes whether any recorded
positive occurs in the first k candidates; it is not the fraction of positives
retrieved. MR is the mean of positive ranks within a query, averaged over queries.
MRR similarly averages their reciprocals, rather than taking only the first
positive.

The renderers do not automatically turn these outputs into every analysis input.
Aggregate matching split/direction/seed records as specified below. Fixed-model
comparisons additionally need latent-only scores or separate EC score blocks
from the same checkpoint; a retrained variant is not a replacement. Posterior
trees need model-predicted distributions and the training vocabulary. Case
depictions need externally supplied molecular inventories and annotations.
Projection fitting and case selection are separate preparation steps, not
operations performed by these commands.

### Retrieval-depth curves

```bash
python analysis/build_figure2.py --data-dir inputs/depth --output-dir results/figures/depth
```

Supply `local_baseline_comparison.csv` with one row for each combination of:

- `split`: `time`, `seq_smi`, `mol_smi`;
- `direction`: `E-R`, `R-E`;
- `method`: `HiER`, `TIGER_ESM2Text_local`, `TIGER_ProtT3_local`.

Required metadata columns are `seed_count`, `training_epochs`, `query_unit`,
`query_count`, and `candidate_count`. The paper layout expects three seeds and
100 training epochs. Required numeric columns are `hit1_mean`, `hit1_std`,
`hit5_mean`, `hit5_std`, `hit10_mean`, `hit10_std`, `hit20_mean`, and `hit20_std`.
Store these metrics as fractions, not percentages. Prepare the mean and sample
standard deviation (`ddof=1`) from the three directional evaluation records for
each method and split. Baseline records must come from the corresponding
baseline evaluations; this renderer does not run them. It also writes
`figure2_values.csv` in the output directory.

### Paired ranks from fixed models

```bash
python analysis/build_figure3.py --data-dir inputs/ranks --output-dir results/figures/ranks
```

Supply these two CSVs:

- `figure3_rank_pairs.csv`: `query_id`, `direction`, `latent_mr`, `full_mr`,
  `rank_change`, `seed_count`;
- `figure3_rank_summary.csv`: `direction`, `N`, `C`, `improved_percent`,
  `unchanged_percent`, `worsened_percent`.

For each query, compute its mean positive rank separately for each fitted model
and scoring condition, then average over the three paired seeds. Classify the
change in that query mean as `Improved`, `Unchanged`, or `Worsened`, with the
preparation step's numerical tie tolerance applied consistently. Count queries
in each category to prepare the percentages and `N`; `C` is the unchanged
candidate-library size. Do not substitute Hit@1 changes for these rank categories.
The renderer uses every supplied query, verifies the category percentages, and
writes `figure3_display_validation.json` alongside the plot.

### Hierarchical matching and score removal

```bash
python analysis/build_hierarchy_figure.py --data-dir inputs/hierarchy --output-dir results/figures/hierarchy
```

Supply four CSVs:

| File | Required columns |
| --- | --- |
| `case_tree_nodes.csv` | `entity_id, level, node, parent, probability, other` |
| `case_score_components.csv` | `candidate_id, recorded_positive, latent_rank, full_rank, archived_full_score, latent, L1, L2, L3, L4` |
| `metrics_per_seed.csv` | `split, seed, direction, condition, delta_hit1, delta_mrr` |
| `metrics_summary.csv` | `split, direction, condition, delta_hit1_mean, delta_hit1_sd, delta_mrr_mean, delta_mrr_sd` |

Boolean fields must be `True`/`False` or `1`/`0`. Tree rows are grouped in display
order: one reaction query followed by two enzyme candidates. At L1–L3, include
one selected branch node and one `other` node per level. At L4, include four
selected leaves and their remaining mass. Use the same nodes for all three
entities; `root` is the implicit parent above L1. `Other:` node identifiers
identify aggregated branches. The probabilities are full-distribution masses,
not probabilities renormalized over just the displayed nodes. The candidate IDs
in the two enzyme trees must occur in the score table.

`latent` and `L1`–`L4` are additive contributions to the scaled final score:
`gamma*(1-alpha)*s_latent` and `gamma*alpha*B_l/4`. Their sum must agree with
`archived_full_score`. Ranks use the complete candidate library, not the displayed
subset. A recorded positive is an annotated association, not a claim that every
unrecorded candidate is experimentally negative.

For each split, direction, seed (42, 43, 44), and condition `without_L1` through
`without_L4`, form the score by subtracting only the indicated additive block.
Retain all other coefficients, the candidate library, positive sets, and tie
ordering; do not renormalize or retrain. Each delta is the full-score metric minus
the corresponding score-removal metric. `delta_hit1` is a fraction and is
converted to percentage points for display. Summaries are means and sample SDs
of paired seed differences, which the renderer checks against `metrics_per_seed`.
The command writes the combined figure, three individual panels,
`table_hierarchy_scores.tex`, and a text-layout diagnostic JSON.

### Three loss-weight sensitivity scans

```bash
python analysis/build_loss_sensitivity.py --data-dir inputs/loss_sensitivity --output-dir results/figures/loss_sensitivity
```

Supply `validation_per_seed.csv` directly inside `--data-dir`, with columns:
`axis,value,split,seed,subset,direction,query_unit,N,C,Hit1_percent,MRR,MR,source`.
`source` is a run identifier, not a file path to be opened. `subset` must be `VAL`;
the directions use `query_unit` values `enzyme` and `reaction`. `N` and `C` are
query and candidate counts. `Hit1_percent` uses percent units; `MRR` is a fraction.
Map the directional validation metrics to those units without substituting test
metrics or choosing new coefficients from test performance.

The renderer expects the manuscript's one-parameter scans on all three splits
and seeds 42–44:

- `beta`: 0, 0.1, 0.2, 0.5, 1;
- `lambda_ec`: 0, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9;
- `lambda_align`: 0, 0.01, 0.02, 0.04, 0.1, 0.2, 0.5, 1.

The reported reference settings are 0.2, 0.1, and 0.02, respectively; the other
two coefficients stay fixed in each scan. The common reference run is reused
across all three axes, with matching metrics and the same `source` identifier.
The full input has 396 directional rows describing 180 distinct runs. This is an
input contract, not a statement that the release has executed these runs.

For each run, compute the geometric mean of the two directional Hit@1 values
and separately the geometric mean of MRR. Average these over the three splits
within each seed; plot the three-seed mean with sample SD (`ddof=1`) across those
seed averages. The command performs this aggregation without changing the
reference settings. It writes the figure, per-run and per-seed aggregate CSVs,
`aggregate_validation.csv`, `table_lambda_ec.tex`, `table_lambda_align.tex`, and
environment metadata. It checks supplied records, not external training logs.

### t-SNE display

```bash
python analysis/build_figureS2_tsne.py --data-dir inputs/projection --output-dir results/figures/projection
```

Supply `figure3_tsne_coordinates.csv` with `entity_id`, `entity_type`,
`reference_L1`, `displayed`, `latent_tsne_1`, `latent_tsne_2`, `joint_tsne_1`, and
`joint_tsne_2`. Entity types are `Reaction` and `Enzyme`; the pair of type and ID
must be unique. `displayed` must be true for every row. `reference_L1` is one of
`1`–`7` or `Unresolved`. An entity with conflicting or unresolved reference roots
belongs to `Unresolved`, rather than being removed.

Fit the projections beforehand from latent and joint retrieval vectors of the
same fitted model and entity set. In the model's concatenated representation,
the latent block is scaled by `sqrt(1-alpha)`; recover the latent vectors using
the fitted alpha and the configured latent dimension, rather than assuming a
fixed total embedding dimension. Use reference annotations only for coloring,
not for fitting coordinates. Record the t-SNE parameters and fitting entity set
with your analysis. This command only renders supplied coordinates, draws every
row, and reports display counts; it cannot verify how the projection was fitted.

### Molecular case renderers

```bash
python analysis/build_figure4.py --case-data inputs/cases/case_expansion_main.json --display-data inputs/cases/case_molecular_display.json --output-dir results/correction_sources
python analysis/build_figure4.py --case-data inputs/cases/case_expansion_supplement.json --display-data inputs/cases/case_molecular_display.json --layout supplement --stem figureS3_rank_corrections --output-dir results/figures/corrections
python analysis/build_figureS1.py --case-data inputs/cases/successful_cases.json --display-data inputs/cases/case_molecular_display.json --output-dir results/figures/inventories
```

These JSON files are inputs, not bundled assets:

- A rank-correction file has a top-level `cases` list of one to three cases. Each
  case contains `case` (panel letter), `reaction_id`, `raw_molecular_input`,
  `canonical_molecules`, `recorded_ec`, `query_predictions` (L1–L4 objects with a
  `label`), and `candidates`. Candidate fields are `enzyme_id`, `recorded_positive`,
  `recorded_ec`, `latent_rank`, `full_rank`, `latent_score`, `ec_scores` (four
  scaled terms), and `total_score`. The selected correction must move a recorded
  positive to rank 1 from a lower latent rank; the prior top candidate must be
  unrecorded, with non-overlapping reference ECs. These selection checks describe
  the depicted case type, not an overall evaluation criterion.
- The molecular-display file has `cases`, a mapping keyed by reaction ID. Each
  entry contains `source_molecular_inventory` and `display_molecules`, a list of
  source SMILES components. Preserve unspecified atoms for depiction. The script
  verifies that replacing `*` by `C` in the source inventory reproduces the saved
  model input, matching the original data preparation convention; it does not
  use that replacement for the displayed structures. Supported correction
  layouts have two to five components.
- `successful_cases.json` has a `reactions` list. The S1 layout uses entries 1 and
  2 (zero-based) as R2/R3, each containing `reaction_id`, `annotated_EC_union`, and
  `raw_input`, with exactly two display components. The case composer additionally
  selects the entry with `is_RE_query=true` for its reaction-to-enzyme panel.

Obtain ranks and positive flags from full-library evaluation, score terms from
the fixed model, and reference ECs from the evaluation annotations. Keep those
reference annotations distinct from `query_predictions`. Structural drawings
show molecular context and do not supply new supervision or establish catalytic
validation of an unrecorded pair.

### Case composition requires an editable SVG source

```bash
python analysis/compose_case_studies.py --source-dir inputs/case_svg --data-dir inputs/cases --output-dir results/figures/case_studies
```

The source directory must contain:

1. `figure4_rank_correction.svg`, generated from three main-layout correction
   cases by `build_figure4.py`, with a `0 0 1000 875` viewBox and panels a–c.
2. `figure5_bidirectional_examples.svg`, an externally prepared editable layout
   with a `0 0 1450 588` viewBox. It contains the enzyme-to-reaction and
   reaction-to-enzyme panels, their scores and ranks, and any structure cartoons.
   This release does not generate this second SVG from evaluator outputs.

The compositor retains the existing SVG layout contract: headings
`a  Enzyme → reaction` and `b  Reaction → enzyme`; molecular inventory labels
`R1 · complete molecular input` and `Query R4 · complete molecular input`; and
two R4 molecule SVG slots at y=133. It combines the first correction row with
the two supplied panels to form Figure 6, and uses correction rows two and three
to form Figure S4. It updates the inventory depictions from the display records
and writes a composition manifest. It does not validate scores already embedded
in the supplied SVG against the evaluation records; that correspondence must be
checked when preparing the source artwork. Without this SVG, the standalone
rank-correction and inventory renderers still work, but complete Figure 6
reproduction remains unavailable.
