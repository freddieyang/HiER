#!/usr/bin/env python3
"""Controlled training budget or reverse-pool extension of the same residue model.

Only TRAIN associations update parameters or fit molecule statistics. VAL is
reserved for checkpoint selection. This script has no test or refit option.
Original trainers and completed studies are preserved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.morgan_dropout_schedule import morgan_probability_for_epoch
from ec_retrieval.matched_ablation import ARMS, ec_objectives
from ec_retrieval.coarse_ec_targets import load_train_consensus, supplement_reaction_targets
from ec_retrieval.bidirectional_loss import bidirectional_retrieval_loss
from ec_retrieval.reverse_pool_extension import ReverseReactionPool, isolated_extra_rng, expanded_bidirectional_loss
from ec_retrieval.latent_retrieval_supervision import add_latent_retrieval_supervision
from ec_retrieval.proposal_supervision import add_proposal_ec_supervision
from ec_retrieval.parameter_ema import ParameterEMA
from ec_retrieval.protein_ec_supervision import load_train_enzyme_annotations, fill_missing_protein_coarse_targets
from ec_retrieval.input_ambiguity import TrainInputAmbiguity, exclude_neutral_comparisons
from ec_retrieval.isolated_context_training import isolated_context_objective
from ec_retrieval.controlled_budget import step_prefix_scheduler, save_training_state, verify_prefix
from ec_retrieval.bidirectional_validation import add_reverse_validation
from ec_retrieval.training_candidates import (TrainCandidateDomain,
    mine_current_train_candidates, sample_training_candidates)
from ec_retrieval.residue_readout import ResidueReadoutRetriever
from ec_retrieval.candidate_background import make_residue_retriever, validation_selection_key
from ec_retrieval.prefetch_residue_store import PrefetchResidueFeatureStore
from ec_retrieval.sharded_residue_store import ShardedResidueFeatureStore, training_cache_priority
from ec_retrieval.molecule_store import MoleculeFeatureStore
from ec_retrieval.target_store import ECTargetStore
from ec_retrieval.official_split_view import validate_split_view
from ec_retrieval.molecule_pooling import fit_molecule_document_frequency, smooth_molecule_idf, pool_molecule_features
from ec_retrieval import (multi_positive_tail_risk_loss, hierarchical_multi_positive_loss,
                         hierarchy_consistency_loss, cross_modal_ec_alignment_loss)
from scripts.extract_tiger_unimol_features import split_components
from scripts.train_function_transfer_retriever import load_frozen_features, evaluate
from scripts.train_global_anchored_retriever import atomic_torch_save, validate_train_only_protocol
from scripts.train_structured_retriever import (
    atomic_json, append_jsonl, seed_everything, amp_policy, load_contract,
    validate_official_inputs, load_official_ec_annotations, infer_ec_depth,
    build_ec_vocab, target_tokens_for_ids, protein_target_tokens, dense_target_masks,
    build_train_adjacency, sample_query_balanced_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-ablation", choices=("none", *ARMS), default="none")
    parser.add_argument("--protein-readout", choices=("mean", "residue_attention"), required=True)
    parser.add_argument("--residue-cache", type=Path, required=True)
    parser.add_argument("--residue-prefetch-depth", type=int, choices=(0,1,2), default=2,
                        help="Queued TRAIN residue uploads; 0 is serial control; VAL remains synchronous")
    parser.add_argument("--resident-cache-devices", required=True, help="Comma-separated logical CUDA indices, in cache priority order")
    parser.add_argument("--resident-cache-gib", required=True, help="Comma-separated positive GiB budgets aligned with cache devices")
    parser.add_argument("--cache-main-headroom-gib", type=float, default=10.)
    parser.add_argument("--cache-helper-headroom-gib", type=float, default=2.)
    parser.add_argument("--residue-mix", type=float, default=.5)
    parser.add_argument("--residue-block-size", type=int, default=128)
    parser.add_argument("--reference-run", type=Path, required=True, help="Existing audited TRAIN-only config, not its weights")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/features/unimol")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pooling", choices=("moments", "attention", "ec_attention"), default="ec_attention")
    parser.add_argument("--ec-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion-sharing", choices=("independent_matched", "shared"), required=True,
                        help="Both variants start with the same fusion weights; only shared ties them throughout training")
    parser.add_argument("--drop-morgan-probability", type=float, default=0., help="TRAIN-only whole-view masking; never mask both views")
    parser.add_argument("--drop-unimol-probability", type=float, default=0.)
    parser.add_argument("--morgan-dropout-schedule", choices=("constant", "linear_decay"), required=True)
    parser.add_argument("--reaction-ec-supervision", choices=("strict",), required=True)
    parser.add_argument("--protein-ec-supervision", choices=("reaction_consensus", "fill_missing_official_coarse"),
                        default="reaction_consensus", help="Optional TRAIN-only unique L1-L3 enzyme annotations for otherwise unsupervised proteins")
    parser.add_argument("--training-comparison-policy", choices=("original", "input_ambiguity_neutral"),
                        default="original", help="Optionally exclude TRAIN input-ambiguous unobserved comparisons; no new positives")
    parser.add_argument("--reaction-context", choices=("none", "molecule_set"), default="none",
                        help="Optional intrareaction molecule-set residual before EC prediction; no new inference inputs")
    parser.add_argument('--atomic-readout', choices=('none', 'conditional'), default='none',
                        help='Reaction-context attention over cached published-dictionary atoms')
    parser.add_argument('--atomic-cache', type=Path,
                        help='Audited paired atom cache; required only for conditional atomic readout')
    parser.add_argument('--atom-attention-dim', type=int, default=64)
    parser.add_argument('--atomic-mix', type=float, default=.5)
    parser.add_argument('--atomic-logit-bound', type=float, default=0.,
                        help='Zero preserves raw atom attention; positive c applies c*tanh((logits-component_mean)/c)')
    parser.add_argument("--context-hidden-dim", type=int, default=128)
    parser.add_argument("--context-heads", type=int, default=2)
    parser.add_argument("--context-gradient-mode", choices=("joint", "isolated"), default="joint",
                        help="Isolated mode trains a residual objective without changing reference-route gradients")
    parser.add_argument("--tail-selector", choices=("raw",), required=True,
                        help="Choose difficult queries by original CE or sampled excess rank, preserving CE gradients")
    parser.add_argument("--candidate-sampling", choices=("random", "dynamic"), required=True)
    parser.add_argument("--positive-coverage", choices=("legacy", "factorized"), required=True,
                        help="Replace all-positive and tail risk only; preserve any-positive branch and mixture weights")
    parser.add_argument("--reverse-retrieval-weight", type=float, choices=(0., .5), default=0.,
                        help="0 preserves R-to-E training; 0.5 equally averages the existing objective in both directions")
    parser.add_argument("--mining-warmup-epochs", type=int, default=5)
    parser.add_argument("--mining-refresh-epochs", type=int, default=5)
    parser.add_argument("--mining-bank-size", type=int, default=64)
    parser.add_argument("--mined-per-query", type=int, default=4)
    parser.add_argument("--mining-query-batch-size", type=int, default=64)
    parser.add_argument("--mining-protein-batch-size", type=int, default=2048)
    parser.add_argument("--random-reserve-fraction", type=float, default=.25,
                        help="Reserve up to this pool fraction after preserving direct positives")
    parser.add_argument("--cache-ec-targets", action=argparse.BooleanOptionalAction, default=False,
                        help="Preload TRAIN boolean EC targets once to avoid per-step Python target assembly")
    parser.add_argument("--scoring", choices=("explicit",), required=True)
    parser.add_argument("--ec-posterior", choices=("legacy", "leaf_marginal"), required=True)
    parser.add_argument("--ec-score-depth", type=int, default=3)
    parser.add_argument("--initial-ec-weight", type=float, default=.25)
    parser.add_argument("--candidate-background", choices=("none", "learned"), default="none",
                        help="Optional single TRAIN-learned latent vector for feature-dependent candidate bias")
    parser.add_argument("--background-max-norm", type=float, default=1.,
                        help="Positive bound for the learned background; fixed before experiments")
    parser.add_argument("--checkpoint-selection", choices=("mean_rank", "hit1_mrr", "bidirectional_hit1_mrr"), default="mean_rank",
                        help="Full VAL selection: historical MR, R-E head, or geometric mean of both directions' Hit1 then MRR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--checkpoint-milestones", type=int, nargs="*", default=[],
                        help="Completed epoch counts at which to retain the best VAL checkpoint so far; no effect on updates")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--positives-per-query", type=int, default=16)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--temperature", type=float, default=.07)
    parser.add_argument("--all-positive-weight", type=float, default=.2)
    parser.add_argument("--tail-risk-weight", type=float, default=0.,
                        help="Query-tail emphasis; disabled in the current HiER objective. Use 0.1 only to reproduce archived runs.")
    parser.add_argument("--tail-risk-fraction", type=float, default=.2)
    parser.add_argument("--ec-loss-weight", type=float, default=.1)
    parser.add_argument("--proposal-ec-weight", type=float, default=0.,
                        help="Optional TRAIN-only L1-L3 parent-proposal supervision; 0 is exact bypass")
    parser.add_argument("--consistency-loss-weight", type=float, default=.02)
    parser.add_argument("--cross-modal-ec-weight", type=float, default=.02)
    parser.add_argument("--latent-aux-weight", type=float, default=0.,
                        help="Direct latent-only retrieval supervision; zero exactly bypasses the auxiliary path")
    parser.add_argument("--latent-aux-temperature", type=float, default=.07,
                        help="Fixed auxiliary temperature, independent of the learned primary score scale")
    parser.add_argument("--ema-half-life-epochs", type=float, default=0.,
                        help="0 bypasses EMA exactly; positive values select an independently averaged single model")
    parser.add_argument("--ema-warmup-epochs", type=int, default=3,
                        help="Copy online parameters through these completed TRAIN epochs before averaging")
    parser.add_argument("--aux-warmup-epochs", type=int, default=3)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--eval-query-batch-size", type=int, default=128)
    parser.add_argument("--eval-protein-batch-size", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Smoke only; 0 processes all TRAIN queries")
    parser.add_argument("--schedule-epochs", type=int, default=0,
                        help="Original cosine/masking horizon; 0 uses epochs; later epochs hold terminal LR and masking")
    parser.add_argument("--reverse-reaction-pool-size", type=int, default=0,
                        help="0 preserves the legacy batch; 512 samples TRAIN reactions; -1 uses all FIT-TRAIN reactions in E-R")
    parser.add_argument("--reverse-loss-reduction", choices=("enzyme_mean", "reaction_macro"), default="enzyme_mean",
                        help="Average reverse CE over enzyme anchors (legacy) or over their observed TRAIN reaction groups")
    parser.add_argument("--prefix-control-run", type=Path,
                        help="Require bitwise prefix reproduction before extending a completed control")
    return parser.parse_args()


def load_component_features(reference, source, device):
    """Reconstruct the audited official feature means before exposing tokens."""
    frozen_reference = (json.loads(Path(reference['frozen_feature_reference']).read_text())
                        if reference.get('official_split_view') else reference)
    qids, pids, original_mean, proteins, audit = load_frozen_features(
        source / "reaction_atomic_mean.pt", frozen_reference, torch.device("cpu"))
    if reference.get('official_split_view'):
        audit['split_view'] = validate_split_view(reference)
        protein_rows = np.load(reference['protein_row_mapping'])
        pids = json.loads(Path(reference['protein_view_ids']).read_text())['enzyme_ids']
        proteins = proteins.index_select(0, torch.from_numpy(protein_rows))
    manifest = json.loads((source / "reaction_feature_manifest.json").read_text())
    molecules, vectors = [], []
    for entry in manifest["shards"]:
        shard = torch.load(entry["path"], map_location="cpu", weights_only=False)
        molecules.extend(shard["smiles"])
        vectors.append(shard["atomic_mean"].float())
    vectors = torch.cat(vectors)
    if len(set(molecules)) != len(molecules) or vectors.shape != (len(molecules), 512):
        raise ValueError("Invalid audited per-molecule cache")
    processed = Path(reference["processed_dir"])
    frame = pd.read_parquet(processed / "reactions.parquet", columns=["reaction_id", "official_molecule_set_smiles"])
    if frame.reaction_id.duplicated().any() or set(frame.reaction_id.astype(str)) != set(qids):
        raise ValueError("Official reaction metadata differs from cached rows")
    components = {str(r.reaction_id): split_components(str(r.official_molecule_set_smiles))[0]
                  for r in frame.itertuples(index=False)}
    train_ids = sorted(set(pd.read_parquet(processed / "pairs.parquet", columns=["reaction_id"],
        filters=[("split", "==", "train")]).reaction_id.astype(str)))
    counts, n_train = fit_molecule_document_frequency(components, train_ids)
    weights = smooth_molecule_idf(molecules, counts, n_train)
    lookup = {m: i for i, m in enumerate(molecules)}
    component_rows = [[lookup[m] for m in components[q]] for q in qids]
    lengths = [len(rows) for rows in component_rows]
    reconstructed = pool_molecule_features(vectors, [v for row in component_rows for v in row], lengths)
    error = float((reconstructed - original_mean).abs().max())
    if error > 2e-6:
        raise ValueError(f"Token cache fails original mean reconstruction: {error}")
    bundle = torch.load(frozen_reference["global_feature_bundle"], map_location="cpu", weights_only=False)
    if list(map(str, bundle["reaction_ids"])) != qids:
        raise ValueError("Morgan rows differ from the audited molecule rows")
    contract = bundle["feature_contract"]
    if any(contract.get(k, True) for k in ("uses_pair_associations", "uses_split_labels", "uses_enzyme_or_ec_labels", "uses_reaction_direction")):
        raise ValueError("Morgan bundle violates the reaction-only input contract")
    store = MoleculeFeatureStore(vectors, component_rows, weights, bundle["reaction_fingerprints"], device)
    audit.update(original_mean_reconstruction_max_abs=error, unique_molecules=len(molecules),
        total_components=sum(lengths), maximum_components=max(lengths),
        molecule_fitting_splits=["train"], train_frequency_queries=n_train,
        official_component_multiset_preserved=True, pretrained_encoders_frozen=True,
        feature_statistics_source="TRAIN reactions only; no pair labels in model input")
    frequency = {"training_splits": ["train"], "train_ids": train_ids,
                 "document_frequency": counts, "train_query_count": n_train}
    return qids, pids, store, proteins.to(device), audit, frequency


def validate_args(args):
    if not math.isfinite(args.atomic_logit_bound) or args.atomic_logit_bound < 0:
        raise ValueError('Atomic logit bound must be finite and nonnegative')
    if args.atomic_logit_bound and args.atomic_readout == 'none':
        raise ValueError('An atomic logit bound requires conditional atomic readout')
    if args.atomic_readout != 'none':
        if (args.atomic_cache is None or args.atom_attention_dim < 1 or args.atomic_mix != .5
                or args.reaction_context != 'none' or args.candidate_background != 'none'
                or args.proposal_ec_weight or args.latent_aux_weight or args.ema_half_life_epochs
                or args.ec_posterior != 'leaf_marginal' or args.pooling != 'moments'
                or args.drop_unimol_probability != 0 or args.reverse_reaction_pool_size != 512
                or args.reverse_loss_reduction != 'enzyme_mean'
                or args.training_comparison_policy != 'original'
                or args.protein_ec_supervision != 'reaction_consensus' or args.positive_coverage != 'legacy'):
            raise ValueError('Isolate conditional atomic readout with original losses and reverse512')
    elif args.atomic_cache is not None:
        raise ValueError('Do not load atoms for a disabled atomic readout')
    if not math.isfinite(args.proposal_ec_weight) or args.proposal_ec_weight < 0:
        raise ValueError('Proposal EC weight must be finite and nonnegative')
    if args.proposal_ec_weight and (
            args.ema_half_life_epochs or args.latent_aux_weight or args.reverse_reaction_pool_size != 512
            or args.reverse_loss_reduction != 'enzyme_mean' or args.protein_ec_supervision != 'reaction_consensus'
            or args.training_comparison_policy != 'original' or args.candidate_background != 'none'
            or args.reaction_ec_supervision != 'strict' or args.positive_coverage != 'legacy'
            or args.reaction_context != 'none' or args.ec_posterior != 'leaf_marginal'):
        raise ValueError('Isolate parent-proposal supervision on unchanged reverse512 marginal4')
    if args.context_gradient_mode == 'isolated' and args.reaction_context != 'molecule_set':
        raise ValueError('Isolated context gradients require the explicit molecule-set branch')
    if args.reaction_context != 'none' and (
            args.ema_half_life_epochs or args.latent_aux_weight or args.reverse_reaction_pool_size != 512
            or args.reverse_loss_reduction != 'enzyme_mean' or args.protein_ec_supervision != 'reaction_consensus'
            or args.training_comparison_policy != 'original' or args.candidate_background != 'none'
            or args.reaction_ec_supervision != 'strict' or args.positive_coverage != 'legacy'
            or args.context_hidden_dim != 128 or args.context_heads != 2):
        raise ValueError('Isolate the prespecified 128-dim/two-head context branch on original reverse512')
    if args.training_comparison_policy=='input_ambiguity_neutral' and (
            args.ema_half_life_epochs or args.latent_aux_weight or args.reverse_reaction_pool_size!=512
            or args.reverse_loss_reduction!='enzyme_mean' or args.protein_ec_supervision!='reaction_consensus'
            or args.reaction_ec_supervision!='strict' or args.positive_coverage!='legacy'):
        raise ValueError("Isolate neutral TRAIN comparisons on original reverse512 with unchanged targets and loss weights")
    if args.protein_ec_supervision=='fill_missing_official_coarse' and (
            args.ema_half_life_epochs or args.latent_aux_weight or args.reverse_reaction_pool_size!=512
            or args.reverse_loss_reduction!='enzyme_mean'):
        raise ValueError("Isolate enzyme supervision on the original reverse512 model without EMA or other loss changes")
    if (not math.isfinite(args.ema_half_life_epochs) or args.ema_half_life_epochs < 0
            or args.ema_warmup_epochs < 0):
        raise ValueError("EMA half-life/warmup must be finite and nonnegative")
    if args.ema_half_life_epochs and (args.latent_aux_weight or args.reverse_loss_reduction != 'enzyme_mean'
            or args.reverse_reaction_pool_size != 512 or args.amp_dtype != 'bfloat16'):
        raise ValueError("Isolate EMA on unchanged reverse512 BF16 training, with no other experimental loss")
    if (not math.isfinite(args.latent_aux_weight) or args.latent_aux_weight < 0
            or not math.isfinite(args.latent_aux_temperature) or args.latent_aux_temperature <= 0):
        raise ValueError("Latent auxiliary weight/temperature must be finite and nonnegative/positive")
    if (len(set(args.checkpoint_milestones)) != len(args.checkpoint_milestones)
            or any(v < 1 or v > args.epochs for v in args.checkpoint_milestones)):
        raise ValueError("Checkpoint milestones must be distinct completed epochs within the training budget")
    devices=[int(v) for v in args.resident_cache_devices.split(',')]
    budgets=[float(v) for v in args.resident_cache_gib.split(',')]
    if len(devices)!=len(budgets) or len(set(devices))!=len(devices) or any(v<=0 for v in budgets):
        raise ValueError("Explicit distinct cache devices and positive budgets required")
    if args.residue_prefetch_depth==0 or args.candidate_sampling!='random':
        raise ValueError("This bounded cache check requires prefetch and the unchanged random sampler")
    if args.protein_readout != "residue_attention":
        raise ValueError("Prefetch training entry is scoped to the existing residue-attention model")
    if args.residue_mix != .5 or args.residue_block_size < 1 or args.ec_posterior != "leaf_marginal" or args.ec_score_depth != 4:
        raise ValueError("Fix half attentive readout and the preselected marginal4 model")
    for name in ("epochs", "query_batch_size", "positives_per_query", "candidate_pool_size",
                 "projection_hidden_dim", "embedding_dim", "num_threads", "eval_query_batch_size", "eval_protein_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_steps_per_epoch < 0 or args.aux_warmup_epochs < 0:
        raise ValueError("Step cap and warmup must be nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 <= args.dropout < 1 or args.temperature <= 0 or not 0 < args.tail_risk_fraction <= 1:
        raise ValueError("Invalid dropout, temperature or tail fraction")
    if any(not 0 <= getattr(args, key) <= 1 for key in ("all_positive_weight", "tail_risk_weight")):
        raise ValueError("Retrieval mixture weights must lie in [0,1]")
    if any(getattr(args, key) < 0 for key in ("ec_loss_weight", "consistency_loss_weight", "cross_modal_ec_weight")):
        raise ValueError("Auxiliary weights must be nonnegative")
    if not (0 <= args.drop_morgan_probability <= 1 and 0 <= args.drop_unimol_probability <= 1
            and args.drop_morgan_probability + args.drop_unimol_probability <= 1):
        raise ValueError("Invalid reaction view-drop probabilities")
    if args.pooling != "moments" and args.drop_morgan_probability + args.drop_unimol_probability > 0:
        raise ValueError("View masking is defined for moments pooling only")
    if args.mining_warmup_epochs < 0 or min(args.mining_refresh_epochs, args.mining_bank_size,
            args.mined_per_query, args.mining_query_batch_size, args.mining_protein_batch_size) < 1:
        raise ValueError("Invalid candidate mining schedule or batch size")
    if not 0 <= args.random_reserve_fraction <= 1:
        raise ValueError("Invalid random reserve")
    if args.positive_coverage != "legacy" or args.ec_score_depth not in (3,4) or args.initial_ec_weight != .25:
        raise ValueError("Fix legacy loss, depth 3 or 4, and initial EC weight 0.25")
    if args.ec_posterior == "leaf_marginal" and args.ec_score_depth != 4:
        raise ValueError("The predeclared marginal candidate uses four EC levels")
    if args.candidate_sampling != "random" or args.candidate_pool_size != 1024:
        raise ValueError("Positive-coverage study fixes random TRAIN candidates and pool size 1024")
    morgan_probability_for_epoch(args.morgan_dropout_schedule, args.drop_morgan_probability, 0, args.epochs)
    if args.fusion_sharing != "independent_matched" or args.drop_unimol_probability != 0:
        raise ValueError("Schedule study requires independent fusion and unmasked Uni-Mol")


def retain_milestone(output, completed_epochs, best_epoch, best_metrics, selection_policy="mean_rank"):
    """Copy the selected checkpoint without touching model, optimizer, or RNG."""
    target = output / f"best_within_{completed_epochs:03d}.pt"
    if target.exists() or not 0 <= best_epoch < completed_epochs:
        raise ValueError("Invalid or previously retained checkpoint milestone")
    temporary = target.with_suffix(".pt.tmp")
    shutil.copyfile(output / "best.pt", temporary)
    temporary.replace(target)
    atomic_json(output / f"milestone_{completed_epochs:03d}.json", dict(
        completed_epochs=completed_epochs, selected_epoch_zero_based=best_epoch,
        checkpoint=str(target), validation_metrics=best_metrics,
        selection={"mean_rank":"minimum_full_validation_mean_rank_within_prefix",
                   "hit1_mrr":"maximum_full_validation_hit1_then_mrr_then_minimum_mr_within_prefix",
                   "bidirectional_hit1_mrr":"maximum_full_validation_bidirectional_hit1_geomean_then_mrr_geomean_within_prefix"}[selection_policy],
        model_updates_changed=False, test_used=False))


def main():
    args = parse_args()
    validate_args(args)
    requested_reverse_pool_size = args.reverse_reaction_pool_size
    args.schedule_epochs = args.schedule_epochs or args.epochs
    if not 2 <= args.schedule_epochs <= args.epochs:
        raise ValueError("Schedule horizon must be between 2 and the total epoch budget")
    if args.reverse_reaction_pool_size not in (0,512,-1) or args.reverse_retrieval_weight != .5:
        raise ValueError("Fix symmetric loss; reverse pool is 0, 512, or -1 for all FIT-TRAIN reactions")
    if args.reverse_loss_reduction=='reaction_macro' and (requested_reverse_pool_size!=-1 or args.latent_aux_weight!=0):
        raise ValueError("Reaction balancing is an isolated full TRAIN pool comparison without latent auxiliary loss")
    if args.reverse_reaction_pool_size and args.epochs != args.schedule_epochs:
        raise ValueError("Do not combine duration and reverse-pool changes in this study")
    if args.epochs > args.schedule_epochs and args.prefix_control_run is None:
        raise ValueError("Long training requires a verified unchanged control prefix")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no formal training was started. Use --device cpu only explicitly.")
        torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an experiment directory")
    torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    reference = json.loads((args.reference_run / "config.json").read_text())
    validate_train_only_protocol(reference.get("refit_train_val", False))
    if reference.get("training_splits", ["train"]) != ["train"]:
        raise ValueError("Reference config must certify TRAIN-only optimization")
    processed = Path(reference["processed_dir"])
    official = (validate_split_view(reference) if reference.get('official_split_view') else
        validate_official_inputs(processed, Path(reference["graph_bundle"]),
        Path(reference["protein_token_manifest"]), Path(reference["validation_contract"]),
        Path(reference["official_uniprot_source"])))
    qids, pids, qfeatures, pfeatures, audit, frequency = load_component_features(reference, args.source_dir.resolve(), device)
    if args.atomic_readout != 'none':
        from ec_retrieval.atomic_store import attach_atomic_features
        qfeatures, audit['atomic_readout'] = attach_atomic_features(qfeatures, args.atomic_cache, args.source_dir)
    output.mkdir(parents=True, exist_ok=True)
    residue_metadata = json.loads(args.residue_cache.read_text())
    if residue_metadata["source_mean"] != reference["protein_global_embeddings"]:
        raise ValueError("Residue cache does not match the frozen mean encoder")
    if args.protein_readout == "residue_attention" and residue_metadata["status"] != "completed":
        raise ValueError("Residue attention requires a completed and audited cache")
    atomic_json(output / "residue_cache_audit.json", residue_metadata)
    if args.protein_readout == "residue_attention":
        pfeatures = PrefetchResidueFeatureStore(pfeatures, pids, args.residue_cache)
    atomic_json(output / "official_input_audit.json", official)
    atomic_json(output / "feature_audit.json", audit)
    atomic_json(output / "train_frequency.json", frequency)
    qrow, prow = {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)}
    train = pd.read_parquet(processed / "pairs.parquet", columns=["reaction_id", "enzyme_id", "split"],
                           filters=[("split", "==", "train")])
    if set(train.split.astype(str)) != {"train"} or train.duplicated(["reaction_id", "enzyme_id"]).any():
        raise ValueError("Invalid TRAIN associations")
    train_queries = set(train.reaction_id.astype(str))
    labels, ec_audit = load_official_ec_annotations(processed, Path(reference["official_uniprot_source"]), ["train"])
    depth = infer_ec_depth(labels["train"])
    vocab, token_to_index, parents = build_ec_vocab(labels["train"], depth)
    vocab_sizes = [len(v) for v in vocab]
    qtargets = target_tokens_for_ids(qids, labels["train"], token_to_index)
    ptargets = protein_target_tokens(pids, train, labels["train"], token_to_index)
    if args.protein_ec_supervision=='fill_missing_official_coarse':
        train_enzyme_ids=set(train.enzyme_id.astype(str))
        protein_annotations=load_train_enzyme_annotations(processed,Path(reference['official_uniprot_source']),train_enzyme_ids)
        ptargets,protein_target_audit,protein_additions=fill_missing_protein_coarse_targets(
            pids,ptargets,token_to_index,train_enzyme_ids,protein_annotations)
        protein_target_audit['source']=str(reference['official_uniprot_source'])
    else:
        protein_target_audit=dict(policy='reaction_consensus',training_splits=['train'],
            additional_protein_annotation_source_read=False,added_by_level=[0,0,0,0],
            original_targets_preserved=True,l4_unchanged=True,vocabulary_unchanged=True,
            reaction_targets_unchanged=True,nontrain_targets_unchanged=True,ground_truth_ec_is_model_input=False)
        protein_additions=[]
    atomic_json(output/'protein_ec_target_audit.json',protein_target_audit)
    pd.DataFrame(protein_additions,columns=['enzyme_id','level','prefix','raw_ec']).to_csv(
        output/'protein_ec_supplement.csv',index=False)
    # The protein targets are constructed from the unchanged strict labels.
    # Optional enzyme-only supplementation is explicit above. Reaction targets
    # remain independently governed by the unchanged strict reaction policy.
    if args.reaction_ec_supervision == "coarse_consensus_l1_l3":
        consensus = load_train_consensus(processed, Path(reference["official_uniprot_source"]), train)
        qtargets, target_audit, supplement = supplement_reaction_targets(
            qids, qtargets, token_to_index, consensus, set(labels["train"]))
    else:
        counts = [sum(bool(v) for v in level) for level in qtargets]
        target_audit = dict(policy="strict", training_splits=["train"], supervision_before=counts,
            supervision_after=counts, added_by_level=[0]*depth, added_query_count=0,
            original_targets_preserved=True, l4_unchanged=True, vocabulary_unchanged=True,
            nontrain_targets_unchanged=True, protein_targets_unchanged=True,
            missing_children_imputed=False, later_rhea_snapshot_read=False, test_labels_read=False)
        supplement = []
    atomic_json(output / "reaction_ec_target_audit.json", target_audit)
    pd.DataFrame(supplement, columns=["reaction_id", "level", "prefix"]).to_csv(
        output / "reaction_ec_supplement.csv", index=False)
    atomic_torch_save(output / "training_ec_targets.pt", dict(query_ids=qids, protein_ids=pids,
        reaction_targets=qtargets, protein_targets=ptargets, training_splits=["train"]))

    qtarget_store = ECTargetStore(qtargets, vocab_sizes, device) if args.cache_ec_targets else None
    ptarget_store = ECTargetStore(ptargets, vocab_sizes, device) if args.cache_ec_targets else None
    adjacency = build_train_adjacency(train, qrow, prow)
    universe = np.asarray(sorted(set(train.enzyme_id.astype(str).map(prow))), dtype=np.int64)
    domain = TrainCandidateDomain(adjacency, universe)
    ambiguity = None
    if args.training_comparison_policy=='input_ambiguity_neutral':
        train_qids=set(train.reaction_id.astype(str));train_pids=set(train.enzyme_id.astype(str))
        reaction_metadata=pd.read_parquet(processed/'reactions.parquet',columns=['reaction_id','official_molecule_set_smiles'])
        reaction_metadata=reaction_metadata.loc[reaction_metadata.reaction_id.astype(str).isin(train_qids)]
        enzyme_metadata=pd.read_parquet(processed/'enzymes.parquet',columns=['enzyme_id','sequence'])
        enzyme_metadata=enzyme_metadata.loc[enzyme_metadata.enzyme_id.astype(str).isin(train_pids)]
        components={str(r.reaction_id):split_components(str(r.official_molecule_set_smiles))[0]
                    for r in reaction_metadata.itertuples(index=False)}
        sequences=dict(zip(enzyme_metadata.enzyme_id.astype(str),enzyme_metadata.sequence.astype(str)))
        ambiguity=TrainInputAmbiguity(qids,pids,zip(train.reaction_id.astype(str),train.enzyme_id.astype(str)),
            components,sequences,qfeatures.global_features.detach().cpu().numpy())
        atomic_json(output/'train_ambiguity_audit.json',ambiguity.audit)
        atomic_torch_save(output/'train_neutral_comparisons.pt',ambiguity.export())
    if requested_reverse_pool_size == -1:
        # Resolve only after filtering TRAIN associations. Neither validation
        # nor outer-heldout reaction identities enter the retrieval denominator.
        args.reverse_reaction_pool_size = len(domain.query_rows)
    reverse_pool = (ReverseReactionPool(domain,args.reverse_reaction_pool_size,args.seed+4_000_037)
                    if args.reverse_reaction_pool_size else None)
    reverse_trace = []
    pfeatures=ShardedResidueFeatureStore(pfeatures.mean,pids,args.residue_cache,
        train_rows=universe,priority=training_cache_priority(adjacency,len(pids),args.positives_per_query),
        cache_budgets=list(zip([int(v) for v in args.resident_cache_devices.split(',')],
                              [int(float(v)*2**30) for v in args.resident_cache_gib.split(',')])),
        main_headroom_bytes=int(args.cache_main_headroom_gib*2**30),
        helper_headroom_bytes=int(args.cache_helper_headroom_gib*2**30))
    atomic_json(output / "resident_cache_audit.json",pfeatures.snapshot())
    for cache_device in pfeatures.shards:torch.cuda.reset_peak_memory_stats(cache_device)
    print(json.dumps(dict(event="resident_cache_ready",plan=pfeatures.cache_plan)),flush=True)
    atomic_torch_save(output / "candidate_domain.pt", dict(query_rows=torch.from_numpy(domain.query_rows),
        candidate_rows=torch.from_numpy(universe), training_splits=["train"]))
    atomic_json(output / "train_ec_audit.json", ec_audit)
    atomic_json(output / "ec_vocabulary.json", {"source": "official_train_only", "levels": vocab, "depth": depth})
    # Materialize all optimization targets before any validation diagnostics.
    val_labels, val_audit = load_official_ec_annotations(processed, Path(reference["official_uniprot_source"]), ["val"])
    vqids, vpids, relevance = load_contract(Path(reference["validation_contract"]), "val")
    vqrows, vprows = [qrow[q] for q in vqids], [prow[p] for p in vpids]
    atomic_json(output / "validation_ec_diagnostic_audit.json", val_audit)
    kwargs = dict(molecule_input_dim=qfeatures.molecule_input_dim, global_input_dim=qfeatures.global_input_dim,
        protein_input_dim=pfeatures.shape[1], ec_vocab_sizes=vocab_sizes, child_to_parent=parents,
        projection_hidden_dim=args.projection_hidden_dim, embedding_dim=args.embedding_dim,
        dropout=args.dropout, temperature=args.temperature, pooling=args.pooling,
        ec_conditioning=args.ec_conditioning, ec_router_normalization="parent_prior",
        drop_morgan_probability=args.drop_morgan_probability, drop_unimol_probability=args.drop_unimol_probability,
        fusion_sharing=args.fusion_sharing, scoring=args.scoring,
        ec_score_depth=args.ec_score_depth, initial_ec_weight=args.initial_ec_weight, ec_posterior=args.ec_posterior,
        protein_readout=args.protein_readout,residue_mix=args.residue_mix,residue_block_size=args.residue_block_size)
    if args.fusion_sharing != "independent_matched" or args.drop_unimol_probability != 0:
        raise ValueError("This schedule study fixes independent fusion and preserves the Uni-Mol view")
    schedule = [morgan_probability_for_epoch(args.morgan_dropout_schedule,
        args.drop_morgan_probability, min(epoch,args.schedule_epochs-1), args.schedule_epochs) for epoch in range(args.epochs)]
    atomic_json(output / "masking_schedule.json", dict(policy=args.morgan_dropout_schedule,
        probabilities=schedule, mean_probability=sum(schedule)/len(schedule),
        source="declared_epoch_index_only", validation_used=False))
    if args.candidate_background == "learned":
        kwargs.update(candidate_background=args.candidate_background, background_max_norm=args.background_max_norm)
    if args.reaction_context != 'none':
        kwargs.update(reaction_context=args.reaction_context, context_hidden_dim=args.context_hidden_dim,
                      context_heads=args.context_heads)
        if args.context_gradient_mode == 'isolated':
            kwargs['context_gradient_mode'] = args.context_gradient_mode
    if args.atomic_readout != 'none':
        kwargs.update(atomic_readout=args.atomic_readout, atom_attention_dim=args.atom_attention_dim,
                      atomic_mix=args.atomic_mix)
        if args.atomic_logit_bound:
            kwargs['atomic_logit_bound'] = args.atomic_logit_bound
    if args.matched_ablation != 'none':
        kwargs['matched_ablation'] = args.matched_ablation
    model = make_residue_retriever(kwargs).to(device)
    atomic_torch_save(output / "initial_state.pt", model.state_dict())
    planned_steps = math.ceil(len(adjacency) / args.query_batch_size)
    if args.max_steps_per_epoch:
        planned_steps = min(planned_steps, args.max_steps_per_epoch)
    ema = (ParameterEMA(model, half_life_epochs=args.ema_half_life_epochs,
                        steps_per_epoch=planned_steps, warmup_epochs=args.ema_warmup_epochs)
           if args.ema_half_life_epochs else None)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(query_tail_weighting_active=args.tail_risk_weight > 0,
                  query_tail_diagnostic_inactive_value=0.0 if args.tail_risk_weight == 0 else None)
    config.update(proposal_ec_supervision_active=args.proposal_ec_weight > 0,
                  proposal_ec_levels=[1, 2, 3], proposal_ec_added_parameters=0,
                  proposal_ec_additional_encoder_calls=0, proposal_ec_inference_change=False,
                  proposal_ec_target_scope='unchanged current FIT-TRAIN reaction/enzyme masks only',
                  proposal_ec_includes_extra_reverse_candidates=False)
    config.update(requested_reverse_reaction_pool_size=requested_reverse_pool_size,
                  full_train_reverse_pool=requested_reverse_pool_size == -1)
    config.update(reverse_loss_group_source='complete FIT-TRAIN positive mask only',
                  reverse_loss_tail_unit='reaction_group' if args.reverse_loss_reduction=='reaction_macro' else 'enzyme_query',
                  reverse_loss_inference_change=False)
    config.update(architecture="residue_readout_retriever", training_splits=["train"],
        pretrained_encoders_frozen=["Uni-Mol", "ESM-2"], initialization="random_trainable_modules_no_teacher_or_anchor",
        fusion_initialization="all shared weights identical; zero residue scorer is exactly neutral",
        ground_truth_ec_is_model_input=False, checkpoint_selection="minimum_full_validation_mean_rank",
        reference_weights_loaded=False, parameter_count=sum(p.numel() for p in model.parameters()),
        ec_depth=depth, ec_vocab_sizes=vocab_sizes, processed_dir=str(processed),
        validation_contract=reference["validation_contract"], test_relevance_opened=False,
        historical_time_test_previously_observed=True, is_smoke=args.max_steps_per_epoch > 0,
        candidate_index_dimension=model.index_dimension, ensemble_members=1,
        external_support_bank=False, posthoc_density_correction=False,
        optimization_scope="all retrieval parameters jointly; pretrained feature extractors frozen",
        ec_target_cache_bytes=(qtarget_store.nbytes + ptarget_store.nbytes) if args.cache_ec_targets else 0,
        parallelism="unchanged CSR; optional bounded TRAIN prefetch with two pinned buffers and a copy stream; synchronous VAL",
        data_prefetch_changes_model_math=False)
    config.update(positive_coverage_formula="log1p(mean_positive sum_unobserved exp(score_u-score_p))" if args.positive_coverage == "factorized" else "legacy_coupled_CE",
        positive_coverage_scope="replace all-positive and query-tail risk; preserve existing weights and any-positive branch",
        candidate_source="enzymes_with_TRAIN_associations_only",
        mining_active=args.candidate_sampling == "dynamic", inference_requires_mining_bank=False,
        known_positive_policy="all_TRAIN_associations_in_pool_are_positive")
    config.update(retrieval_directions=['reaction_to_enzyme','enzyme_to_reaction']
                  if args.reverse_retrieval_weight else ['reaction_to_enzyme'],
        retrieval_direction_weights=[1-args.reverse_retrieval_weight, args.reverse_retrieval_weight],
        reverse_anchor_policy='only enzymes with at least one known TRAIN positive in current reaction batch',
        bidirectional_same_score_matrix=True, reverse_loss_changes_sampler=False,
        reverse_loss_adds_parameters=False)
    if requested_reverse_pool_size == -1:
        config['reverse_anchor_policy'] = 'every sampled TRAIN enzyme has at least one positive in the complete FIT-TRAIN reaction pool'
    config['checkpoint_selection_policy'] = args.checkpoint_selection
    if args.checkpoint_selection == 'hit1_mrr':
        config['checkpoint_selection'] = 'maximum_full_validation_hit1_then_mrr_then_minimum_mr'
    if args.checkpoint_selection == 'bidirectional_hit1_mrr':
        config['checkpoint_selection'] = 'maximum_full_validation_bidirectional_hit1_geomean_then_mrr_geomean'
        config['validation_directions'] = ['reaction_to_enzyme','enzyme_to_reaction']
        config['validation_direction_log_weights'] = [.5,.5]
        config['mean_rank_used_for_selection'] = False
    if args.candidate_background == 'learned':
        config['architecture'] = 'residue_readout_with_learned_candidate_background'
        config['candidate_bias_parameters'] = args.embedding_dim
    if args.atomic_readout != 'none':
        config.update(architecture='residue_and_reaction_context_atomic_readout',
            atomic_added_parameters=sum(p.numel() for p in model.atomic_readout.parameters()),
            atomic_initialization='zero query; random key; isolated initialization RNG; uniform delta zero',
            atomic_context_source='normalized original moments and Morgan AFTER original view dropout',
            atomic_pretrained_encoder_calls=0, atomic_index_width_unchanged=True,
            atomic_uniform_anchor='original FP32 molecule mean plus attention-minus-uniform on FP16 atoms',
            atomic_logit_transform=('centered_smooth_bound' if args.atomic_logit_bound else 'raw'),
            atomic_parallelism='vectorized per-component CSR; no per-atom or per-molecule forward loops')
    if args.reaction_context == 'molecule_set':
        config.update(architecture='residue_readout_with_molecule_set_context',
            reaction_context_added_parameters=sum(p.numel() for p in model.molecule_context.parameters()),
            reaction_context_initialization='zero output affine; common initialization and training RNG preserved',
            reaction_context_scope='within each official unordered component multiset; before EC prediction',
            reaction_context_extra_pretrained_encoder_calls=0,
            reaction_context_score_formula_unchanged=True,reaction_context_index_width_unchanged=True,
            reaction_context_parallelism='batch/head/component tensor operations; no per-reaction forward loop')
    if args.context_gradient_mode == 'isolated':
        config.update(architecture='residue_readout_with_gradient_isolated_molecule_context',
            optimization_scope='simultaneous reference objective and contextual objective on disjoint parameter groups',
            context_objective_weight=1., context_objective='same bidirectional retrieval and auxiliary EC objective',
            context_gradient_boundary='detach reference reaction state, protein vectors/logits, shared EC parameters, alpha and scale',
            context_clip_scope='separate reference/context parameter groups; same norm limit',
            context_requires_teacher_checkpoint=False,
            context_dropout='original EC-head dropout with an isolated step-dependent RNG stream',
            context_validation='complete contextual prediction; unchanged inner selection policy',
            context_inference_extra_pass=False)
    config.update(training_extension="reverse_full_train" if requested_reverse_pool_size == -1 else "reverse512" if reverse_pool else "fixed_prefix_budget",
        effective_ec_conditioning=bool(model.ec_conditioning),effective_context_fusion=False,
        reverse_candidate_source="TRAIN_reactions_only",reverse_pool_changes_auxiliary_ec_exposure=False,
        reverse_pool_rng_stream="independent_numpy_and_isolated_torch",test_used_for_selection=False,
        cosine_horizon_epochs=args.schedule_epochs,lr_after_horizon=.05*args.learning_rate,
        masking_horizon_epochs=args.schedule_epochs,
        bidirectional_same_score_matrix=not bool(reverse_pool),
        reverse_anchor_policy="enzymes with at least one known TRAIN positive in the expanded pool" if reverse_pool else config['reverse_anchor_policy'])
    config.update(latent_aux_active=args.latent_aux_weight > 0,
        latent_aux_temperature_policy="fixed_no_gradient",
        latent_aux_uses_ec_probabilities=False, latent_aux_uses_alpha=False,
        latent_aux_added_parameters=0, latent_aux_additional_encoder_calls=0,
        latent_aux_candidates="same_as_primary_complete_TRAIN_positive_masks",
        latent_aux_changes_inference=False)
    config.update(parameter_ema_active=ema is not None,
        evaluated_parameter_source='ema' if ema is not None else 'online',
        ema_decay_per_update=ema.decay if ema is not None else None,
        ema_warmup_steps=ema.warmup_steps if ema is not None else None,
        ema_changes_online_optimizer=False, ema_changes_inference_architecture=False)
    atomic_json(output / "config.json", config)
    atomic_json(output / "data_audit.json", dict(training_splits=["train"], train_edges=len(train),
        train_queries=len(adjacency), train_enzymes=len(universe), validation_queries=len(vqids),
        validation_candidates=len(vpids), test_relevance_opened=False,
        validation_supervises_training=False, unobserved_pairs_are_confirmed_negatives=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    isolated_context = args.context_gradient_mode == 'isolated'
    if isolated_context:
        common_parameters, context_parameters = model.isolated_parameter_groups()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs, eta_min=.05*args.learning_rate)
    enabled, dtype = amp_policy(device, args.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp_dtype == "float16")
    rng, negative_rng = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 1_000_003)
    exposure = {q: np.zeros(len(v), dtype=np.int64) for q, v in adjacency.items()}
    hard_rng = np.random.default_rng(args.seed + 2_000_033)
    bank, mining_events, query_orders = None, [], []
    planned_mining_epochs = (list(range(args.mining_warmup_epochs, args.epochs, args.mining_refresh_epochs))
        if args.candidate_sampling == "dynamic" else [])
    def save_mining_audit():
        atomic_json(output / "candidate_mining_audit.json", dict(training_splits=["train"],
            mode=args.candidate_sampling, planned_epochs_zero_based=planned_mining_epochs,
            events=mining_events, validation_or_test_associations_used=False,
            ground_truth_ec_input=False, inference_dependency=False))
    save_mining_audit()
    best_mr = math.inf
    best_selection_key = None
    best_epoch, best_metrics = None, None
    online_best_key, online_best_epoch = None, None
    for epoch in range(args.epochs):
        began = time.perf_counter()
        if epoch in planned_mining_epochs:
            bank, artifact = mine_current_train_candidates(model, qfeatures, pfeatures, domain,
                bank_size=args.mining_bank_size, query_batch_size=args.mining_query_batch_size,
                protein_batch_size=args.mining_protein_batch_size, amp_enabled=enabled, amp_dtype=dtype)
            artifact.update(before_epoch_zero_based=epoch, completed_training_epochs=epoch)
            path = output / "mining" / f"before_epoch{epoch:03d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(path, artifact)
            mining_events.append(dict(before_epoch_zero_based=epoch, path=str(path),
                seconds=artifact["seconds"], query_count=artifact["query_count"],
                candidate_count=artifact["candidate_count"], known_positive_overlap=0))
            save_mining_audit()
            print(json.dumps(dict(event="candidate_refresh", epoch=epoch,
                seconds=artifact["seconds"], training_splits=["train"])), flush=True)
            del artifact
        model.train()
        model.view_dropout.morgan_probability = schedule[epoch]
        totals, steps = defaultdict(float), 0
        reverse_epoch_rows=[]
        epoch_lr=optimizer.param_groups[0]["lr"]
        order = rng.permutation(np.asarray(sorted(adjacency), dtype=np.int64))
        query_orders.append(torch.from_numpy(order.copy()))
        sampling_totals = defaultdict(float)
        # Sampling uses separate NumPy RNGs and never depends on optimizer state.
        # Materializing only this epoch preserves the original draws/exposures.
        epoch_batches = []
        for start in range(0, len(order), args.query_batch_size):
            if args.max_steps_per_epoch and len(epoch_batches) >= args.max_steps_per_epoch:
                break
            queries = list(map(int, order[start:start+args.query_batch_size]))
            candidates, mask, sampling = sample_training_candidates(queries, domain, args.positives_per_query, rng,
                target_candidate_count=args.candidate_pool_size, negative_rng=negative_rng,
                positive_exposure_counts=exposure, bank=bank, hard_rng=hard_rng,
                hard_per_query=args.mined_per_query, random_reserve_fraction=args.random_reserve_fraction)
            epoch_batches.append((queries, candidates, mask, sampling))
        context = (pfeatures.prefetch([batch[1] for batch in epoch_batches], depth=args.residue_prefetch_depth)
                   if args.residue_prefetch_depth else nullcontext(None))
        with context as prefetched:
            for queries, candidates, mask, sampling in epoch_batches:
                protein_features = next(prefetched) if prefetched is not None else pfeatures[candidates]
                neutral_mask = None
                for key, value in sampling.items():
                    sampling_totals[key] += value
                if reverse_pool:
                    extra_queries,extended_mask,all_reverse_rows=reverse_pool.sample(queries,candidates,mask)
                    if requested_reverse_pool_size == -1 and not bool(extended_mask.any(0).all()):
                        raise AssertionError("Every TRAIN enzyme must have a positive in the complete FIT-TRAIN reaction pool")
                    reverse_epoch_rows.append(torch.from_numpy(all_reverse_rows.copy()))
                    if ambiguity is not None:
                        # CPU sparse extraction/checks precede the existing transfer.
                        neutral_mask=ambiguity.batch_mask(all_reverse_rows,candidates,extended_mask).to(device)
                    extended_mask=extended_mask.to(device)
                mask = mask.to(device)
                qt = qtarget_store[queries] if qtarget_store is not None else dense_target_masks(queries, qtargets, vocab_sizes, device)
                pt = ptarget_store[candidates] if ptarget_store is not None else dense_target_masks(candidates, ptargets, vocab_sizes, device)
                optimizer.zero_grad(set_to_none=True)
                isolated_diagnostics = {}
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                    reaction_features = qfeatures[queries]
                    scores, reaction, protein = model(reaction_features, protein_features)
                    if reverse_pool:
                        with isolated_extra_rng(device,args.seed+5_000_039+epoch*len(epoch_batches)+steps):
                            extra_features = qfeatures[extra_queries]
                            extra=model.encode_reaction(extra_features)
                        extra_scores=model.score_embeddings(extra["embedding"],protein["embedding"])
                        retrieval_scores=exclude_neutral_comparisons(scores,neutral_mask[:len(scores)] if neutral_mask is not None else None)
                        extra_retrieval_scores=exclude_neutral_comparisons(extra_scores,neutral_mask[len(scores):] if neutral_mask is not None else None)
                        retrieval,components=expanded_bidirectional_loss(retrieval_scores,mask,extra_retrieval_scores,extended_mask,
                            reverse_weight=args.reverse_retrieval_weight,mode=args.positive_coverage,
                            all_positive_weight=args.all_positive_weight,tail_risk_weight=args.tail_risk_weight,
                            tail_fraction=args.tail_risk_fraction,reverse_reduction=args.reverse_loss_reduction)
                        if neutral_mask is not None:
                            active_reverse=neutral_mask&extended_mask.any(0)[None,:]
                            components.update(neutral_forward_pairs=neutral_mask[:len(scores)].sum().float(),
                                neutral_reverse_pairs=active_reverse.sum().float(),
                                neutral_forward_anchors=neutral_mask[:len(scores)].any(1).sum().float(),
                                neutral_reverse_anchors=active_reverse.any(0).sum().float())
                    else:
                        retrieval, components = bidirectional_retrieval_loss(scores, mask, mode=args.positive_coverage,
                            reverse_weight=args.reverse_retrieval_weight,
                            all_positive_weight=args.all_positive_weight, tail_risk_weight=args.tail_risk_weight,
                            tail_fraction=args.tail_risk_fraction)
                    ec, consistency, alignment = ec_objectives(
                        model, reaction, protein, qt, pt, mask,
                        alignment_weight=args.cross_modal_ec_weight)
                    warmup = min(1., (epoch+1)/args.aux_warmup_epochs) if args.aux_warmup_epochs else 1.
                    loss = retrieval + warmup * (args.ec_loss_weight*ec + args.consistency_loss_weight*consistency
                                                 + args.cross_modal_ec_weight*alignment)
                    loss, proposal_diagnostics = add_proposal_ec_supervision(
                        loss, reaction, protein, qt, pt,
                        weight=args.proposal_ec_weight, warmup=warmup)
                    loss, latent_diagnostics = add_latent_retrieval_supervision(
                        loss, reaction, protein, mask, weight=args.latent_aux_weight,
                        temperature=args.latent_aux_temperature,
                        extra_reaction=extra if reverse_pool else None,
                        extended_mask=extended_mask if reverse_pool else None,
                        reverse_weight=args.reverse_retrieval_weight, mode=args.positive_coverage,
                        all_positive_weight=args.all_positive_weight,
                        tail_risk_weight=args.tail_risk_weight, tail_fraction=args.tail_risk_fraction)
                    if isolated_context:
                        contextual_loss, isolated_diagnostics = isolated_context_objective(
                            model, reaction, protein, extra, reaction_features, extra_features,
                            mask, extended_mask, qt, pt, warmup=warmup,
                            reverse_weight=args.reverse_retrieval_weight,
                            all_positive_weight=args.all_positive_weight,
                            tail_risk_weight=args.tail_risk_weight, tail_fraction=args.tail_risk_fraction,
                            ec_weight=args.ec_loss_weight, consistency_weight=args.consistency_loss_weight,
                            alignment_weight=args.cross_modal_ec_weight,
                            dropout_seed=args.seed+7_000_043+epoch*len(epoch_batches)+steps)
                        isolated_diagnostics['base_loss'] = loss
                        loss = loss + contextual_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {steps}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if isolated_context:
                    norm = torch.nn.utils.clip_grad_norm_(common_parameters, args.gradient_clip_norm, error_if_nonfinite=True)
                    isolated_diagnostics['context_gradient_norm'] = torch.nn.utils.clip_grad_norm_(
                        context_parameters, args.gradient_clip_norm, error_if_nonfinite=True)
                else:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                for key, value in dict(loss=loss, retrieval=retrieval, ec=ec, consistency=consistency,
                                       alignment=alignment, gradient_norm=norm, **components,
                                       **latent_diagnostics, **isolated_diagnostics, **proposal_diagnostics).items():
                    # Keep detached FP64 scalar sums on device; one transfer per epoch.
                    totals[key] = totals[key] + value.detach().double()
                steps += 1
        step_prefix_scheduler(scheduler,epoch,args.schedule_epochs)
        if reverse_pool:
            reverse_trace.append(torch.stack(reverse_epoch_rows))
        metric_values = torch.stack(list(totals.values())).div(steps).cpu().tolist()
        train_metrics = dict(zip(totals, metric_values))
        train_metrics.update(epoch=epoch, steps=steps, seconds=time.perf_counter()-began,
            morgan_drop_probability=schedule[epoch],learning_rate=epoch_lr)
        if args.scoring == "explicit":
            train_metrics["learned_ec_weight"] = float(model.ec_weight().detach())
        if args.candidate_background == 'learned':
            train_metrics['background_norm'] = float(model.background_vector().detach().norm())
        if args.protein_readout == "residue_attention":
            train_metrics["residue_weight_norm"] = float(model.residue_weight.detach().norm())
            train_metrics["last_batch_residue_effective_fraction"] = float(protein["residue_effective_fraction"].detach().mean())
            train_metrics["last_batch_residue_refinement_norm"] = float(protein["residue_refinement_norm"].detach().mean())
        if args.atomic_readout != 'none':
            train_metrics['atomic_query_norm'] = float(model.atomic_readout.query.weight.detach().norm())
            train_metrics['atomic_key_norm'] = float(model.atomic_readout.key.weight.detach().norm())
            train_metrics['last_batch_atomic_effective_fraction'] = float(reaction['atomic_effective_fraction'].detach().mean())
            train_metrics['last_batch_atomic_refinement_norm'] = float(reaction['atomic_refinement_norm'].detach().mean())
        train_metrics.update({"sampling_" + k: v / steps for k, v in sampling_totals.items()})
        append_jsonl(output / "train_metrics.jsonl", train_metrics)
        atomic_json(output / "resident_cache_audit.json",pfeatures.snapshot())
        atomic_json(output / "resident_device_memory.json",{str(d):dict(
            peak_allocated_mib=torch.cuda.max_memory_allocated(d)/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(d)/2**20) for d in pfeatures.shards})
        metrics, _, val_qv, val_pv = evaluate(model, qfeatures, pfeatures, vqrows, vprows, vqids, vpids, relevance,
            train_queries, val_labels.get("val", {}), token_to_index, device, args)
        if args.checkpoint_selection == 'bidirectional_hit1_mrr':
            metrics = add_reverse_validation(metrics,model,val_qv,val_pv,vqids,vpids,relevance,device,args.eval_query_batch_size)
        metrics["epoch"] = epoch
        selected_model = model
        if ema is not None:
            # Keep the unmodified online trajectory as a contemporaneous control.
            # Validation does not supervise EMA or select its decay/warmup.
            append_jsonl(output / "online_validation_metrics.jsonl", metrics)
            online_key = validation_selection_key(metrics, args.checkpoint_selection)
            if online_best_key is None or online_key < online_best_key:
                online_best_key, online_best_epoch = online_key, epoch
                atomic_torch_save(output / "online_best.pt", dict(model_state=model.state_dict(),
                    model_kwargs=kwargs, epoch=epoch, validation_metrics=metrics, config=config,
                    parameter_source='online_control'))
            selected_model = ema.model
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                metrics, _, val_qv, val_pv = evaluate(selected_model, qfeatures, pfeatures,
                    vqrows, vprows, vqids, vpids, relevance, train_queries,
                    val_labels.get("val", {}), token_to_index, device, args)
                if args.checkpoint_selection == 'bidirectional_hit1_mrr':
                    metrics = add_reverse_validation(metrics,selected_model,val_qv,val_pv,
                        vqids,vpids,relevance,device,args.eval_query_batch_size)
            metrics['epoch'] = epoch
            metrics['ema_updates'] = ema.updates
        append_jsonl(output / "validation_metrics.jsonl", metrics)
        selection_key = validation_selection_key(metrics, args.checkpoint_selection)
        improved = best_selection_key is None or selection_key < best_selection_key
        if improved:
            best_selection_key = selection_key
            best_mr = metrics["official_reactzyme_mean_rank"]
            best_epoch, best_metrics = epoch, dict(metrics)
            atomic_torch_save(output / "best.pt", dict(model_state=selected_model.state_dict(), model_kwargs=kwargs,
                epoch=epoch, validation_metrics=metrics, config=config,
                training_morgan_drop_probability=schedule[epoch]))
        if epoch + 1 in args.checkpoint_milestones:
            retain_milestone(output, epoch + 1, best_epoch, best_metrics, args.checkpoint_selection)
        if epoch+1 == args.schedule_epochs and args.prefix_control_run:
            verify_prefix(output,args.prefix_control_run.resolve(),args.schedule_epochs,exposure,query_orders,domain)
        if epoch+1 in set(args.checkpoint_milestones)|{args.epochs,args.schedule_epochs}:
            save_training_state(output/f"training_state_epoch{epoch+1:03d}.pt",model=model,optimizer=optimizer,
                scheduler=scheduler,scaler=scaler,epoch=epoch,best_epoch=best_epoch,best_metrics=best_metrics,
                rng=rng,negative_rng=negative_rng,hard_rng=hard_rng,reverse_pool=reverse_pool,
                exposure=exposure,query_orders=query_orders,config=config)
            if ema is not None:
                atomic_torch_save(output/f"ema_state_epoch{epoch+1:03d}.pt", ema.state_dict())
        print(json.dumps(dict(epoch=epoch, train_loss=train_metrics["loss"],
            val_mean_rank=metrics["official_reactzyme_mean_rank"], val_hit1=metrics["hit@1"],
            val_all_mrr=metrics["official_reactzyme_mrr"], improved=improved, is_smoke=config["is_smoke"])), flush=True)
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    final, ranks, qv, pv = evaluate(model, qfeatures, pfeatures, vqrows, vprows, vqids, vpids, relevance,
        train_queries, val_labels.get("val", {}), token_to_index, device, args)
    reload_keys = ["official_reactzyme_mean_rank", "hit@1", "official_reactzyme_mrr"]
    if args.checkpoint_selection == 'bidirectional_hit1_mrr':
        final = add_reverse_validation(final,model,qv,pv,vqids,vpids,relevance,device,args.eval_query_batch_size)
        reload_keys += ['e2r_hit@1','e2r_official_reactzyme_mrr','e2r_official_reactzyme_mean_rank',
                        'joint_hit1_geomean','joint_mrr_geomean']
    for key in reload_keys:
        if abs(final[key] - best["validation_metrics"][key]) > 1e-6:
            raise ValueError("Reloaded checkpoint failed validation reproduction")
    for row in ranks:
        append_jsonl(output / "validation_topk.jsonl", row)
    atomic_torch_save(output / "validation_embeddings.pt", dict(query_ids=vqids, candidate_ids=vpids,
        query_embeddings=qv, candidate_embeddings=pv, split="val", epoch=best["epoch"]))
    atomic_torch_save(output / "sampling_trace.pt", dict(query_orders=torch.stack(query_orders),
        query_rows=torch.from_numpy(domain.query_rows),
        direct_positive_exposures=[torch.from_numpy(exposure[int(q)].copy()) for q in domain.query_rows],
        training_splits=["train"]))
    if reverse_pool:
        atomic_torch_save(output/"reverse_sampling_trace.pt",dict(reaction_pools=reverse_trace,
            training_splits=['train'],pool_size=args.reverse_reaction_pool_size,
            complete_train_positive_masks=True,validation_or_test_associations_used=False))
    coverage = np.concatenate(list(exposure.values()))
    summary = dict(status="smoke_completed" if config["is_smoke"] else "completed", epochs_completed=args.epochs,
        best_epoch=best["epoch"], best_validation_metrics=final, training_splits=["train"],
        is_smoke=config["is_smoke"], checkpoint_reload_verified=True, test_relevance_opened=False,
        direct_positive_sampling_coverage=float((coverage > 0).mean()), parameter_count=config["parameter_count"],
        index_dimension=model.index_dimension, trainable_backbones=False, ensemble_members=1,
        candidate_sampling=args.candidate_sampling, mining_refreshes_completed=len(mining_events),
        positive_coverage=args.positive_coverage, scoring=args.scoring, ec_posterior=args.ec_posterior, protein_readout=args.protein_readout)
    summary.update(latent_aux_weight=args.latent_aux_weight,
        latent_aux_temperature=args.latent_aux_temperature, reverse_loss_reduction=args.reverse_loss_reduction,
        inference_architecture_unchanged=args.reaction_context == 'none' and args.atomic_readout == 'none')
    if isolated_context:
        summary.update(context_gradient_mode='isolated',
            context_requires_teacher_checkpoint=False,
            reference_and_context_trained_simultaneously=True,
            validation_uses_complete_contextual_model=True)
    summary.update(parameter_ema_active=ema is not None, online_best_epoch=online_best_epoch,
        ema_updates=ema.updates if ema is not None else 0,
        evaluated_parameter_source='ema' if ema is not None else 'online')
    if device.type == "cuda":
        summary.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
