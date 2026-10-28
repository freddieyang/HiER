"""Check paper CLI settings without launching a training job."""
import argparse
import sys
import unittest
from unittest.mock import patch
from scripts.train import build_command, ROOT
from scripts.train_hier_controlled_extensions import parse_args


def arguments(**changes):
    fields = dict(split='time', variant='full', seed=42, data_root=ROOT / 'data',
                  reference_run=None, source_dir=None, residue_cache=None,
                  output=ROOT / 'runs' / 'example', device='cuda:0',
                  num_threads=4, prefetch_depth=2, cache_devices='0', cache_gib='6',
                  beta=None, lambda_ec=None, lambda_align=None, dry_run=True)
    fields.update(changes)
    return argparse.Namespace(**fields)


class EntrypointTests(unittest.TestCase):
    def test_paper_flags_are_accepted_by_trainer(self):
        for variant in ('full', 'no_ec_branch', 'no_latent_similarity', 'no_ec_agreement'):
            with self.subTest(variant=variant):
                command = build_command(arguments(variant=variant))
                with patch.object(sys, 'argv', command[2:]):
                    parsed = parse_args()
                self.assertEqual(parsed.epochs, 100)
                self.assertEqual(parsed.ec_posterior, 'leaf_marginal')
                self.assertEqual(parsed.ec_score_depth, 4)
                self.assertEqual(parsed.tail_risk_weight, 0.)
                self.assertEqual(parsed.all_positive_weight, .2)
                self.assertEqual(parsed.ec_loss_weight, .1)
                self.assertEqual(parsed.cross_modal_ec_weight, .02)
                self.assertEqual(parsed.matched_ablation, 'none' if variant == 'full' else variant)

    def test_scan_changes_only_one_loss_weight(self):
        command = build_command(arguments(lambda_ec=.5))
        with patch.object(sys, 'argv', command[2:]):
            parsed = parse_args()
        self.assertEqual(parsed.ec_loss_weight, .5)
        self.assertEqual(parsed.all_positive_weight, .2)
        self.assertEqual(parsed.cross_modal_ec_weight, .02)
        with self.assertRaisesRegex(ValueError, 'one loss weight'):
            build_command(arguments(lambda_ec=.5, beta=.1))


if __name__ == '__main__':
    unittest.main()
