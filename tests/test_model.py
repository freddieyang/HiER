"""Synthetic CPU checks of the released model, not benchmark experiments."""
import math
import unittest

import torch
import torch.nn.functional as F

from ec_retrieval.candidate_background import make_residue_retriever
from ec_retrieval.matched_ablation import ec_objectives
from ec_retrieval.reaction_ec_bridge import hierarchical_multi_positive_loss
from ec_retrieval.residue_readout import residue_attention_delta
from ec_retrieval.functional_set_retriever import molecule_moments


def model_kwargs():
    return dict(molecule_input_dim=6, global_input_dim=8, protein_input_dim=10,
                ec_vocab_sizes=[2, 3, 4, 6],
                child_to_parent=[torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1, 2]),
                                 torch.tensor([0, 0, 1, 2, 3, 3])],
                projection_hidden_dim=16, embedding_dim=12, dropout=0.,
                temperature=.07, pooling='moments', fusion_sharing='independent_matched',
                scoring='explicit', ec_posterior='leaf_marginal', ec_score_depth=4,
                initial_ec_weight=.25, protein_readout='residue_attention')


def inputs():
    generator = torch.Generator().manual_seed(11)
    reaction = dict(molecules=torch.randn(2, 3, 6, generator=generator),
                    mask=torch.tensor([[True, True, False], [True, True, True]]),
                    weights=torch.tensor([[1., 2., 0.], [2., 1., 3.]]),
                    global_features=torch.randn(2, 8, generator=generator))
    values = torch.randn(8, 10, generator=generator)
    protein = dict(mean=torch.stack([values[:3].mean(0), values[3:].mean(0)]),
                   values=values, offsets_cpu=[0, 3, 8])
    return reaction, protein


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, variant=None):
        torch.manual_seed(42)
        kwargs = model_kwargs()
        if variant:
            kwargs['matched_ablation'] = variant
        return make_residue_retriever(kwargs).eval()

    def test_hierarchy_marginals_and_index_score(self):
        model = self.make_model()
        qin, pin = inputs()
        scores, query, protein = model(qin, pin)
        for output in (query, protein):
            for level, mapping in enumerate(model.ec_router.parent_maps()):
                marginal = torch.zeros_like(output['probabilities'][level])
                marginal.scatter_add_(1, mapping[None].expand(2, -1),
                                      output['probabilities'][level + 1])
                torch.testing.assert_close(output['probabilities'][level], marginal, atol=2e-7, rtol=1e-6)
        parts = model.explain_scores(query, protein)
        torch.testing.assert_close(scores, parts['total'], atol=2e-6, rtol=1e-6)
        # Encoding one reaction at a time leaves all candidate scores unchanged.
        single = model.encode_reaction({key: value[:1] for key, value in qin.items()})
        torch.testing.assert_close(model.score_embeddings(single['embedding'], protein['embedding']),
                                   scores[:1], atol=2e-6, rtol=1e-6)
        scores.sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_three_variant_scores_and_branch_removal(self):
        qin, pin = inputs()
        for variant in ('no_ec_branch', 'no_latent_similarity', 'no_ec_agreement'):
            with self.subTest(variant=variant):
                model = self.make_model(variant)
                scores, q, p = model(qin, pin)
                scale = model.logit_scale.clamp(max=math.log(100.)).exp()
                if variant == 'no_latent_similarity':
                    expected = scale / 4 * sum(a @ b.T for a, b in zip(q['ec_sqrt_probabilities'], p['ec_sqrt_probabilities']))
                    self.assertEqual(q['embedding'].shape[-1], 15)
                else:
                    expected = scale * (q['latent_embedding'] @ p['latent_embedding'].T)
                    self.assertEqual(q['embedding'].shape[-1], 12)
                torch.testing.assert_close(scores, expected, atol=2e-6, rtol=1e-6)
                self.assertFalse(hasattr(model, 'ec_weight_logit'))
                if variant == 'no_ec_branch':
                    self.assertEqual(sum(p.numel() for p in model.ec_router.parameters()), 0)
                    losses = ec_objectives(model, q, p, [], [], torch.eye(2, dtype=torch.bool), alignment_weight=.02)
                    self.assertEqual([float(x) for x in losses], [0., 0., 0.])
                else:
                    self.assertEqual(len(q['probabilities']), 4)
                    targets = [F.one_hot(torch.zeros(2, dtype=torch.long), n).bool() for n in [2, 3, 4, 6]]
                    ec, _, alignment = ec_objectives(model, q, p, targets, targets,
                                                     torch.eye(2, dtype=torch.bool), alignment_weight=.02)
                    self.assertGreater(float(ec), 0.)
                    self.assertTrue(torch.isfinite(alignment))

    def test_uniform_residue_readout_preserves_mean(self):
        _, protein = inputs()
        delta, fraction = residue_attention_delta(protein['values'], torch.tensor([0, 3, 8]), torch.zeros(10))
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0., rtol=0.)
        torch.testing.assert_close(fraction, torch.ones_like(fraction))

    def test_original_ec_set_loss_reduces_to_nll(self):
        logits = torch.tensor([[1., 2., -1.], [-2., .5, 1.]], requires_grad=True)
        target = torch.tensor([1, 2])
        loss = hierarchical_multi_positive_loss([logits], [F.one_hot(target, 3).bool()])
        torch.testing.assert_close(loss, F.cross_entropy(logits, target))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighted_moments_ignore_padding(self):
        values = torch.tensor([[[1., 2.], [3., 4.], [float('nan'), float('nan')]]])
        moments, prior = molecule_moments(values, torch.tensor([[True, True, False]]), torch.tensor([[1., 3., 0.]]))
        torch.testing.assert_close(prior, torch.tensor([[.25, .75, 0.]]))
        torch.testing.assert_close(moments[:, :2], torch.tensor([[2.5, 3.5]]))
        torch.testing.assert_close(moments[:, 2:], torch.full((1, 2), math.sqrt(.75 + 1e-8)))


if __name__ == '__main__':
    unittest.main()
