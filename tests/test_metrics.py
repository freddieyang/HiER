"""Check tie handling and all-positive versus first-positive metrics."""
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.evaluate_hier_bidirectional import score_direction
from scripts.train_structured_retriever import rank_metrics


class MetricTests(unittest.TestCase):
    def test_all_positive_metrics_with_stable_ties(self):
        scores = torch.tensor([[2., 2., 1., 0.], [0., 3., 2., 1.]])
        query_ids = ['q0', 'q1']
        candidates = ['e0', 'e1', 'e2', 'e3']
        truth = {'q0': {'e1', 'e3'}, 'q1': {'e1', 'e2'}}
        metrics, rows = rank_metrics(torch.arange(2)[:, None], torch.arange(4)[:, None],
                                    query_ids, candidates, truth, torch.device('cpu'), 2, 4,
                                    score_function=lambda q, p: scores[q[:, 0].long()])
        self.assertEqual([r['positive_ranks'] for r in rows], [[2, 4], [1, 2]])
        self.assertEqual(metrics['hit@1'], .5)
        self.assertEqual(metrics['recall@1'], .25)
        self.assertEqual(metrics['official_reactzyme_mean_rank'], 2.25)
        self.assertEqual(metrics['official_reactzyme_mrr_all_relevant'], .5625)
        self.assertEqual(metrics['mrr'], .75)

    def test_official_verifier_and_direction_neutral_output(self):
        scores = torch.arange(60., 0., -1.)[None].repeat(2, 1)
        candidates = [f'c{i:02d}' for i in range(60)]
        truth = {'q0': {'c00', 'c03'}, 'q1': {'c01', 'c10'}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'R-E'
            summary = score_direction(scores, ['q0', 'q1'], candidates, truth, output)
            self.assertEqual(summary['hit1'], .5)
            content = (output / 'per_query_ranks.jsonl').read_text()
            self.assertIn('positive_candidate_ids', content)
            self.assertNotIn('positive_enzyme_ids', content)
            self.assertTrue((output / 'verification.json').is_file())


if __name__ == '__main__':
    unittest.main()
