"""Tiny synthetic checks of the public-data adapter; no benchmark files are read."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("hier_prepare_data_test_target", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        directory = self.raw / "time_split"
        directory.mkdir(parents=True)
        cleaned, molecules, train = [], [], {}

        def add_source(eid, molecule_set, sequence, ec="1.1.1.1"):
            cleaned.append({"Entry": eid, "Sequence": sequence, "EC number": ec})
            molecules.append({"uniprot_id": eid, "molecules": molecule_set})

        # Distinct accession/EC identities share their complete input sequence.
        add_source("TrainB", "CC", "MAA", "2.2.2.2")
        add_source("TrainA", "CC", "MAA", "1.1.1.1")
        train["z_key"] = ("CC", "MAA")
        train["a_key"] = ("CC", "MAA")
        for ordinal in range(2, 20):
            molecule_set, sequence = "C" * (ordinal + 2), "MV" + "A" * ordinal
            add_source(f"E{ordinal:03d}", molecule_set, sequence)
            train[f"key_{20 - ordinal:02d}"] = (molecule_set, sequence)
        add_source("TestB", "N", "MGG", "2.2.2.2")
        add_source("TestA", "N", "MGG", "1.1.1.1")
        add_source("TestC", "O", "MGG", "3.3.3.3")
        test = {"t_second": ("N", "MGG"), "t_first": ("N", "MGG"), "t_other": ("O", "MGG")}
        pd.DataFrame(cleaned).to_csv(self.raw / "cleaned_uniprot_rhea.tsv", sep="\t", index=False)
        pd.DataFrame(molecules).to_csv(self.raw / "uniprot_molecules.tsv", sep="\t", index=False)
        torch.save(train, directory / "positive_train_val_time.pt")
        torch.save(test, directory / "positive_test_time.pt")
        self.entries = prepare.accession_lookup(self.raw)

    def build(self):
        return prepare.build_split(self.raw, "time", self.entries)

    def test_pair_split_and_training_accessions_are_preserved(self):
        pairs, enzymes, _, metadata = self.build()
        self.assertEqual(pairs.split.value_counts().to_dict(), {"train": 18, "test": 3, "val": 2})
        # Known first two permuted ordinals for default_rng(42) with n=20.
        self.assertEqual(set(pairs.loc[pairs.split == "val", "source_archive_ordinal"]), {9, 15})
        self.assertEqual(set(pairs.loc[pairs.split == "test", "source_archive_ordinal"]), {0, 1, 2})
        self.assertEqual(metadata["official_train_val_records"], 20)
        self.assertEqual(metadata["official_test_records"], 3)
        self.assertEqual(len(pairs), 23)
        shared = enzymes.loc[enzymes.sequence == "MAA"]
        self.assertEqual(shared.enzyme_id.tolist(), ["TrainA", "TrainB"])
        self.assertEqual(shared.uniprot_id.tolist(), ["TrainA", "TrainB"])
        training = pairs.loc[pairs.enzyme_id.isin(["TrainA", "TrainB"])].set_index("enzyme_id")
        self.assertEqual(training.loc["TrainA", "source_pair_id"], "positive_train_val_time:z_key")
        self.assertEqual(training.loc["TrainB", "source_pair_id"], "positive_train_val_time:a_key")
        self.assertEqual(set(training.split), {"train"})

    def test_evaluation_deduplicates_sequences_without_changing_pairs(self):
        pairs, enzymes, _, _ = self.build()
        before = pairs.copy(deep=True)
        contract = prepare.evaluation_contract(pairs, enzymes, self.root / "processed", "test")
        self.assertEqual(contract["candidate_ids"], ["TestA"])
        self.assertEqual(contract["query_ids"], sorted([prepare.reaction_id("N"), prepare.reaction_id("O")]))
        self.assertEqual(set(map(tuple, contract["relevance"].values())), {("TestA",)})
        self.assertEqual(contract["source_positive_rows"], 3)
        self.assertEqual(contract["unique_positive_pairs"], 2)
        self.assertEqual(contract["identical_sequence_alias_count"], 2)
        pd.testing.assert_frame_equal(pairs, before)
        self.assertTrue({"TestA", "TestB", "TestC"}.issubset(set(enzymes.enzyme_id)))

    def test_data_only_cli_marks_features_missing(self):
        output = self.root / "processed"
        argv = [str(SCRIPT), "--raw-dir", str(self.raw), "--output-dir", str(output), "--splits", "time"]
        with mock.patch("sys.argv", argv), redirect_stdout(io.StringIO()) as stdout:
            prepare.main()
        config = json.loads((output / "time" / "config.json").read_text())
        self.assertFalse(config["features_ready"])
        self.assertTrue(config["missing_features"])
        self.assertFalse(config["official_split_view"])
        self.assertEqual(config["training_splits"], ["train"])
        self.assertFalse(config["refit_train_val"])
        self.assertFalse((output / "time" / "residue_view.json").exists())
        self.assertIn("training is NOT ready", stdout.getvalue())
        self.assertEqual(len(pd.read_parquet(output / "time" / "pairs.parquet")), 23)

    def test_unmapped_positive_is_an_error_not_a_dropped_record(self):
        del self.entries[("CC", "MAA")]
        with self.assertRaisesRegex(ValueError, "No unambiguous official accession"):
            self.build()

    def test_relative_feature_paths_use_reference_directory(self):
        parent = self.root / "features"
        self.assertEqual(prepare.resolve_path("esm2/mean.safetensors", parent), parent / "esm2" / "mean.safetensors")
        absolute = self.root / "elsewhere" / "mean.safetensors"
        self.assertEqual(prepare.resolve_path(str(absolute), parent), absolute)


if __name__ == "__main__":
    unittest.main()
