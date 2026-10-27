"""Checks for fixed ranking, split isolation, counts and non-overwriting output."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('budgets', ROOT / 'scripts/build_week2_budgets.py')
budgets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budgets)


class BudgetChecks(unittest.TestCase):
    def setUp(self):
        self.split = budgets.read_csv(ROOT / 'manifests/split_subjects.csv')
        self.v1 = budgets.read_csv(ROOT / 'manifests/label_subsets_v1.csv')
        self.counts = budgets.read_csv(ROOT / 'manifests/label_record_counts_v2.csv')

    def test_exact_preservation_nested_subjects_and_known_counts(self):
        original = copy.deepcopy(self.v1)
        budgets.validate_sources(self.split, self.v1)
        v2, stats, _ = budgets.build_tables(self.v1, self.counts)
        self.assertEqual(self.v1, original)
        self.assertEqual([{k: r[k] for k in original[0]} for r in v2], original)
        chosen = [r['subject_id'] for r in v2 if r['label10'] == 1]
        self.assertEqual(chosen, ['SC456', 'SC421', 'SC474', 'SC412', 'SC403'])
        sets = [{r['subject_id'] for r in v2 if int(r[f'label{b}'])} for b, _ in budgets.BUDGETS]
        self.assertEqual(list(map(len, sets)), [5, 14, 27, 54])
        self.assertTrue(all(a <= b for a, b in zip(sets, sets[1:])))
        self.assertEqual((stats[0]['labeled_records'], stats[0]['labeled_epochs'], stats[0]['N1']),
                         (10, 11936, 1719))
        self.assertEqual(stats[-1]['labeled_epochs'], 132938)

    def test_reject_validation_subject_and_duplicate_subject(self):
        bad = copy.deepcopy(self.v1)
        bad[0]['subject_id'] = next(r['subject_id'] for r in self.split if r['split'] == 'val')
        with self.assertRaises(ValueError):
            budgets.validate_sources(self.split, bad)
        with self.assertRaises(ValueError):
            budgets.validate_sources(self.split + [self.split[0]], self.v1)

    def test_reject_reranking_and_changed_legacy_flags(self):
        for column in ('sampling_seed', 'sampling_rank', 'label25', 'record_ids'):
            bad = copy.deepcopy(self.v1)
            bad[0][column] = '0'
            with self.assertRaises(ValueError, msg=column):
                budgets.validate_sources(self.split, bad)

    def test_reject_missing_duplicate_or_external_record_counts(self):
        for counts in (self.counts[:-1], self.counts + [self.counts[0]],
                       self.counts + [dict(self.counts[0], record_id='SC9991')]):
            with self.assertRaises(ValueError):
                budgets.build_tables(self.v1, counts)

    def test_reject_negative_class_and_epoch_mismatch(self):
        for column, value in [('N1', '-1'), ('n_epochs', '999999')]:
            bad = copy.deepcopy(self.counts)
            bad[0][column] = value
            with self.assertRaises(ValueError):
                budgets.build_tables(self.v1, bad)

    def test_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.csv'
            budgets.write_csv(path, ('a',), [{'a': 1}])
            with self.assertRaises(FileExistsError):
                budgets.write_csv(path, ('a',), [{'a': 2}])
            self.assertEqual(path.read_text(), 'a\n1\n')


if __name__ == '__main__':
    unittest.main()
