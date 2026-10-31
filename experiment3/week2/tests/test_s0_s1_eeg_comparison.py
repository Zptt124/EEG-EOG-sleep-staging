"""Information isolation, paired seeds, early stopping and cache corruption checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_s0_s1_eeg_comparison as stage


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id='SC4001_0010', subject_id='SC400', record_id='SC4001',
                          epoch_index='10', split='train', y_true='1', feature_path='SC4001.npy', row_index='0'),
                     dict(sample_id='SC4011_0020', subject_id='SC401', record_id='SC4011',
                          epoch_index='20', split='train', y_true='2', feature_path='SC4011.npy', row_index='0'),
                     dict(sample_id='SC4061_0030', subject_id='SC406', record_id='SC4061',
                          epoch_index='30', split='val', y_true='4', feature_path='SC4061.npy', row_index='0')]
        self.splits = [dict(subject_id=r['subject_id'], record_ids=r['record_id'], split=r['split']) for r in self.rows]
        self.subsets = [dict(subject_id='SC400', sampling_seed='2024', label10='1', label25='1'),
                        dict(subject_id='SC401', sampling_seed='2024', label10='0', label25='1')]

    def test_reject_incomplete_index(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            stage.audit_index(self.rows, self.splits)

    def test_reject_duplicate_test_and_path_traversal(self):
        for field, value in [('sample_id', self.rows[1]['sample_id']), ('split', 'test'),
                             ('feature_path', '../SC4001.npy'), ('row_index', '1')]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                stage.audit_index(rows, self.splits, strict=False)

    def test_reject_misaligned_labels_and_order(self):
        stage.align_indices(self.rows, copy.deepcopy(self.rows))
        changed = copy.deepcopy(self.rows)
        changed[0]['y_true'] = '0'
        for bad in [changed, self.rows[::-1], self.rows[:-1]]:
            with self.assertRaises(ValueError):
                stage.align_indices(self.rows, bad)

    def test_budget_and_normalization_exclude_unlabeled_and_validation(self):
        x = np.array([[2., 9.], [1e8, -1e8], [-1e9, 1e9]], dtype=np.float32)
        selected, train, val = stage.selected_positions(self.rows, self.subsets, 'label10', strict=False)
        self.assertEqual(selected, {'SC400'})
        np.testing.assert_array_equal(train, [0])
        np.testing.assert_array_equal(val, [2])
        mean, std = stage.fit_normalization(x[train])
        np.testing.assert_array_equal(mean, [2., 9.])
        np.testing.assert_array_equal(std, np.full(2, 1e-6, dtype=np.float32))

    def test_budget_cannot_include_val_subject(self):
        self.subsets[0]['subject_id'] = 'SC406'
        with self.assertRaisesRegex(ValueError, 'non-training'):
            stage.selected_positions(self.rows, self.subsets, 'label10', strict=False)

    def test_fixed_float32_pool_matches_reference(self):
        raw = np.random.default_rng(0).normal(size=(2, 20, 30, 128)).astype(np.float16)
        actual = stage.pool_block(raw)
        np.testing.assert_array_equal(actual, raw.astype(np.float32).mean(axis=2).reshape(2, 2560))
        self.assertEqual(actual.dtype, np.float32)
        raw[0, 0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            stage.pool_block(raw)

    def test_tied_score_does_not_replace_best(self):
        improved, best, stale = stage.selection_update(.7, .7, 9)
        self.assertFalse(improved)
        self.assertEqual((best, stale), (.7, 10))
        self.assertEqual(stage.selection_update(.71, .7, 9), (True, .71, 0))

    def test_paired_seed_initialization_and_rng_order(self):
        model0, _ = stage.new_head(2)
        order0 = torch.randperm(20)
        model1, _ = stage.new_head(2)
        order1 = torch.randperm(20)
        for p, q in zip(model0.parameters(), model1.parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertTrue(torch.equal(order0, order1))
        model2, _ = stage.new_head(3)
        self.assertFalse(torch.equal(model0.weight, model2.weight))

    def test_s1_corrupt_raw_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/'source'; source.mkdir()
            output = base/'output'; output.mkdir()
            stage.write_json(source/'summary.json', dict(status='complete', split_counts=stage.EXPECTED,
                                                        test_used=False, selected_weight_sha256='fixed'))
            stage.write_json(source/'feature_config.json', dict(checkpoint_sha256='fixed', representation='S1',
                                                                adaptation_steps=10000, weight_role='selected_new_ema'))
            preparation = dict(selected_weight_sha256='fixed')
            stage.write_json(source/'extraction_binding.json', preparation)
            prep_path = base/'preparation.json'; stage.write_json(prep_path, preparation)
            np.save(source/'SC4001.npy', np.zeros((1, 20, 30, 128), dtype=np.float16))
            stage.write_json(source/'SC4001.done.json', dict(record_id='SC4001', n_samples=1,
                                                             sha256='corrupt', weight_sha256='fixed'))
            with mock.patch.dict(stage.SOURCES, {'S1': source}), mock.patch.object(stage, 'PREPARATION', prep_path):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    stage.audit_and_pool('S1', self.rows[:1], {'SC4001': [(0, self.rows[0])]}, output)
            self.assertFalse((output/'S1_pooled.npy').exists())

    def test_runtime_binding_refuses_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = Path(directory)/'binding.json'
            stage.write_json(binding, dict(status='bound_before_head_training',
                                          input_sha256={'scripts/run_s0_s1_eeg_comparison.py': 'changed'}))
            with self.assertRaisesRegex(ValueError, 'bound input changed'):
                stage.validate_binding(binding)


if __name__ == '__main__':
    unittest.main()
