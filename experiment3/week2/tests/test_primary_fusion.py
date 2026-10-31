"""Primary fusion preserves the registered math, dimensions and local provenance."""
import builtins
import copy
import dis
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_primary_fusion as job
import train_primary_fusion_head as head


class PrimaryFusionChecks(unittest.TestCase):
    def config(self):
        return json.loads((job.REPORT/'fusion_config.json').read_text())

    def test_registered_numerical_source_preserved(self):
        expected = inspect.getsource(job.runtime.train_head)
        expected = expected.replace('model, optimizer = new_head(seed)', 'model, optimizer = new_head(seed, dimension=5120)')
        expected = expected.replace("modality='EEG'", "modality='EEG_EOG'")
        expected = expected.replace("role='diagnostic', input_dim=2560, classifier='Linear(2560,5)'",
                                    "role='primary', input_dim=5120, classifier='Linear(5120,5)'")
        expected = expected.replace('EEG-only diagnostic; fixed adaptation seed {adaptation_seed} for S1.',
                                    'EEG+EOG primary comparison; adaptation seed {adaptation_seed}.')
        self.assertEqual(inspect.getsource(head.train_head).strip(), expected.strip())

    def test_runtime_dependencies_are_bound(self):
        for function in [head.train_head, job.validate_config, job.plan, job.concatenate,
                         job.wait_for_eog, job.verify_eeg_pools, job.run]:
            missing = {i.argval for i in dis.get_instructions(function) if i.opname == 'LOAD_GLOBAL'
                       and i.argval not in function.__globals__ and not hasattr(builtins, i.argval)}
            self.assertEqual(missing, set(), function.__name__)

    def test_hyperparameter_or_budget_drift_rejected(self):
        cfg = self.config()
        self.assertTrue(job.validate_config(cfg))
        for key, value in [('learning_rate', .01), ('max_epochs', 100), ('loss', 'weighted_cross_entropy')]:
            bad = copy.deepcopy(cfg); bad['classifier'][key] = value
            with self.assertRaises(ValueError):
                job.validate_config(bad)
        bad = copy.deepcopy(cfg); bad['budgets'] = ['label25']
        with self.assertRaises(ValueError):
            job.validate_config(bad)

    def test_all_40_positions_and_three_adaptations_are_unconditional(self):
        rows = job.plan(self.config())
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({r['run_id'] for r in rows}), 40)
        self.assertEqual({(r['adaptation_seed'], r['budget'], r['downstream_seed']) for r in rows},
                         {(a, b, s) for a in [None, 0, 1, 2] for b in ['label10','label25'] for s in range(5)})

    def test_concatenation_dimension_order_and_dtype(self):
        eeg = np.ones((2, 2560), dtype=np.float32)
        eog = np.full((2, 2560), 2, dtype=np.float32)
        combined = job.concatenate(eeg, eog)
        self.assertEqual(combined.shape, (2, 5120))
        np.testing.assert_array_equal(combined[:, :2560], eeg)
        np.testing.assert_array_equal(combined[:, 2560:], eog)
        for bad in [eog[:1], eog.astype(np.float16)]:
            with self.assertRaises(ValueError):
                job.concatenate(eeg, bad)

    def test_normalization_is_per_feature_across_modalities(self):
        eeg = np.stack([np.full(2560, 1.), np.full(2560, 3.)]).astype(np.float32)
        eog = np.stack([np.full(2560, 100.), np.full(2560, 104.)]).astype(np.float32)
        combined = job.concatenate(eeg, eog)
        mean, std = job.heads.fit_normalization(combined)
        a, b = job.heads.fit_normalization(eeg)
        c, d = job.heads.fit_normalization(eog)
        np.testing.assert_array_equal(mean, np.concatenate([a,c]))
        np.testing.assert_array_equal(std, np.concatenate([b,d]))

    def test_local_cache_cannot_be_mislabeled_as_received_D_bytes(self):
        for field, value in [('D_full_cache_received', True), ('D_raw_file_byte_equivalence_unverified', False),
                             ('test_used', True)]:
            cfg = self.config(); cfg[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                job.validate_config(cfg)

    def test_synthetic_head_executes_with_5120_columns_and_primary_metadata(self):
        official = job.heads.official_module()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(head, 'ROOT', Path(directory)):
            entry = dict(run_id='synthetic_primary', representation='S1', budget='label10', downstream_seed=0)
            rows = [dict(sample_id=f'SC4001_{i:04d}', subject_id='SC400', record_id='SC4001', epoch_index=str(i), y_true='0')
                    for i in range(2)]
            prior = torch.get_num_threads(); torch.set_num_threads(1)
            try:
                with self.assertRaisesRegex(ValueError, 'Validation prediction coverage differs'):
                    head.train_head(entry, rows, torch.zeros((4,5120)), torch.zeros(4,dtype=torch.long),
                                    torch.zeros((2,5120)), np.zeros(2,dtype=int), np.zeros(5120,dtype=np.float32),
                                    np.ones(5120,dtype=np.float32), {}, official, adaptation_seed=2)
            finally:
                torch.set_num_threads(prior)
            folder = Path(directory)/'runs/synthetic_primary'
            cfg = yaml.safe_load((folder/'config.yaml').read_text())
            self.assertEqual((cfg['input_dim'],cfg['modality'],cfg['role'],cfg['adaptation_seed']),
                             (5120,'EEG_EOG','primary',2))
            checkpoint = torch.load(folder/'best_head.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['model']['weight'].shape, (5,5120))


if __name__ == '__main__':
    unittest.main()
