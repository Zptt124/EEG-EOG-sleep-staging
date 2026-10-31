"""Guard seed identity, baseline reuse and exact numerical-method preservation."""
import copy
import builtins
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
import run_replication_features_heads as job
import replication_feature_head_runtime as runtime


class RepeatFeatureHeadChecks(unittest.TestCase):
    def config(self):
        cfg = yaml.safe_load(job.PILOT_FEATURE_CONFIG.read_text())
        parent = ROOT/'configs/adapt_W2_C_S1_aseed1_v01.yaml'
        cfg.update(config_id='W2_C_S1_aseed1_features_v01', adaptation_seed=1,
                   adaptation_config=parent.relative_to(ROOT).as_posix(), adaptation_config_sha256=job.sha(parent),
                   adaptation_run='runs/W2_C_EEG_S1_aseed1_steps10000_v01',
                   S1_feature_root='features/eeg_cond14_S1_aseed1_steps10000_v01')
        return cfg

    def test_head_math_rng_and_selection_source_identical_to_pilot(self):
        expected = inspect.getsource(job.heads.train_head)
        expected = expected.replace('context, official):', 'context, official, adaptation_seed):')
        expected = expected.replace('adaptation_seed=0 if', 'adaptation_seed=adaptation_seed if')
        expected = expected.replace('fixed adaptation seed 0 for S1.', 'fixed adaptation seed {adaptation_seed} for S1.')
        self.assertEqual(inspect.getsource(runtime.train_head).strip(), expected.strip())

    def test_extraction_math_and_loop_source_identical_to_pilot(self):
        expected = inspect.getsource(job.extraction.extract_locked).replace('adaptation_seed=0,', "adaptation_seed=config['adaptation_seed'],")
        self.assertEqual(inspect.getsource(runtime.extract_locked).strip(), expected.strip())

    def test_audit_and_pool_math_identical_to_pilot(self):
        expected = inspect.getsource(job.heads.audit_and_pool)
        expected = expected.replace('def audit_and_pool(representation, rows, grouped, output):',
                                    'def audit_and_pool(representation, rows, grouped, output, folder, preparation_path):')
        expected = expected.replace('    folder = SOURCES[representation]\n', '')
        expected = expected.replace('json.loads(PREPARATION.read_text())', 'json.loads(preparation_path.read_text())')
        self.assertEqual(inspect.getsource(runtime.audit_and_pool).strip(), expected.strip())

    def test_exact_registered_feature_derivation(self):
        self.assertEqual(job.validate_feature_derivation(self.config()), 1)
        revised = self.config()
        revised.update(feature_revision=2, config_id='W2_C_S1_aseed1_features_v02',
                       S1_feature_root='features/eeg_cond14_S1_aseed1_steps10000_v02')
        self.assertEqual(job.validate_feature_derivation(revised), 1)

    def test_all_runtime_global_function_references_are_bound(self):
        for function in [runtime.extract_locked, runtime.audit_and_pool, runtime.train_head]:
            missing = {instruction.argval for instruction in dis.get_instructions(function)
                       if instruction.opname == 'LOAD_GLOBAL'
                       and instruction.argval not in function.__globals__
                       and not hasattr(builtins, instruction.argval)}
            self.assertEqual(missing, set(), function.__name__)

    def test_reject_transform_drift_and_seed0_cache_destination(self):
        for key, value in [('random_input_noise', True), ('lead_condition', 10), ('batch_size', 8),
                           ('S1_feature_root', 'features/eeg_cond14_S1_aseed0_steps10000_v01'),
                           ('adaptation_run', 'runs/W2_C_EEG_S1_aseed0_steps10000_v01')]:
            cfg = self.config(); cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                job.validate_feature_derivation(cfg)

    def test_reject_selected_checkpoint_from_wrong_adaptation_seed(self):
        job.check_checkpoint_seed({'adaptation_seed': 1}, 1)
        for checkpoint in [{}, {'adaptation_seed': 0}, {'adaptation_seed': 2}]:
            with self.assertRaises(ValueError):
                job.check_checkpoint_seed(checkpoint, 1)

    def test_baseline_full_match_rejects_normalization_seed_or_lr_drift(self):
        cfg = yaml.safe_load((ROOT/'runs/W2_C_S0_EEG_label10_sseed2024_seed0_v01/config.yaml').read_text())
        args = ('label10', 0, set(cfg['training_subjects']), 11936, set(cfg['validation_subjects']))
        job.check_baseline_config(cfg, *args)
        for key, value in [('adaptation_seed', 0), ('downstream_seed', 1), ('learning_rate', .01),
                           ('normalization', 'all training and validation subjects')]:
            bad = copy.deepcopy(cfg); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                job.check_baseline_config(bad, *args)

    def test_bound_input_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'binding.json'
            path.write_text(json.dumps(dict(status='bound_before_full_extraction',
                input_sha256={'scripts/run_replication_features_heads.py': 'wrong'})))
            with self.assertRaisesRegex(ValueError, 'Pipeline input changed'):
                job.validate_binding(path)

    def test_synthetic_head_executes_and_saves_correct_seed_metadata(self):
        official = job.heads.official_module()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runtime, 'ROOT', Path(directory)):
            plan = dict(run_id='synthetic_seed1', downstream_seed=0, representation='S1', budget='label10')
            rows = [dict(sample_id=f'SC4001_{i:04d}', subject_id='SC400', record_id='SC4001', epoch_index=str(i), y_true='0')
                    for i in range(2)]
            train = torch.zeros((4, 2560))
            val = torch.zeros((2, 2560))
            mean, std = np.zeros(2560, dtype=np.float32), np.ones(2560, dtype=np.float32)
            # Stop deliberately at the real-data coverage gate after exercising
            # config I/O, optimizer/selection, checkpoint reload and prediction I/O.
            prior_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with self.assertRaisesRegex(ValueError, 'Validation prediction coverage differs'):
                    runtime.train_head(plan, rows, train, torch.zeros(4, dtype=torch.long), val,
                                       np.zeros(2, dtype=int), mean, std, {}, official, adaptation_seed=1)
            finally:
                torch.set_num_threads(prior_threads)
            folder = Path(directory)/'runs/synthetic_seed1'
            cfg = yaml.safe_load((folder/'config.yaml').read_text())
            self.assertEqual(cfg['adaptation_seed'], 1)
            checkpoint = torch.load(folder/'best_head.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['config']['adaptation_seed'], 1)
            self.assertTrue((folder/'predictions_val.csv').exists())


if __name__ == '__main__':
    unittest.main()
