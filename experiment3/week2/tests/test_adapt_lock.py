"""Negative checks for contamination, hidden budgets and post-hoc selection."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lock', ROOT / 'scripts/validate_adapt_lock.py')
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)


class AdaptLockChecks(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(lock.DEFAULT_CONFIG.read_text(encoding='utf-8'))

    def test_contract_and_actual_files(self):
        result = lock.validate_locked_files()
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['formal_training_started'])

    def test_reject_label_or_external_signal_access(self):
        for key in ['sleep_stage_labels_visible', 'read_validation_signals', 'read_test_signals']:
            config = copy.deepcopy(self.config)
            config['data'][key] = True
            with self.assertRaises(ValueError, msg=key):
                lock.validate_contract(config)

    def test_reject_hidden_updates_and_budget_changes(self):
        for key, value in [('extra_warmup_updates', 100), ('diagnostic_updates_inherited', 100),
                           ('max_successful_optimizer_updates', 20000)]:
            config = copy.deepcopy(self.config)
            config['training'][key] = value
            with self.assertRaises(ValueError, msg=key):
                lock.validate_contract(config)

    def test_reject_posthoc_backbone_selection(self):
        for key in ['early_stopping', 'external_validation_used_for_backbone_selection',
                    'choose_by_adaptation_loss', 'choose_by_N1_or_macro_f1']:
            config = copy.deepcopy(self.config)
            config['stopping_and_selection'][key] = True
            with self.assertRaises(ValueError, msg=key):
                lock.validate_contract(config)

    def test_reject_changed_ema_policy_or_s0_overwrite(self):
        config = copy.deepcopy(self.config)
        config['ema']['update_every'] = 10
        with self.assertRaises(ValueError):
            lock.validate_contract(config)
        config = copy.deepcopy(self.config)
        config['feature_extraction_after_adaptation']['overwrite_S0'] = True
        with self.assertRaises(ValueError):
            lock.validate_contract(config)

    def test_reject_tampered_config_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            # Equivalent YAML semantics still changes the locked byte identity.
            path.write_text(lock.DEFAULT_CONFIG.read_text(encoding='utf-8') + '\n# edited\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                lock.validate_locked_files(path, lock.DEFAULT_CONFIG.with_suffix('.lock.json'))


if __name__ == '__main__':
    unittest.main()
