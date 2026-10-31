"""A repeat may alter seeds and artifact identities, never the scientific method."""
import copy
from pathlib import Path
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_adaptation_replication as repeat


class ReplicationChecks(unittest.TestCase):
    def setUp(self):
        self.parent = yaml.safe_load(repeat.PARENT.read_text(encoding='utf-8'))

    def test_preregistered_seed_one_and_two_exact_changes(self):
        for seed in (1, 2):
            derived = repeat.derive(self.parent, seed)
            self.assertTrue(repeat.validate_derivation(self.parent, derived, seed))
            self.assertEqual(derived['sampling']['adaptation_seed'], seed)
            self.assertEqual(derived['sampling']['data_seed'], seed)
            self.assertEqual(derived['sampling']['noise_seed'], seed)
            self.assertEqual(derived['initialization'], self.parent['initialization'])
            self.assertEqual(derived['training'], self.parent['training'])

    def test_parent_is_never_mutated(self):
        original = copy.deepcopy(self.parent)
        repeat.derive(self.parent, 1)
        self.assertEqual(self.parent, original)

    def test_reject_unregistered_seed(self):
        for seed in (0, 3, -1):
            with self.assertRaises(ValueError):
                repeat.derive(self.parent, seed)

    def test_reject_hyperparameter_or_budget_drift(self):
        for path, value in [(('training', 'learning_rate'), 2e-5),
                            (('training', 'max_successful_optimizer_updates'), 20000),
                            (('ema', 'beta'), .99),
                            (('model', 'trainable_prefixes'), ['layers.19.'])]:
            derived = repeat.derive(self.parent, 1)
            repeat.set_nested(derived, path, value)
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'beyond seed'):
                repeat.validate_derivation(self.parent, derived, 1)

    def test_reject_seed0_output_collision(self):
        for path in [('checkpoints', 'run_root'), ('feature_extraction_after_adaptation', 'feature_root_S1')]:
            derived = repeat.derive(self.parent, 1)
            value = self.parent[path[0]][path[1]]
            repeat.set_nested(derived, path, value)
            with self.assertRaises(ValueError):
                repeat.validate_derivation(self.parent, derived, 1)

    def test_reject_weight_inheritance_or_validation_access(self):
        for path, value in [(('initialization', 'checkpoint'), 'runs/seed0/selected_ema_step10000.pt'),
                            (('initialization', 'inherit_optimizer_or_ema_counters'), True),
                            (('data', 'read_validation_signals'), True),
                            (('data', 'sleep_stage_labels_visible'), True),
                            (('stopping_and_selection', 'choose_by_N1_or_macro_f1'), True)]:
            derived = repeat.derive(self.parent, 1)
            repeat.set_nested(derived, path, value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                repeat.validate_derivation(self.parent, derived, 1)


if __name__ == '__main__':
    unittest.main()
