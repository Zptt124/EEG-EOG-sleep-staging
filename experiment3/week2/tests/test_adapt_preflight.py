"""Data leakage guards, W1 preprocessing consistency and official target parity."""
import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/EEGDM'))
spec = importlib.util.spec_from_file_location('adapt', ROOT / 'scripts/adapt_preflight.py')
adapt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapt)


class AdaptGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        signal = np.stack([np.tile(np.linspace(-250, 250, 3000, dtype=np.float32), (2, 1))] * 4)
        with zipfile.ZipFile(self.directory / 'SC4001.npz', 'w') as archive:
            for name, array in [('signal', signal), ('epoch_index', np.array([10, 11, 14, 15]))]:
                stream = io.BytesIO()
                np.save(stream, array, allow_pickle=False)
                archive.writestr(name + '.npy', stream.getvalue())
            # Deliberately malformed. Accessing it would fail even without auditing.
            archive.writestr('labels.npy', b'THIS MEMBER MUST NEVER BE LOADED')
        self.reader = adapt.SignalOnlyReader(self.directory, {'SC4001': 'train', 'SC4011': 'val', 'SC4021': 'test'})
        self.subjects = [dict(subject_id='SC400', record_ids='SC4001')]

    def tearDown(self):
        self.tmp.cleanup()

    def test_reject_labels_and_val_test_before_open(self):
        for record, member in [('SC4001', 'labels.npy'), ('SC4011', 'signal.npy'), ('SC4021', 'signal.npy')]:
            with self.assertRaises(ValueError):
                self.reader.read_member(record, member)
        self.assertEqual(self.reader.accesses, [])

    def test_real_batch_uses_only_whitelisted_members_and_hidden_labels(self):
        provider = adapt.TrainingSignalBatches(self.reader, self.subjects, 2, 0)
        batch = provider.next(1)
        self.assertEqual(batch.shape, (2, 1, 1000))
        self.assertTrue(np.isfinite(batch).all())
        self.assertEqual({r['member'] for r in self.reader.accesses}, {'signal.npy', 'epoch_index.npy'})
        self.assertTrue(all(r['y_hidden'] == -1 for r in provider.examples))
        self.assertTrue(all(r['epoch_index'] in [10, 11, 14, 15] for r in provider.examples))

    def test_transform_exactly_matches_w1_full_epoch_then_crop(self):
        provider = adapt.TrainingSignalBatches(self.reader, self.subjects, 2, 4)
        result = provider.next(1)
        extract_spec = importlib.util.spec_from_file_location('extract', ROOT / 'scripts/extract_features.py')
        extract = importlib.util.module_from_spec(extract_spec)
        extract_spec.loader.exec_module(extract)
        original = self.reader.read_member('SC4001', 'signal.npy')
        for i, row in enumerate(provider.examples):
            transformed = extract.transform(original[row['row_index']:row['row_index']+1, 0, :])
            crop = row['segment_index']
            np.testing.assert_array_equal(result[i], transformed[0, :, crop*1000:(crop+1)*1000])

    def test_data_sampling_reproducible(self):
        a = adapt.TrainingSignalBatches(self.reader, self.subjects, 2, 7)
        b = adapt.TrainingSignalBatches(self.reader, self.subjects, 2, 7)
        np.testing.assert_array_equal(a.next(1), b.next(1))
        self.assertEqual(a.examples, b.examples)

    def test_split_rejects_overlap_and_changed_partition(self):
        rows = adapt.read_csv(ROOT / 'vendor/experiment1-review/manifests/split_subjects.csv')
        train, partitions = adapt.validate_split(rows)
        self.assertEqual(len(train), 54)
        self.assertEqual(sum(v == 'train' for v in partitions.values()), 106)
        with self.assertRaises(ValueError):
            adapt.validate_split(rows + [rows[0]])
        altered = [dict(r) for r in rows]
        altered[0]['split'] = 'val'
        with self.assertRaises(ValueError):
            adapt.validate_split(altered)

    def test_diffusion_target_matches_upstream_training_method(self):
        import torch
        from diffusers import DDPMScheduler
        from model.diffusion_model_pl import PLDiffusionModel
        scheduler = DDPMScheduler(num_train_timesteps=50, beta_start=0.0001, beta_end=0.05,
                                  beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
        generator = torch.Generator().manual_seed(0)
        clean = torch.randn(2, 1, 1000, generator=generator)
        noise = torch.randn(2, 1, 1000, generator=generator)
        times = torch.tensor([3, 40])
        for kind in ['v_prediction', 'epsilon', 'sample']:
            official_self = SimpleNamespace(pred_target=kind, noise_sch=scheduler)
            expected = PLDiffusionModel.get_pred_target(official_self, clean, noise, times)
            actual = adapt.diffusion_target(scheduler, kind, clean, noise, times)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            adapt.diffusion_target(scheduler, 'wrong', clean, noise, times)


if __name__ == '__main__':
    unittest.main()
