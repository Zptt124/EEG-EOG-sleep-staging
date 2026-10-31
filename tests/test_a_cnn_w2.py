"""test_a_cnn_w2.py — 实验① A W2 代码的单元测试(无 GPU 依赖)。

运行:python -m unittest tests.test_a_cnn_w2 -v   (repo 根目录)
覆盖:模型形状/编码器维度、双通道参数化、标准化文件往返、加载例子核心逻辑、
预算嵌套关系、macro_f1 与 B 官方 evaluate.py 数值一致。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cnn_model import (CHANNEL_NAMES, EPOCH_PTS, SmallCNN, apply_normalization,  # noqa: E402
                       build_norm_json, load_encoder)
from train_cnn_w2 import labeled_records, load_subjects, macro_f1_np  # noqa: E402


class TestSmallCNN(unittest.TestCase):
    def test_shapes_and_encoder_dim(self):
        for c in (1, 2):
            model = SmallCNN(in_channels=c)
            x = torch.randn(4, c, EPOCH_PTS)
            out = model(x)
            self.assertEqual(out.shape, (4, 5))
            feats = model.encoder(x)
            self.assertEqual(feats.shape, (4, 128))
            self.assertEqual(feats.dtype, torch.float32)

    def test_dual_only_changes_first_layer_channels(self):
        s, d = SmallCNN(1), SmallCNN(2)
        self.assertEqual(s.features[0][0].weight.shape, (32, 1, 50))
        self.assertEqual(d.features[0][0].weight.shape, (32, 2, 50))
        # 其余层结构完全一致
        for name, p in s.state_dict().items():
            if name.startswith("features.0"):
                continue
            self.assertEqual(p.shape, d.state_dict()[name].shape, name)

    def test_invalid_channels_rejected(self):
        with self.assertRaises(ValueError):
            SmallCNN(in_channels=3)

    def test_encoder_state_dict_only_features(self):
        model = SmallCNN(2)
        sd = model.encoder_state_dict()
        self.assertTrue(all(k.startswith("features.") for k in sd))


class TestNormalization(unittest.TestCase):
    def test_build_norm_json(self):
        norm = build_norm_json([-0.1, -2.5], [21.1, 49.5],
                               {"nominal_budget": "label25"}, in_channels=2)
        self.assertEqual(norm["input_shape"], [2, 3000])
        self.assertEqual(len(norm["channels"]), 2)
        self.assertEqual(norm["channels"][0]["name"], CHANNEL_NAMES[0])

    def test_build_norm_json_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_norm_json([0.0], [1.0, 2.0], {}, in_channels=2)

    def test_apply_normalization(self):
        norm = build_norm_json([0.0, 10.0], [1.0, 5.0], {}, in_channels=2)
        x = torch.tensor([[[1.0], [15.0]]])  # (1,2,1)
        y = apply_normalization(x, norm)
        self.assertTrue(torch.allclose(y, torch.tensor([[[1.0], [1.0]]])))

    def test_export_load_roundtrip(self):
        model = SmallCNN(in_channels=2)
        with tempfile.TemporaryDirectory() as td:
            enc = Path(td)
            norm = build_norm_json([1.0, 2.0], [3.0, 4.0],
                                   {"nominal_budget": "label10",
                                    "labeled_subjects": ["SC456"]}, in_channels=2)
            torch.save({"in_channels": 2, "state_dict": model.encoder_state_dict(),
                        "output_dim": 128}, enc / "encoder_weights.pt")
            (enc / "normalization.json").write_text(json.dumps(norm), encoding="utf-8")
            loaded, norm2 = load_encoder(enc)
            self.assertFalse(loaded.training)          # eval 模式
            self.assertEqual(norm2["channels"][1]["std"], 4.0)
            for k, v in model.encoder_state_dict().items():
                self.assertTrue(torch.equal(loaded.state_dict()[k], v), k)


class TestBudgets(unittest.TestCase):
    def setUp(self):
        self.subset = load_subjects(REPO / "manifests" / "label_subsets_v2.csv")

    def test_nested_budgets(self):
        b10 = set(labeled_records(self.subset, "label10"))
        b25 = set(labeled_records(self.subset, "label25"))
        b50 = set(labeled_records(self.subset, "label50"))
        b100 = set(labeled_records(self.subset, "label100"))
        self.assertEqual(len(b10), 10)
        self.assertEqual(len(b25), 27)
        self.assertEqual(len(b50), 53)
        self.assertEqual(len(b100), 106)
        self.assertTrue(b10 < b25 < b50 < b100)

    def test_budget_subjects_in_train(self):
        split = np.loadtxt(REPO / "manifests" / "split_subjects.csv",
                           delimiter=",", skiprows=1, dtype=str)
        train_recs = set()
        for row in split:
            if row[2] == "train":
                train_recs.update(row[1].split(";"))
        for budget in ("label10", "label25", "label50", "label100"):
            self.assertTrue(set(labeled_records(self.subset, budget)) <= train_recs, budget)


class TestMacroF1(unittest.TestCase):
    def test_matches_official_definition(self):
        """与 B 官方 evaluate.py 的 metrics_from_confusion 数值一致。"""
        sys.path.pop(0)
        sys.path.insert(0, str(REPO / "src" / "evaluation"))
        from evaluate import metrics_from_confusion  # noqa: E402
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 5, 1000)
        y_pred = rng.integers(0, 5, 1000)
        cm = np.zeros((5, 5), dtype=np.int64)
        np.add.at(cm, (y_true, y_pred), 1)
        off = metrics_from_confusion(cm)
        self.assertAlmostEqual(macro_f1_np(y_true, y_pred), off["macro_f1"], places=10)


if __name__ == "__main__":
    unittest.main()
