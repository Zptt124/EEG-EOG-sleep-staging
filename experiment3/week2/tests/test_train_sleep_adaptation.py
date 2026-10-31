"""EMA/frozen-state and exact optimizer/RNG/sampler recovery on CPU toy models.

No real EEG training, hyperparameter search or formal optimization updates here.
"""
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'vendor/EEGDM'))
import train_sleep_adaptation as trainer


def make_model():
    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    model.register_buffer('fixed_buffer', torch.tensor([0.123456789]))
    return model


def make_provider(seed=0):
    reader = SimpleNamespace(accesses=[])
    return SimpleNamespace(order=[3,1,2], rng=np.random.default_rng(seed), examples=[], reader=reader)


class TrainRuntimeChecks(unittest.TestCase):
    def test_frozen_buffer_changes_are_rejected(self):
        model=make_model()
        ema=trainer.FixedEMA(model,['weight'])
        frozen=trainer.reference.state_digest(model,frozen_only=True)
        trainer.assert_frozen(model,ema,frozen)
        with torch.no_grad(): model.fixed_buffer.add_(1)
        with self.assertRaises(ValueError): trainer.assert_frozen(model,ema,frozen)

    def test_ema_initializes_exactly_and_never_averages_frozen_state(self):
        model = make_model()
        ema = trainer.FixedEMA(model, ['weight'])
        original = {k:v.clone() for k,v in model.state_dict().items()}
        for key,value in original.items():
            torch.testing.assert_close(ema.state[key],value,rtol=0,atol=0)
        with torch.no_grad():
            model.weight.add_(10)
            model.bias.add_(7)
            model.fixed_buffer.add_(2)
        ema.update(model)
        torch.testing.assert_close(ema.state['bias'],original['bias'],rtol=0,atol=0)
        torch.testing.assert_close(ema.state['fixed_buffer'],original['fixed_buffer'],rtol=0,atol=0)
        self.assertEqual(ema.updates,1)

    def test_fixed_ema_equation_and_counter(self):
        model = make_model()
        ema = trainer.FixedEMA(model,['weight'],0.999)
        expected = model.weight.detach().clone()
        for _ in range(3):
            with torch.no_grad(): model.weight.add_(0.1)
            expected.mul_(0.999).add_(model.weight.detach(),alpha=0.001)
            ema.update(model)
        torch.testing.assert_close(ema.state['weight'],expected,rtol=0,atol=1e-7)
        self.assertEqual(ema.updates,3)

    def test_global_rng_roundtrip(self):
        random.seed(19); np.random.seed(19); torch.manual_seed(19)
        state=trainer.capture_rng()
        expected=(random.random(),np.random.rand(4),torch.randn(4))
        random.seed(90); np.random.seed(90); torch.manual_seed(90)
        trainer.restore_rng(state)
        self.assertEqual(random.random(),expected[0])
        np.testing.assert_array_equal(np.random.rand(4),expected[1])
        torch.testing.assert_close(torch.randn(4),expected[2],rtol=0,atol=0)

    def test_actual_reference_sampler_roundtrip(self):
        class FakeSignalReader:
            def __init__(self): self.accesses=[]
            def read_member(self,record,member):
                self.accesses.append(dict(record_id=record,split='train',member=member))
                if member=='epoch_index.npy': return np.array([0,1,5,7])
                return np.arange(4*2*3000,dtype=np.float32).reshape(4,2,3000)/100
        subjects=[dict(subject_id='SC400',record_ids='SC4001;SC4002')]
        a=trainer.reference.TrainingSignalBatches(FakeSignalReader(),subjects,2,0)
        a.next(1)
        state=trainer.capture_sampler(a)
        expected=a.next(2)
        b=trainer.reference.TrainingSignalBatches(FakeSignalReader(),subjects,2,99)
        trainer.restore_sampler(b,state)
        np.testing.assert_array_equal(b.next(2),expected)
        self.assertEqual(a.examples,b.examples)
        self.assertEqual(a.reader.accesses,b.reader.accesses)

    def test_cpu_checkpoint_resume_matches_uninterrupted_training_bitwise(self):
        def setup():
            random.seed(9); np.random.seed(9); torch.manual_seed(9)
            model=make_model()
            opt=torch.optim.AdamW([model.weight],lr=1e-5,foreach=False,weight_decay=0)
            return model,opt,trainer.FixedEMA(model,['weight']),make_provider()
        def update(model,opt,ema):
            x=torch.randn(2,2)*float(np.random.uniform(0.5,1.5))
            target=torch.full((2,1),random.random())
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(x),target)
            loss.backward(); opt.step(); ema.update(model)
            return float(loss.detach())
        a,oa,ea,pa=setup()
        expected=[update(a,oa,ea) for _ in range(6)]
        b,ob,eb,pb=setup()
        prefix=[update(b,ob,eb) for _ in range(3)]
        payload=trainer.make_recovery(b,ob,eb,pb,3,[dict(step=i+1,loss=v) for i,v in enumerate(prefix)],'config','binding',0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recovery.pt'
            info=trainer.atomic_save(path,payload)
            self.assertEqual(info['sha256'],trainer.sha(path))
            self.assertFalse(path.with_name('recovery.pt.partial').exists())
            saved=torch.load(path,map_location='cpu',weights_only=False)
        c,oc,ec,pc=setup()
        trainer.restore_recovery(saved,c,oc,ec,pc,'config','binding')
        resumed=prefix+[update(c,oc,ec) for _ in range(3)]
        self.assertEqual(expected,resumed)
        for key,value in a.state_dict().items():
            torch.testing.assert_close(c.state_dict()[key],value,rtol=0,atol=0)
            torch.testing.assert_close(ec.state[key],ea.state[key],rtol=0,atol=0)
        self.assertEqual(ec.updates,6)

    def test_recovery_rejects_wrong_config_or_counter(self):
        model=make_model(); opt=torch.optim.AdamW([model.weight]); ema=trainer.FixedEMA(model,['weight']); provider=make_provider()
        payload=trainer.make_recovery(model,opt,ema,provider,0,[],'good','binding',0)
        with self.assertRaises(ValueError):
            trainer.restore_recovery(payload,model,opt,ema,provider,'wrong','binding')
        bad=copy.deepcopy(payload); bad['ema_updates']=1
        with self.assertRaises(ValueError):
            trainer.restore_recovery(bad,model,opt,ema,provider,'good','binding')

    def test_binding_rejects_untested_or_changed_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'binding.json'
            for tests,source in [(0,trainer.sha(trainer.__file__)),(7,'wrong')]:
                path.write_text(json.dumps(dict(config_sha256='config',unit_tests_passed=tests,trainer_sha256=source)))
                with self.assertRaises(ValueError): trainer.validate_binding(path,'config')


if __name__=='__main__': unittest.main()
