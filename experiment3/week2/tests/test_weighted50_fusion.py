"""Training-only class weights, matched optimization, and fixed final selection."""
import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_weighted50_fusion as job


class WeightedBudgetChecks(unittest.TestCase):
    def test_inverse_sqrt_formula_and_all_five_classes(self):
        labels=np.repeat(np.arange(5),[1,4,9,16,25])
        counts,weights=job.class_weights(labels)
        expected=1./np.sqrt([1.,4.,9.,16.,25.]);expected/=expected.mean()
        np.testing.assert_array_equal(counts,[1,4,9,16,25])
        np.testing.assert_array_equal(weights,expected.astype(np.float32))
        self.assertEqual(weights.dtype,np.float32)
        self.assertTrue(np.all(np.diff(weights)<0))

    def test_missing_or_invalid_training_classes_are_rejected(self):
        for bad in [np.array([],dtype=int),np.array([0,1,2,3]),np.arange(6),np.arange(5,dtype=float),np.arange(5).reshape(1,5)]:
            with self.subTest(labels=str(bad)),self.assertRaises(ValueError):job.class_weights(bad)

    def test_weighted_mean_and_gradient_match_analytical_objective(self):
        _,weights=job.class_weights(np.repeat(np.arange(5),[100,25,9,4,1]))
        logits=torch.zeros((2,5),requires_grad=True);labels=torch.tensor([0,1]);w=torch.from_numpy(weights)
        loss=torch.nn.functional.cross_entropy(logits,labels,weight=w)
        nll=-logits.log_softmax(1)[torch.arange(2),labels]
        expected=(nll*w[labels]).sum()/w[labels].sum()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertGreater(logits.grad[1].abs().sum().item(),logits.grad[0].abs().sum().item())

    def test_optimizer_arithmetic_changes_only_the_loss_weights(self):
        def core(function,weighted=False):
            tree=ast.parse(inspect.getsource(function))
            epoch=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
            if weighted:
                inner=epoch.body[2]
                inner.body=[n for n in inner.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                    and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                    and n.value.func.value.id=='batch_digest')]
                for n in ast.walk(inner):
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cross_entropy':
                        n.keywords=[k for k in n.keywords if k.arg!='weight']
            return [ast.dump(n,include_attributes=False) for n in epoch.body[:7]]
        self.assertEqual(core(job.train_weighted50,True),core(job.baseline.train_final50))

    def test_matrix_has_forty_new_runs_and_forty_matched_references(self):
        entries=job.plan()
        self.assertEqual(len(entries),40)
        self.assertEqual(len({e['run_id'] for e in entries}),40)
        self.assertEqual(len({e['fixed50_run_id'] for e in entries}),40)
        self.assertEqual({(e['adaptation_seed'],e['budget'],e['downstream_seed']) for e in entries},
            {(a,b,s) for a in [None,0,1,2] for b in ['label10','label25'] for s in range(5)})
        self.assertTrue(all(e['run_id'].startswith('W2X_C_WCE50_') and e['run_id']!=e['fixed50_run_id'] for e in entries))

    def test_uniform_weights_replay_unweighted_training_and_declining_validation_still_runs_50(self):
        real=job.heads.official_module();calls=0
        def declining(y,p):
            nonlocal calls
            calls+=1
            return dict(macro_f1=1.-.01*calls)
        fake=SimpleNamespace(compute_metrics=declining,**{name:getattr(real,name) for name in
            ['PROB_COLS','STAGE_NAMES','load_predictions','evaluate','per_subject_table']})
        prior_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);reference=root/'reference';reference.mkdir()
                x=torch.linspace(-.05,.05,10*5120).reshape(10,5120);y=torch.tensor([0,1,2,3,4]*2)
                vx=x[:5].clone();vy=np.arange(5)
                mean=np.zeros(5120,dtype=np.float32);std=np.ones(5120,dtype=np.float32)
                seed=2;model,optimizer=job.heads.new_head(seed,dimension=5120)
                for _ in range(50):
                    model.train()
                    for ids in torch.randperm(len(x)).split(128):
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids]);loss.backward();optimizer.step()
                cfg=dict(early_stopping=False,selected_epoch=50,max_epochs=50,runtime_binding_sha256='unweighted-binding',
                    test_used=False,selection='fixed final epoch 50; validation never selects weights',loss='unweighted_cross_entropy')
                (reference/'config.yaml').write_text(yaml.safe_dump(cfg))
                torch.save(dict(model=model.state_dict(),epoch=50,mean=torch.from_numpy(mean),std=torch.from_numpy(std),config=cfg),reference/'final_head_epoch50.pt')
                rows=[dict(sample_id=f'case_{i}',subject_id='SC400',record_id='SC4001',epoch_index=str(i),y_true=str(i)) for i in range(5)]
                entry=dict(run_id='synthetic_weighted',fixed50_run_id='synthetic_fixed50',representation='S1',
                    adaptation_seed=1,source_key='S1_aseed1',budget='label10',downstream_seed=seed)
                with mock.patch('builtins.print'):
                    result=job.train_weighted50(entry,x,y,vx,vy,mean,std,rows,fake,reference,root/'new','weighted-binding')
                self.assertEqual(calls,50)
                self.assertEqual((result['epochs'],result['selected_epoch']),(50,50))
                final=torch.load(root/'new/final_head_epoch50.pt',map_location='cpu',weights_only=True)
                self.assertEqual(final['config']['class_weights'],[1.]*5)
                self.assertEqual(final['config']['optimizer_updates'],50)
                self.assertFalse((root/'new/best_head.pt').exists())
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(final['model'][key],value,rtol=1e-6,atol=1e-7)
                audit=json.loads((root/'new/paired_design_audit.json').read_text())
                self.assertTrue(audit['matches_locked_unweighted_initializer_and_rng_recipe'])
                self.assertTrue(audit['normalization_matches_unweighted_exactly'])
                real.load_predictions(root/'new/predictions_val.csv')
        finally:torch.set_num_threads(prior_threads)


if __name__=='__main__':unittest.main()
