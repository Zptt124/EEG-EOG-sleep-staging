"""Fixed-budget diagnostics preserve optimization and never select by validation."""
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
import run_fixed50_fusion as job
import train_primary_fusion_head as original


class FixedBudgetChecks(unittest.TestCase):
    def test_complete_matched_matrix_uses_new_run_names(self):
        entries=job.plan()
        self.assertEqual(len(entries),40)
        self.assertEqual(len({r['run_id'] for r in entries}),40)
        self.assertEqual({(r['adaptation_seed'],r['budget'],r['downstream_seed']) for r in entries},
            {(a,b,s) for a in [None,0,1,2] for b in ['label10','label25'] for s in range(5)})
        self.assertTrue(all(r['run_id']!=r['original_run_id'] and r['run_id'].startswith('W2X_C_F50_') for r in entries))

    def test_original_training_arithmetic_is_identical(self):
        def core(function):
            code=inspect.getsource(function).replace('heads.predict(', 'predict(')
            tree=ast.parse(code)
            epoch=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
            return [ast.dump(n,include_attributes=False) for n in epoch.body[:7]]
        self.assertEqual(core(job.train_final50),core(original.train_head))

    def test_declining_validation_cannot_stop_or_select_an_earlier_head(self):
        real=job.heads.official_module()
        calls=0
        def declining(y,p):
            nonlocal calls
            calls+=1
            return dict(macro_f1=1.-calls*.01)
        fake=SimpleNamespace(compute_metrics=declining,**{name:getattr(real,name) for name in
            ['PROB_COLS','STAGE_NAMES','load_predictions','evaluate','per_subject_table']})
        prior_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);old=root/'original';old.mkdir()
                seed=3
                x=torch.linspace(-.1,.1,4*5120).reshape(4,5120);y=torch.tensor([0,1,2,3])
                vx=x[:2].clone();vy=np.array([0,1])
                mean=np.zeros(5120,dtype=np.float32);std=np.ones(5120,dtype=np.float32)
                rows=[dict(sample_id=f'case_{i}',subject_id='SC400',record_id='SC4001',epoch_index=str(i),y_true=str(i)) for i in range(2)]
                cfg=dict(runtime_binding_sha256='original-binding',test_used=False)
                (old/'config.yaml').write_text(yaml.safe_dump(cfg))
                model,optimizer=job.heads.new_head(seed,dimension=5120);history=[]
                for epoch in range(1,4):
                    model.train();total=0.
                    for ids in torch.randperm(len(x)).split(128):
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids]);loss.backward();optimizer.step()
                        total+=loss.item()*len(ids)
                    model.eval();pred=job.heads.predict(model,vx);score=declining(vy,pred)['macro_f1']
                    history.append(dict(epoch=epoch,train_loss=total/len(x),val_macro_f1=score))
                    if epoch==1:
                        torch.save(dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),epoch=1,config=cfg),old/'best_head.pt')
                (old/'history.json').write_text(json.dumps(history));calls=0
                entry=dict(run_id='synthetic_fixed50',original_run_id='synthetic_original',representation='S1',
                           adaptation_seed=0,source_key='S1_aseed0',budget='label10',downstream_seed=seed)
                with mock.patch('builtins.print'):
                    result=job.train_final50(entry,x,y,vx,vy,mean,std,rows,fake,old,root/'new','new-binding')
                self.assertEqual(calls,50)
                self.assertEqual((result['epochs'],result['selected_epoch']),(50,50))
                final=torch.load(root/'new/final_head_epoch50.pt',map_location='cpu',weights_only=True)
                self.assertEqual(final['epoch'],50)
                self.assertEqual(final['config']['optimizer_updates'],50)
                self.assertFalse(final['config']['early_stopping'])
                self.assertFalse((root/'new/best_head.pt').exists())
                self.assertTrue(result['original_history_and_selected_weights_reproduced'])
                real.load_predictions(root/'new/predictions_val.csv')
                first=torch.load(old/'best_head.pt',map_location='cpu',weights_only=True)
                self.assertFalse(torch.equal(first['model']['weight'],final['model']['weight']))
        finally:
            torch.set_num_threads(prior_threads)


if __name__=='__main__':unittest.main()
