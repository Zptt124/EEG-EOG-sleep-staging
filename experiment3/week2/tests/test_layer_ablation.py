"""Layer-axis identity, modality isolation, and matched unweighted optimization."""
import builtins
import dis
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_layer_ablation as job


class LayerAblationChecks(unittest.TestCase):
    def test_predefined_layers_and_flattened_coordinates(self):
        np.testing.assert_array_equal(job.retained_columns('last5'),np.arange(15*128,20*128))
        np.testing.assert_array_equal(job.retained_columns('last2'),np.arange(18*128,20*128))
        np.testing.assert_array_equal(job.retained_columns('all20'),np.arange(2560))
        with self.assertRaises(ValueError):job.retained_columns('best_layers')

    def test_selection_from_existing_pool_equals_selection_before_time_pooling(self):
        raw=np.random.default_rng(11).normal(size=(2,20,30,128)).astype(np.float16)
        pooled=job.heads.pool_block(raw)
        for group,layers in job.GROUPS.items():
            from_pool=pooled[:,job.retained_columns(group)]
            from_raw=raw[:,layers,:,:].astype(np.float32).mean(axis=2,dtype=np.float32).reshape(2,-1)
            np.testing.assert_array_equal(from_pool,from_raw)

    def test_mask_changes_only_unselected_EEG_and_preserves_EOG(self):
        original=np.random.default_rng(3).normal(size=(3,5120)).astype(np.float32)
        for group in ['last5','last2']:
            values=original.copy();self.assertIs(job.mask_input(values,group),values)
            keep=job.retained_columns(group);drop=np.setdiff1d(np.arange(2560),keep)
            np.testing.assert_array_equal(values[:,keep],original[:,keep])
            np.testing.assert_array_equal(values[:,2560:],original[:,2560:])
            self.assertTrue((values[:,drop]==0).all())
        np.testing.assert_array_equal(job.mask_input(original.copy(),'all20'),original)

    def test_invalid_or_nonfinite_features_are_not_silently_hidden(self):
        for values in [np.ones((1,2560),np.float32),np.ones((1,5120),np.float16)]:
            with self.assertRaises(ValueError):job.mask_input(values,'last2')
        values=np.ones((1,5120),np.float32);values[0,0]=np.nan
        with self.assertRaises(ValueError):job.mask_input(values,'last2')

    def test_dropped_EEG_coordinates_have_zero_data_gradient(self):
        x=np.random.default_rng(4).normal(size=(4,5120)).astype(np.float32);job.mask_input(x,'last2')
        model=torch.nn.Linear(5120,5);loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(x)),torch.tensor([0,1,2,3]))
        loss.backward()
        drop=np.arange(18*128)
        self.assertTrue(bool((model.weight.grad[:,drop]==0).all()))
        self.assertGreater(model.weight.grad[:,18*128:].abs().sum().item(),0.)

    def test_eighty_new_runs_and_forty_shared_all20_references(self):
        entries=job.plan();self.assertEqual(len(entries),80)
        self.assertEqual(len({r['run_id'] for r in entries}),80)
        self.assertEqual(len({r['reference_run_id'] for r in entries}),40)
        self.assertEqual({r['layer_group'] for r in entries},{'last5','last2'})
        self.assertTrue(all(r['run_id']!=r['reference_run_id'] for r in entries))

    def test_runtime_global_dependencies_are_bound(self):
        for fn in [job.retained_columns,job.mask_input,job.plan,job.validate_reference,job.prepare,job.bind,
                   job.check_binding,job.verify_pools,job.layer_shift,job.train_masked50,job.get_metric,job.analyze,job.run]:
            missing={i.argval for i in dis.get_instructions(fn) if i.opname=='LOAD_GLOBAL'
                     and i.argval not in fn.__globals__ and not hasattr(builtins,i.argval)}
            self.assertEqual(missing,set(),fn.__name__)

    def test_all20_identity_replays_unweighted_final50_exactly(self):
        official=job.heads.official_module();previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);reference=root/'reference';reference.mkdir()
                x=torch.linspace(-.05,.05,10*5120).reshape(10,5120);y=torch.tensor([0,1,2,3,4]*2)
                vx=x[:5].clone();vy=np.arange(5);mean=np.zeros(5120,np.float32);std=np.ones(5120,np.float32)
                seed=2;model,optimizer=job.heads.new_head(seed,dimension=5120)
                for _ in range(50):
                    model.train()
                    for ids in torch.randperm(len(x)).split(128):
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids]);loss.backward();optimizer.step()
                cfg=dict(early_stopping=False,selected_epoch=50,max_epochs=50,runtime_binding_sha256='reference',
                    test_used=False,loss='unweighted_cross_entropy',selection='final_epoch_50')
                (reference/'config.yaml').write_text(yaml.safe_dump(cfg))
                torch.save(dict(model=model.state_dict(),epoch=50,mean=torch.from_numpy(mean),std=torch.from_numpy(std),config=cfg),reference/'final_head_epoch50.pt')
                rows=[dict(sample_id=f'case_{i}',subject_id='SC400',record_id='SC4001',epoch_index=str(i),y_true=str(i)) for i in range(5)]
                entry=dict(run_id='synthetic_all20',reference_run_id='reference',layer_group='all20',representation='S1',
                           adaptation_seed=0,source_key='S1_aseed0',budget='label10',downstream_seed=seed)
                with mock.patch('builtins.print'):
                    result=job.train_masked50(entry,x,y,vx,vy,mean,std,rows,official,reference,root/'new','new-binding')
                self.assertEqual((result['epochs'],result['selected_epoch']),(50,50))
                final=torch.load(root/'new/final_head_epoch50.pt',map_location='cpu',weights_only=True)
                for name,value in model.state_dict().items():self.assertTrue(torch.equal(final['model'][name],value),name)
                self.assertEqual(final['config']['loss'],'unweighted_cross_entropy')
                self.assertEqual(final['config']['active_EEG_dim'],2560)
                self.assertFalse((root/'new/best_head.pt').exists())
                self.assertEqual(json.loads((root/'new/paired_design_audit.json').read_text())['status'],'passed')
        finally:torch.set_num_threads(previous_threads)


if __name__=='__main__':unittest.main()
