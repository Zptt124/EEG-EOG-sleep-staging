"""Reject premature/wrong weights, unsafe caches and corrupted sample identities."""
import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import extract_adapted_features as stage


class S1FeatureChecks(unittest.TestCase):
    def setUp(self):
        self.config=yaml.safe_load(stage.DEFAULT_CONFIG.read_text(encoding='utf-8'))
        self.summary=dict(status='completed',completed_updates=10000,ema_updates=10000,
            config_sha256=self.config['adaptation_config_sha256'],trainer_sha256='source',run_id='run',
            frozen_parameters_unchanged=True,stage_labels_read=False,validation_signals_read=False,test_signals_read=False,
            selected_weight=dict(path=self.config['selected_filename'],sha256='weight'))
        self.selected=dict(weight_role='selected_new_ema',completed_updates=10000,ema_updates=10000,
            config_sha256=self.summary['config_sha256'],runtime_source_sha256='source',run_id='run',
            original_checkpoint_sha256=self.config['original_checkpoint_sha256'],
            selected_by='fixed_step_10000_no_external_validation',state_dict={'x':1})
        self.partitions={'SC4001':'train','SC4061':'val','SC4071':'test'}
        self.rows=[dict(sample_id='SC4001_0010',subject_id='SC400',record_id='SC4001',epoch_index='10',split='train',
            y_true='0',feature_path='SC4001.npy',row_index='0',config_id='S0'),
            dict(sample_id='SC4001_0020',subject_id='SC400',record_id='SC4001',epoch_index='20',split='train',
            y_true='1',feature_path='SC4001.npy',row_index='1',config_id='S0'),
            dict(sample_id='SC4061_0010',subject_id='SC406',record_id='SC4061',epoch_index='10',split='val',
            y_true='2',feature_path='SC4061.npy',row_index='0',config_id='S0')]

    def test_reject_running_or_incomplete_adaptation(self):
        stage.check_completed_training(self.summary,self.config)
        for key,value in [('status','running'),('completed_updates',9000),('ema_updates',9999),('stage_labels_read',True)]:
            wrong=copy.deepcopy(self.summary); wrong[key]=value
            with self.assertRaises(ValueError): stage.check_completed_training(wrong,self.config)

    def test_reject_online_intermediate_or_wrong_checkpoint(self):
        stage.check_selected_metadata(self.selected,self.summary,self.config)
        for key,value in [('weight_role','online_recovery_diagnostic_only'),('completed_updates',9000),
                          ('runtime_source_sha256','changed'),('selected_by','best_N1')]:
            wrong=copy.deepcopy(self.selected); wrong[key]=value
            with self.assertRaises(ValueError): stage.check_selected_metadata(wrong,self.summary,self.config)

    def test_index_rejects_duplicate_ids_and_row_reordering(self):
        stage.validate_index(self.rows,self.partitions,strict=False)
        for rows in [self.rows+[self.rows[0]], [self.rows[1],self.rows[0],self.rows[2]]]:
            with self.assertRaises(ValueError): stage.validate_index(rows,self.partitions,strict=False)

    def test_index_rejects_test_and_unsafe_feature_paths(self):
        wrong=copy.deepcopy(self.rows); wrong[0]['split']='test'
        with self.assertRaises(ValueError): stage.validate_index(wrong,self.partitions,strict=False)
        wrong=copy.deepcopy(self.rows); wrong[0]['feature_path']='../other.npy'
        with self.assertRaises(ValueError): stage.validate_index(wrong,self.partitions,strict=False)

    def test_signal_reader_never_reads_npz_labels_and_checks_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'SC4001.npz'
            with zipfile.ZipFile(path,'w') as archive:
                for name,value in [('signal',np.zeros((2,2,3000),dtype=np.float32)),('epoch_index',np.array([10,20]))]:
                    stream=io.BytesIO(); np.save(stream,value,allow_pickle=False); archive.writestr(name+'.npy',stream.getvalue())
                archive.writestr('labels.npy',b'INVALID: THIS MUST NOT BE READ')
            value=stage.load_signal_record(directory,'SC4001',self.partitions,self.rows[:2])
            self.assertEqual(value.shape,(2,3000))
            with self.assertRaises(ValueError): stage.load_signal_record(directory,'SC4071',self.partitions,self.rows[:2])
            with self.assertRaises(ValueError): stage.load_signal_record(directory,'SC4001',self.partitions,self.rows[1::-1])

    def test_s0_cache_overwrite_and_descendant_rejected(self):
        for name in ['features/eeg_cond14_v01','features/eeg_cond14_v01/nested']:
            wrong=copy.deepcopy(self.config); wrong['S1_feature_root']=name
            with self.assertRaises(ValueError): stage.safe_output_root(wrong)

    def test_float32_pooling_and_nonfinite_guard(self):
        x=np.linspace(0,1,2*20*30*128,dtype=np.float32).reshape(2,20,30,128).astype(np.float16)
        actual=stage.pooled_features(x)
        expected=x.astype(np.float32).mean(axis=2).reshape(2,2560)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual.dtype,np.float32)
        x[0,0,0,0]=np.nan
        with self.assertRaises(ValueError): stage.pooled_features(x)

    def test_feature_contract_rejects_noise_or_changed_layer_query(self):
        stage.validate_feature_contract(self.config)
        for key,value in [('random_input_noise',True),('query','inter'),('reuse_S0_kernel_cache',True),('source_channel',1)]:
            wrong=copy.deepcopy(self.config); wrong[key]=value
            with self.assertRaises(ValueError): stage.validate_feature_contract(wrong)


if __name__=='__main__':unittest.main()
