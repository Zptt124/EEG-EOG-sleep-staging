"""Outer-person isolation, immutable rosters, signal-only access and FP16 parity."""
import builtins
import dis
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_subject_holdout as job


class SubjectHoldoutTests(unittest.TestCase):
    def subjects(self):return [f'S{i:04d}' for i in range(54)]

    def test_three_complete_disjoint_outer_folds_are_deterministic(self):
        subjects=self.subjects();assignment=job.outer_assignment(subjects)
        self.assertEqual(assignment,job.outer_assignment(list(reversed(subjects))))
        heldouts=[]
        for f in range(3):
            train,held=job.fold_people(assignment,f)
            self.assertEqual(len(train),36);self.assertEqual(len(held),18)
            self.assertFalse(set(train)&set(held));self.assertEqual(set(train)|set(held),set(subjects))
            heldouts+=held
        self.assertCountEqual(heldouts,subjects)
        with self.assertRaises(ValueError):job.outer_assignment(subjects[:-1])

    def test_all_nights_of_a_person_use_same_outer_fold(self):
        assignment=job.outer_assignment(self.subjects())
        records=[dict(subject_id=s,record_id=s+night) for s in assignment for night in ['1','2']]
        for fold in range(3):
            train,held=job.fold_people(assignment,fold)
            held_records={r['record_id'] for r in records if r['subject_id'] in held}
            train_records={r['record_id'] for r in records if r['subject_id'] in train}
            self.assertFalse(held_records&train_records)
            self.assertEqual(len(held_records),36)

    def test_filter_prior_orders_preserves_order_and_nested_budgets(self):
        subjects=self.subjects();assignment=job.outer_assignment(subjects)
        original={s:list(np.random.default_rng(s).permutation(subjects)) for s in job.ROSTERS}
        for f in range(3):
            train,held=job.fold_people(assignment,f);orders=job.roster_orders(train,original)
            for seed,order in orders.items():
                self.assertEqual(order,[s for s in original[seed] if s in train])
                self.assertTrue(set(order[:4])<set(order[:9]))
                self.assertTrue(set(order[:9]).isdisjoint(held))
        self.assertEqual(job.BUDGETS,dict(n4=4,n9=9))

    def test_fold_reader_rejects_heldout_even_when_original_split_train(self):
        reader=job.FoldSignalReader(Path('/nonexistent'),{'train1':'train','held1':'train','dev1':'val'},['train1'],{})
        with mock.patch.object(job.ref.SignalOnlyReader,'read_member') as parent:
            for record in ['held1','dev1','unknown']:
                with self.assertRaisesRegex(ValueError,'heldout'):reader.read_member(record,'signal.npy')
            parent.assert_not_called()

    def test_stage_labels_rejected_before_any_zip_access(self):
        reader=job.FoldSignalReader(Path('/nonexistent'),{'train1':'train'},['train1'],{})
        with mock.patch.object(job,'sha') as digest,mock.patch.object(job.ref.SignalOnlyReader,'read_member') as parent:
            with self.assertRaisesRegex(ValueError,'Label'):reader.read_member('train1','labels.npy')
            digest.assert_not_called();parent.assert_not_called()

    def test_allowed_signal_checks_hash_and_delegates_whitelist(self):
        reader=job.FoldSignalReader(Path('/data'),{'train1':'train'},['train1'],{'train1':'expected'})
        with mock.patch.object(job,'sha',return_value='expected') as digest,mock.patch.object(job.ref.SignalOnlyReader,'read_member',return_value=np.array([1])) as parent:
            np.testing.assert_array_equal(reader.read_member('train1','signal.npy'),[1])
            reader.read_member('train1','epoch_index.npy')
            self.assertEqual(digest.call_count,1);self.assertEqual(parent.call_count,2)
        reader=job.FoldSignalReader(Path('/data'),{'train1':'train'},['train1'],{'train1':'expected'})
        with mock.patch.object(job,'sha',return_value='wrong'),self.assertRaisesRegex(ValueError,'source changed'):
            reader.read_member('train1','signal.npy')

    def test_full_provider_cannot_have_been_used_before_restriction(self):
        original=SimpleNamespace(reader=SimpleNamespace(accesses=[dict(record='bad')]),examples=[])
        with self.assertRaisesRegex(ValueError,'already read'):
            job.restrict_provider(original,[],Path('/data'),{},{},0)

    def test_new_provider_is_exact36_and_cannot_sample_outer18(self):
        subjects=self.subjects();assignment=job.outer_assignment(subjects)
        train,held=job.fold_people(assignment,0)
        people=[dict(subject_id=s,record_ids=s+'1;'+s+'2') for s in train]
        original=SimpleNamespace(reader=SimpleNamespace(accesses=[]),examples=[])
        provider=job.restrict_provider(original,people,Path('/data'),{},{},0)
        self.assertEqual({r['subject_id'] for r in provider.subjects},set(train))
        self.assertTrue(all(not any(rec.startswith(s) for s in held) for rec in provider.reader.allowed_records))
        self.assertEqual(len(provider.reader.allowed_records),72)
        with self.assertRaises(ValueError):job.restrict_provider(original,people[:-1],Path('/data'),{},{},0)

    def test_direct_pool_matches_old_store_FP16_then_mean_path_exactly(self):
        x=np.random.default_rng(10).normal(size=(3,20,30,128)).astype(np.float32)
        legacy=x.astype(np.float16).astype(np.float32).mean(axis=2,dtype=np.float32).reshape(3,2560)
        np.testing.assert_array_equal(job.pool_like_original(x),legacy)
        self.assertEqual(job.pool_like_original(x).dtype,np.float32)
        self.assertFalse(np.array_equal(legacy,x.mean(axis=2).reshape(3,2560)))
        with self.assertRaises(ValueError):job.pool_like_original(x[:,:,:29])

    def test_array_digest_detects_mutation_and_chunk_order(self):
        x=np.arange(300*3,dtype=np.float32).reshape(300,3)
        a=job.array_digest(x);self.assertEqual(a,job.array_digest(x.copy()))
        x[129,1]+=1.;self.assertNotEqual(a,job.array_digest(x))

    def test_final_matrix_is72_with_matched_baselines(self):
        plan=job.plan();self.assertEqual(len(plan),72)
        self.assertEqual(len({r['run_id'] for r in plan}),72)
        for f in range(3):
            for r in job.ROSTERS:
                for b in job.BUDGETS:
                    cell=[x for x in plan if x['fold']==f and x['roster_seed']==r and x['budget']==b]
                    self.assertEqual({x['source_key'] for x in cell},{'S0','S1_aseed0','S1_aseed1','S1_aseed2'})

    def test_all_runtime_global_names_exist(self):
        for obj in list(job.__dict__.values()):
            if getattr(obj,'__module__',None)!=job.__name__ or not hasattr(obj,'__code__'):continue
            missing={i.argval for i in dis.get_instructions(obj) if i.opname=='LOAD_GLOBAL'
                     and i.argval not in obj.__globals__ and not hasattr(builtins,i.argval)}
            self.assertEqual(missing,set(),obj.__name__)


if __name__=='__main__':unittest.main()
