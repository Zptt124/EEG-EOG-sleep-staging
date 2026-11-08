"""Subject isolation, deterministic rosters, analytic gradient and convergence."""
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
import run_eeg_multisubset as job


class EEGMultiSubsetChecks(unittest.TestCase):
    def world(self):
        subjects=[f'S{i:03d}' for i in range(54)]
        order=list(reversed(subjects))
        reference=[dict(subject_id=s,sampling_rank=str(i+1)) for i,s in enumerate(order)]
        return subjects,reference

    def test_rosters_preserve_reference_train_only_uniqueness_and_nesting(self):
        subjects,reference=self.world();orders=job.make_orders(subjects,reference)
        self.assertEqual(orders[2024],list(reversed(subjects)))
        self.assertEqual(orders,job.make_orders(list(reversed(subjects)),reference))
        for seed,order in orders.items():
            self.assertEqual(len(order),54);self.assertEqual(set(order),set(subjects))
            self.assertTrue(set(order[:5])<set(order[:14]))
        self.assertNotEqual(orders[4101],orders[4102])
        with self.assertRaises(ValueError):job.make_orders(subjects[:-1],reference)

    def test_inner_subject_partitions_never_overlap(self):
        for count in [5,14]:
            subjects=[f'P{i}' for i in range(count)];mapping=job.inner_folds(subjects);seen=[]
            for fold in range(3):
                train={s for s in subjects if mapping[s]!=fold};val={s for s in subjects if mapping[s]==fold}
                self.assertTrue(train.isdisjoint(val));self.assertEqual(train|val,set(subjects));seen.extend(val)
            self.assertCountEqual(seen,subjects)
        with self.assertRaises(ValueError):job.inner_folds(['A','A','B'])

    def test_analytic_softmax_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(15,3));y=np.tile(np.arange(5),3).astype(np.int64)
        theta=rng.normal(scale=.1,size=20);value,gradient=job.loss_gradient(theta,x,y,.03)
        numerical=[];epsilon=1e-6
        for i in range(len(theta)):
            a=theta.copy();b=theta.copy();a[i]+=epsilon;b[i]-=epsilon
            numerical.append((job.loss_gradient(a,x,y,.03)[0]-job.loss_gradient(b,x,y,.03)[0])/(2*epsilon))
        self.assertTrue(np.isfinite(value));np.testing.assert_allclose(gradient,numerical,rtol=1e-5,atol=1e-6)

    def test_stable_probabilities_for_extreme_scores(self):
        x=np.array([[1000.,-1000.],[-1000.,1000.]])
        theta=np.arange(15,dtype=np.float64)*1000
        p=job.predict_probabilities(theta,x)
        self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(axis=1),1.)

    def test_optimizer_converges_and_zero_start_is_reproducible(self):
        y=np.repeat(np.arange(5),20).astype(np.int64);rng=np.random.default_rng(4)
        x=np.eye(5)[y]*3+rng.normal(scale=.2,size=(100,5))
        a,info,_=job.solve(x,y,.01);b,again,_=job.solve(x,y,.01)
        self.assertTrue(info['converged']);self.assertLessEqual(info['gradient_inf'],job.GTOL)
        np.testing.assert_array_equal(a,b)
        self.assertGreater((job.predict_probabilities(a,x).argmax(1)==y).mean(),.95)

    def test_scipy_success_without_gradient_convergence_is_rejected(self):
        x=np.tile(np.eye(5),(2,1));y=np.tile(np.arange(5),2).astype(np.int64)
        fake=SimpleNamespace(x=np.ones(30),success=True,message='not actually converged',nit=1)
        with mock.patch.object(job,'minimize',return_value=fake),self.assertRaisesRegex(ValueError,'gradient tolerance'):
            job.solve(x,y,.01)

    def test_CV_normalization_excludes_inner_holdout_and_ties_use_stronger_L2(self):
        x=np.arange(15,dtype=np.float64).reshape(-1,1);x[10:]+=1000
        y=np.tile(np.arange(5),3).astype(np.int64);subjects=np.repeat(['A','B','C'],5)
        calls=[]
        def fit(tx,ty,mean,std,value,fold,train_subjects,val_subjects):
            keep=subjects!=['A','B','C'][fold]
            np.testing.assert_allclose(mean,x[keep].mean(axis=0))
            np.testing.assert_allclose(tx.mean(axis=0),0.,atol=1e-12)
            self.assertTrue(set(train_subjects).isdisjoint(val_subjects));calls.append((fold,value))
            return np.zeros(10),dict(gradient_inf=0.)
        def metric(y,p):return dict(macro_f1=.1)
        chosen,scores,records,oof=job.regularization_search(x,y,subjects,job.LAMBDAS,fit,metric)
        self.assertEqual(len(calls),9);self.assertEqual(chosen,max(job.LAMBDAS))
        self.assertTrue(all(len(v)==len(y) and (v>=0).all() for v in oof.values()))

    def test_labeled_positions_cannot_include_development_validation(self):
        rows=[dict(subject_id='A',split='train'),dict(subject_id='A',split='val'),dict(subject_id='B',split='train')]
        np.testing.assert_array_equal(job.labeled_positions(rows,['A']),[0])

    def test_complete_final_matrix_and_bound_runtime_names(self):
        plan=job.run_plan();self.assertEqual(len(plan),24);self.assertEqual(len({r['run_id'] for r in plan}),24)
        self.assertEqual({(r['roster_seed'],r['budget'],r['source_key']) for r in plan},
            {(s,b,k) for s in job.SEEDS for b in job.BUDGETS for k in ['S0','S1_aseed0','S1_aseed1','S1_aseed2']})
        self.assertTrue(all(r['budget'] in r['run_id'] for r in plan))
        for fn in [job.solve,job.make_orders,job.prepare,job.bind,job.verify_sources,job.save_fit,job.regularization_search,
                   job.run,job.evaluate_final,job.analyze]:
            missing={i.argval for i in dis.get_instructions(fn) if i.opname=='LOAD_GLOBAL'
                     and i.argval not in fn.__globals__ and not hasattr(builtins,i.argval)}
            self.assertEqual(missing,set(),fn.__name__)


if __name__=='__main__':unittest.main()
