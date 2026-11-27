"""Saved-result identity and aggregation tests; no model execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'analysis' / 'verify_results.py'
SPEC = importlib.util.spec_from_file_location('verify_results', SOURCE)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class VerificationTests(unittest.TestCase):
    def identity(self):
        return {'scored_window_ids':np.array([3,7]),'vehicle_ids':np.array(['a','b']),
                'input_end':np.array([10,20]),'target_time':np.array([15,25]),
                'truth':np.array([[0.,0.],[1.,2.]])}

    def test_join_reorders_consistently(self):
        ref=self.identity(); candidate={k:x[::-1] for k,x in ref.items()}
        np.testing.assert_array_equal(v.check_identity(candidate,ref),[1,0])

    def test_join_rejects_duplicate_and_missing_ids(self):
        for ids in ([3,3],[3,9]):
            candidate=self.identity();candidate['scored_window_ids']=np.array(ids)
            with self.assertRaises(ValueError): v.check_identity(candidate,self.identity())

    def test_join_rejects_target_and_time_mismatch(self):
        for key in ['truth','target_time']:
            candidate=self.identity();candidate[key]=candidate[key]+1
            with self.assertRaisesRegex(ValueError,'Identity mismatch'):
                v.check_identity(candidate,self.identity())

    def test_strict_50m_and_new_harms(self):
        identity=self.identity();errors={'RAN':np.array([51.,50.]),'fusion_full':np.array([50.,51.])}
        rows,_=v.score(errors,{'all':np.ones(2,bool)},identity,'primary','nominal',42,0)
        row=next(r for r in rows if r['model']=='fusion_full')
        self.assertEqual((row['ran_failures'],row['rescues'],row['new_harms'],row['final_failures']),(1,1,1,1))

    def test_seed_first_average_is_not_pooled(self):
        base=dict(family='small_speed',condition='noise',model='fusion_full',stratum='all')
        rows=[]
        for seed,values in [(42,[1.,3.,5.]),(43,[10.])]:
            for rep,value in enumerate(values):
                rows.append(dict(base,seed=seed,realization=rep,**{k:value for k in v.METRICS}))
        per_seed,summary=v.summarize(rows)
        self.assertEqual(summary[0]['mean_m'],6.5)
        self.assertEqual([r['mean_m'] for r in per_seed],[3.,10.])

    def test_contributions_partition_the_overall_reduction(self):
        identity=self.identity()
        errors={'RAN':np.array([8.,4.]),'fusion_full':np.array([3.,5.])}
        masks={'left':np.array([True,False]),'right':np.array([False,True])}
        _,rows=v.score(errors,masks,identity,'primary','nominal',42,0)
        self.assertEqual(sum(r['contribution_m'] for r in rows),2.)
        self.assertEqual([r['share_of_seed_total_pct'] for r in rows],[125.,-25.])

    def test_large_cohort_rejects_whole_labelled_population(self):
        n=249900
        contract={'opportunity_ids':np.arange(n),'scored_window_ids':np.arange(n),
                  'label_available':np.ones(n,bool),'vehicle_ids':np.array(['a']*n),
                  'input_end':np.arange(n),'target_time':np.arange(n)+5,
                  'truth':np.zeros((n,2)),'exact_lag_common':np.ones(n,bool)}
        with self.assertRaisesRegex(ValueError,'220918'):
            v.large_cohort(contract)


if __name__=='__main__':
    unittest.main()
