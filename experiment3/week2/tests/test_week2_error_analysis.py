import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_week2_c_results import changes,context_groups


class ErrorAnalysisChecks(unittest.TestCase):
    def test_N1_recovery_loss_and_false_positive_are_distinct(self):
        y=np.array([1,1,1,0,2,3,4])
        baseline=np.array([0,1,2,1,2,0,4])
        adapted=np.array([1,2,2,2,1,3,4])
        out=changes(y,baseline,adapted)
        self.assertEqual(out['recovered_N1'],1)
        self.assertEqual(out['lost_N1'],1)
        self.assertEqual(out['N1_false_positive_removed'],1)
        self.assertEqual(out['N1_false_positive_added'],1)
        self.assertEqual(out['exact_stage_recovered'],2)
        self.assertEqual(out['exact_stage_lost'],2)

    def test_context_uses_original_indices_and_never_crosses_record(self):
        rows=[dict(record_id='A',epoch_index=str(i),y_true=str(y)) for i,y in [(10,1),(11,1),(12,1),(13,2),(15,1)]]
        rows.append(dict(record_id='B',epoch_index='14',y_true='1'))
        self.assertEqual(context_groups(rows).tolist(),['gap_or_record_edge','stable_neighbors','label_transition',
                                                      'gap_or_record_edge','gap_or_record_edge','gap_or_record_edge'])


if __name__=='__main__':unittest.main()
