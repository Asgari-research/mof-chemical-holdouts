"""Focused cardinality and numerical-integrity checks for final reviewer evidence."""
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
RES=HERE/'results'

class ClosureTests(unittest.TestCase):
    def test_saved_data_denominators_and_membership(self):
        d=pd.read_csv(RES/'target_iqr_denominators.csv')
        self.assertEqual(len(d),4)
        self.assertTrue((d.target_n==263744).all())
        self.assertTrue((d.iqr_mmol_g>0).all())
        e=pd.read_csv(RES/'topology_exact_co2_015bar.csv')
        self.assertEqual(e.n.sum(),263744)
        self.assertEqual(len(e),47)
        s=pd.read_csv(RES/'geometry_support_quartiles.csv')
        self.assertEqual(s.n.sum(),177411)
        self.assertEqual(len(s),4)

    def test_random_reconstruction(self):
        m=pd.read_csv(RES/'random_membership_reconstruction.csv')
        self.assertEqual(len(m),5)
        self.assertTrue((m.symmetric_difference_n==0).all())
        r=pd.read_csv(RES/'random_run_vs_average_rankings.csv')
        self.assertEqual(len(r),10)
        self.assertTrue((r.k==np.ceil(r.n*.05)).all())
        self.assertTrue(r.run_vs_new_average_jaccard.between(0,1).all())
        self.assertTrue(r.run_reference_recovery.between(0,1).all())
        self.assertTrue(r.new_average_reference_recovery.between(0,1).all())
        self.assertTrue((r.list_symmetric_difference_n%2==0).all())
        fixed=pd.read_csv(RES/'random_one_oos_sampling_sensitivity.csv')
        self.assertEqual(len(fixed),5)
        self.assertTrue((fixed.n==177411).all())
        self.assertTrue((fixed.k==8871).all())
        self.assertTrue((fixed.singletons==108167).all())
        self.assertTrue(fixed.one_vs_average_jaccard.between(0,1).all())
        self.assertTrue((fixed.list_symmetric_difference_n%2==0).all())

    def test_paired_chemistry(self):
        audit=json.loads((RES/'rac_join_audit.json').read_text())
        self.assertEqual(audit['covered_n']+audit['missing_n'],263744)
        self.assertEqual(audit['covered_n'],256328)
        self.assertFalse(set(audit['features_baseline'])&set(audit['rac_features_predeclared']))
        d=pd.read_csv(RES/'rac_paired_fold_metrics.csv')
        split=json.loads((RES/'rac_fold_groups.json').read_text())
        self.assertEqual(len(split),5)
        self.assertEqual(len(d),10)
        self.assertEqual(set(d.variant),{'original','original_plus_rac'})
        for fold,g in d.groupby('fold'):
            self.assertEqual(len(g),2)
            self.assertEqual(g.n_train.nunique(),1)
            self.assertEqual(g.n_test.nunique(),1)
            self.assertEqual(g.k.nunique(),1)
            self.assertTrue((g.k==np.ceil(g.n_test*.05)).all())
            info=split[fold]
            self.assertFalse(set(info['train_groups'])&set(info['test_groups']))
        x=pd.read_csv(RES/'rac_paired_fold_deltas.csv')
        self.assertEqual(len(x),5)
        self.assertTrue(np.isfinite(x[['delta_r2','delta_mae','delta_recovery']].to_numpy()).all())
        family=pd.read_csv(RES/'rac_paired_topology_errors.csv')
        sizes=family.groupby(['fold','variant']).n.sum().reset_index()
        self.assertTrue((sizes.merge(d[['fold','variant','n_test']],on=['fold','variant'],validate='one_to_one').eval('n == n_test')).all())

    def test_secondary_pressure_paired_chemistry(self):
        d=pd.read_csv(RES/'co2_0015bar_rac_paired_fold_metrics.csv')
        split=json.loads((RES/'co2_0015bar_rac_fold_groups.json').read_text())
        self.assertEqual(len(d),10)
        self.assertEqual(len(split),5)
        self.assertEqual(set(d.variant),{'original','original_plus_rac'})
        for fold,g in d.groupby('fold'):
            self.assertEqual(len(g),2)
            self.assertEqual(g.n_train.nunique(),1)
            self.assertEqual(g.n_test.nunique(),1)
            self.assertFalse(set(split[fold]['train_groups'])&set(split[fold]['test_groups']))
            self.assertTrue((g.k==np.ceil(g.n_test*.05)).all())
        family=pd.read_csv(RES/'co2_0015bar_rac_paired_topology_errors.csv')
        sizes=family.groupby(['fold','variant']).n.sum().reset_index()
        self.assertTrue((sizes.merge(d[['fold','variant','n_test']],on=['fold','variant'],validate='one_to_one').eval('n == n_test')).all())

    def test_fit_ceiling(self):
        fits=pd.read_csv(RES/'fit_inventory.csv')
        self.assertEqual(len(fits),25)
        self.assertEqual(len(fits.drop_duplicates(['study','target_slug','fold','variant'])),25)
        self.assertLess(fits.elapsed_seconds.sum(),3*60*60)

if __name__=='__main__':unittest.main()
