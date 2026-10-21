"""Focused, fit-free consistency checks for recovered historical evidence."""
from pathlib import Path
import math
import json
import unittest

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


class AuditOutputsTest(unittest.TestCase):
    def test_sources_and_fold_metrics(self):
        checks = table("recovered_file_checksums.csv")
        self.assertEqual(len(checks), 43)
        self.assertTrue(checks.status.eq("PASS").all())
        for name, n in (("original_random_fold_reconciliation.csv", 5),
                        ("original_topology_fold_reconciliation.csv", 20)):
            folds = table(name)
            self.assertEqual(len(folds), n)
            self.assertTrue(folds.n.eq(folds.saved_n).all())
            self.assertLess(folds.max_metric_abs_difference.max(), 1e-12)

    def test_geometry_membership(self):
        geo = table("geometry_fold_integrity.csv")
        self.assertEqual(geo.test_n.sum(), 185113)
        self.assertTrue(geo.group_overlap.eq(0).all())
        self.assertTrue(geo.test_n.eq(geo.saved_metric_test_n).all())
        sources = table("group_source_alignment.csv")
        self.assertTrue(sources.label_mismatches.eq(0).all())
        self.assertEqual(int(sources.loc[sources.group_source.eq("geo-clusters.csv"), "matched_ids"].iloc[0]), 263744)
        summary = json.loads((RESULTS / "geometry_group_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["singleton_rows_excluded"], 78631)
        self.assertEqual(summary["eligible_rows"] + summary["singleton_rows_excluded"],
                         summary["evaluated_master_rows"])

    def test_original_random_archive_and_top_lists(self):
        multiplicity = table("original_random_multiplicity.csv")
        self.assertEqual(multiplicity.structures.sum(), 177411)
        self.assertEqual((multiplicity.oos_predictions_per_mof * multiplicity.structures).sum(), 263745)
        archive = table("original_random_archive_reconciliation.csv").iloc[0]
        self.assertEqual(archive.multiplicity_mismatches, 0)
        self.assertEqual(archive.fold_token_mismatches, 0)
        self.assertLess(archive.original_vs_archive_max_prediction_abs_difference, 1e-12)
        sampled = table("original_random_sampling_sensitivity.csv")
        self.assertEqual(len(sampled), 5)
        self.assertTrue(sampled.population_n.eq(177411).all())
        self.assertTrue(sampled.k.eq(8871).all())
        self.assertTrue(sampled.changed_shortlist_memberships_symmetric_difference.mod(2).eq(0).all())
        lists = table("original_random_topology_shortlist.csv").iloc[0]
        self.assertEqual(lists.common_shortlist_n + lists.random_only_n, lists.k)
        self.assertEqual(lists.common_shortlist_n + lists.topology_only_n, lists.k)
        self.assertAlmostEqual(lists.random_topology_jaccard,
                               lists.common_shortlist_n / (2 * lists.k - lists.common_shortlist_n))
        self.assertTrue(table("v6_shortlist_reconciliation.csv").status.eq("MATCH").all())

    def test_exact_pressure_population_and_normalization(self):
        coverage = table("pressure_id_coverage.csv")
        self.assertTrue(coverage.rows.eq(263744).all())
        self.assertTrue(coverage.unique_ids.eq(263744).all())
        family = table("pressure_family_exact.csv")
        iqrs = table("pressure_target_iqr_exact.csv").set_index("target_slug")
        self.assertEqual(len(family), 188)
        for slug, part in family.groupby("target_slug"):
            self.assertEqual(len(part), 47)
            self.assertEqual(part.n.sum(), 263744)
            self.assertTrue(np.allclose(part.mae_over_target_iqr,
                                        part.mae_mmol_g / iqrs.loc[slug, "iqr_mmol_g"], atol=1e-12))
        paired = table("pressure_pairwise_error_persistence.csv")
        self.assertEqual(len(paired), math.comb(4, 2))
        self.assertTrue(paired.paired_mofs.eq(263744).all())
        self.assertTrue(paired.high_error_top10_k_each.eq(26375).all())
        high = table("pressure_high_error_multiplicity.csv")
        self.assertEqual(high.structures.sum(), 263744)
        self.assertEqual((high.targets_in_global_top_error_decile * high.structures).sum(), 4 * 26375)


if __name__ == "__main__":
    unittest.main()
