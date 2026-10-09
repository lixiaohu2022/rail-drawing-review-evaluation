"""评测器的反例测试；防止重复意见、缺失预测、虚假证据获得不当高分。"""
import copy
import unittest
import demo


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        cases, gold = demo.make_dataset()
        self.cases = [c for c in cases if c["split"] == "test"]
        ids = {c["case_id"] for c in self.cases}
        self.gold = [g for g in gold if g["case_id"] in ids]
        self.pred = [demo.review(c, "evidence") for c in self.cases]

    def score(self, pred):
        return demo.evaluate(self.cases, self.gold, pred)

    def test_split_by_family(self):
        cases, labels = demo.make_dataset()
        splits = {c["case_id"]: c["split"] for c in cases}
        dev = {g["family_id"] for g in labels if splits[g["case_id"]] == "dev"}
        test = {g["family_id"] for g in labels if splits[g["case_id"]] == "test"}
        self.assertFalse(dev & test)

    def test_supported_baseline(self):
        result = self.score(self.pred)
        self.assertTrue(result["demo_gate_pass"])
        self.assertEqual(result["metrics"]["tp"], 24)
        self.assertEqual(result["metrics"]["coverage"], 40 / 48)
        self.assertEqual(result["production_acceptance"], "NOT_ESTABLISHED")

    def test_empty_prediction_denominator(self):
        m = self.score([])["metrics"]
        self.assertEqual(m["missing_predictions"], 48)
        self.assertEqual(m["fn"], 24)
        self.assertEqual(m["decision_accuracy"], 0)
        self.assertIsNone(m["precision"])

    def test_duplicate_case_rejected(self):
        with self.assertRaises(ValueError):
            self.score(self.pred + [self.pred[0]])

    def test_unknown_case_rejected(self):
        with self.assertRaises(ValueError):
            self.score([{ "case_id": "unknown", "status": "complete", "findings": []}])

    def test_duplicate_finding_is_false_positive(self):
        pred = copy.deepcopy(self.pred)
        p = next(p for p in pred if p["findings"])
        p["findings"].append(copy.deepcopy(p["findings"][0]))
        self.assertEqual(self.score(pred)["metrics"]["fp"], 1)

    def test_existing_but_irrelevant_evidence(self):
        pred = copy.deepcopy(self.pred)
        p = next(p for p in pred if p["findings"] and p["findings"][0]["rule_id"] == demo.RULE_DIM)
        p["findings"][0]["evidence_ids"] = ["layer-1"]
        m = self.score(pred)["metrics"]
        self.assertEqual(m["tp"], 24)
        self.assertEqual(m["evidence_supported_recall"], 23 / 24)
        self.assertFalse(self.score(pred)["demo_gate_pass"])

    def test_nonexistent_evidence_not_supported(self):
        pred = copy.deepcopy(self.pred)
        next(p for p in pred if p["findings"])["findings"][0]["evidence_ids"] = ["invented"]
        self.assertLess(self.score(pred)["metrics"]["evidence_supported_recall"], 1)

    def test_abstain_all_is_not_success(self):
        pred = [{"case_id": c["case_id"], "status": "abstain", "findings": []} for c in self.cases]
        result = self.score(pred)
        self.assertFalse(result["demo_gate_pass"])
        self.assertEqual(result["metrics"]["recall"], 0)
        self.assertEqual(result["metrics"]["unknown_abstain_accuracy"], 1)

    def test_abstain_with_findings_rejected(self):
        pred = copy.deepcopy(self.pred)
        next(p for p in pred if p["findings"])["status"] = "abstain"
        with self.assertRaises(ValueError):
            self.score(pred)

    def test_naive_errors_are_visible(self):
        result = self.score([demo.review(c, "naive") for c in self.cases])
        self.assertFalse(result["demo_gate_pass"])
        self.assertGreater(result["metrics"]["normal_false_positive_rate"], 0)
        self.assertEqual(result["metrics"]["unknown_abstain_accuracy"], 0)

    def test_units_and_missing_inputs(self):
        case = copy.deepcopy(self.cases[0])
        case["dimension"]["value"] /= 10
        case["dimension"]["unit"] = "cm"
        self.assertFalse(demo.review(case, "evidence")["findings"])
        case["dimension"]["unit"] = "unsupported"
        self.assertEqual(demo.review(case, "evidence")["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
