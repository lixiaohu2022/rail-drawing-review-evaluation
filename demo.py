#!/usr/bin/env python3
"""事实源＝本脚本（数据、预测及报告由脚本生成）。标准库、离线合成评测。"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

SCENARIOS = ("clean", "dimension_error", "mixed_units", "unit_error", "layer_error", "missing_evidence")
RULE_DIM = "DEMO_DIM"
RULE_LAYER = "DEMO_LAYER"
FACTORS = {"mm": 1.0, "cm": 10.0}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_dataset():
    """真值按注错计划写入，未调用审查基线；同一图纸族不跨数据划分。"""
    cases, gold = [], []
    for family in range(16):
        width = 1800 + family * 170
        split = "dev" if family < 8 else "test"
        for index, scenario in enumerate(SCENARIOS):
            cid = f"case-{family * 6 + index + 1:03d}"
            dimension = {"id": "dimension-1", "value": width, "unit": "mm"}
            layer = "DEMO_COMPONENT"
            findings = []
            status = "complete"
            if scenario == "dimension_error":
                dimension["value"] += 100
            elif scenario == "mixed_units":
                dimension.update(value=width / 10, unit="cm")
            elif scenario == "unit_error":
                dimension["unit"] = "cm"
            elif scenario == "layer_error":
                layer = "DEMO_ANNOTATION"
            elif scenario == "missing_evidence":
                dimension = None
                status = "abstain"
            if scenario in ("dimension_error", "unit_error"):
                findings.append({"rule_id": RULE_DIM, "entity_id": "component-1", "evidence_ids": ["geometry-1", "dimension-1"]})
            if scenario == "layer_error":
                findings.append({"rule_id": RULE_LAYER, "entity_id": "component-1", "evidence_ids": ["layer-1"]})
            case = {"case_id": cid, "split": split, "entity_id": "component-1",
                    "geometry": {"id": "geometry-1", "x0_mm": 0, "x1_mm": width},
                    "dimension": dimension, "layer": {"id": "layer-1", "name": layer}}
            cases.append(case)
            gold.append({"case_id": cid, "family_id": f"family-{family:02d}",
                         "scenario": scenario, "status": status, "findings": findings})
    return cases, gold


def drawing_svg(case):
    dim = case["dimension"]
    label = "Evidence unavailable" if dim is None else f'{dim["value"]} {dim["unit"]}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="300" viewBox="0 0 720 300">
<rect width="720" height="300" fill="#f8fafc"/>
<g font-family="sans-serif" fill="#172554">
<text x="30" y="35" font-size="20">{case["case_id"]} — synthetic drawing</text>
<rect x="90" y="100" width="540" height="90" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
<path d="M90 215 H630 M90 200 V230 M630 200 V230" stroke="#475569"/>
<text x="360" y="245" text-anchor="middle" font-size="18">{escape(label)}</text>
<text x="110" y="150" font-size="17">component-1</text>
<text x="30" y="280" font-size="13">Layer: {escape(case["layer"]["name"])} · not to scale · illustrative only</text>
</g></svg>\n'''


def review(case, mode):
    """只接收可见输入；不接收真值、注错类别或图纸族。"""
    dim = case["dimension"]
    result = {"case_id": case["case_id"], "status": "complete", "findings": []}
    if mode == "evidence" and (dim is None or dim["unit"] not in FACTORS):
        result.update(status="abstain", reason="尺寸证据缺失或单位不在支持范围。")
        return result
    if dim is not None:
        measured = dim["value"] * (FACTORS[dim["unit"]] if mode == "evidence" else 1.0)
        width = case["geometry"]["x1_mm"] - case["geometry"]["x0_mm"]
        if abs(measured - width) > 5:
            result["findings"].append({"rule_id": RULE_DIM, "entity_id": case["entity_id"],
                                       "evidence_ids": ["geometry-1", "dimension-1"]})
    if mode == "evidence" and case["layer"]["name"] != "DEMO_COMPONENT":
        result["findings"].append({"rule_id": RULE_LAYER, "entity_id": case["entity_id"], "evidence_ids": ["layer-1"]})
    return result


def ratio(a, b):
    return a / b if b else None


def wilson(successes, total):
    """案例决策准确率的Wilson 95%区间；相关图纸变体不满足独立性假设。"""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - half, center + half]


def evaluate(cases, gold, predictions):
    inputs = {c["case_id"]: c for c in cases}
    labels = {g["case_id"]: g for g in gold}
    if len(inputs) != len(cases) or len(labels) != len(gold) or set(inputs) != set(labels):
        raise ValueError("输入与真值须唯一且完全对应。")
    predicted = {}
    for p in predictions:
        cid = p["case_id"]
        if cid not in inputs or cid in predicted:
            raise ValueError("预测包含未知或重复案例。")
        if p["status"] not in ("complete", "abstain"):
            raise ValueError("status须为complete或abstain。")
        if p["status"] == "abstain" and p["findings"]:
            raise ValueError("此协议中abstain不得同时输出审查意见。")
        for f in p["findings"]:
            if not all(isinstance(f.get(k), str) for k in ("rule_id", "entity_id")):
                raise ValueError("意见须包含字符串rule_id和entity_id。")
            ids = f.get("evidence_ids")
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
                raise ValueError("evidence_ids须为不重复的字符串列表。")
        predicted[cid] = p
    tp = fp = fn = supported_tp = correct = covered = normal_fp_cases = 0
    normal_count = unknown_count = unknown_correct = missing = 0
    details = []
    strata = {}
    for cid, case in inputs.items():
        g = labels[cid]
        p = predicted.get(cid, {"status": "missing", "findings": []})
        missing += p["status"] == "missing"
        covered += p["status"] == "complete"
        key = lambda f: (f["rule_id"], f["entity_id"])
        expected = Counter(key(f) for f in g["findings"])
        actual = Counter(key(f) for f in p["findings"])
        match = sum((expected & actual).values())
        tp += match
        fp += sum((actual - expected).values())
        fn += sum((expected - actual).values())
        remaining = expected.copy()
        supported = 0
        valid_ids = {v["id"] for v in (case["geometry"], case["dimension"], case["layer"]) if v is not None}
        for f in p["findings"]:
            k = key(f)
            if remaining[k] > 0:
                remaining[k] -= 1
                target = next(x for x in g["findings"] if key(x) == k)
                evidence = set(f["evidence_ids"])
                supported += evidence == set(target["evidence_ids"]) and evidence <= valid_ids
        supported_tp += supported
        passed = p["status"] == g["status"] and actual == expected and supported == sum(expected.values())
        correct += passed
        if g["status"] == "abstain":
            unknown_count += 1
            unknown_correct += p["status"] == "abstain"
        elif not g["findings"]:
            normal_count += 1
            normal_fp_cases += bool(p["findings"])
        details.append({"case_id": cid, "scenario": g["scenario"], "correct": passed,
                        "tp": match, "fp": sum((actual - expected).values()),
                        "fn": sum((expected - actual).values()), "supported_tp": supported})
        group = strata.setdefault(g["scenario"], {"cases": 0, "correct": 0})
        group["cases"] += 1
        group["correct"] += passed
    metrics = {"cases": len(cases), "missing_predictions": missing, "tp": tp, "fp": fp, "fn": fn,
               "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "evidence_supported_precision": ratio(supported_tp, tp + fp),
               "matched_findings_without_required_evidence": tp - supported_tp,
               "evidence_supported_recall": ratio(supported_tp, tp + fn),
               "decision_accuracy": ratio(correct, len(cases)), "decision_correct": correct,
               "decision_accuracy_wilson95_illustrative": wilson(correct, len(cases)),
               "coverage": ratio(covered, len(cases)), "normal_cases": normal_count,
               "normal_false_positive_cases": normal_fp_cases,
               "normal_false_positive_rate": ratio(normal_fp_cases, normal_count),
               "unknown_cases": unknown_count, "unknown_abstain_accuracy": ratio(unknown_correct, unknown_count)}
    gate = (missing == 0 and all(metrics[k] is not None and metrics[k] >= .95
            for k in ("precision", "recall", "evidence_supported_recall"))
            and supported_tp == tp and normal_count > 0 and normal_fp_cases == 0
            and unknown_count > 0 and unknown_correct == unknown_count)
    return {"metrics": metrics, "strata": strata, "demo_gate_pass": gate,
            "production_acceptance": "NOT_ESTABLISHED", "details": details}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    cases, gold = make_dataset()
    save(output / "inputs.json", cases)
    save(output / "gold.json", gold)
    for case in cases:
        path = output / "drawings" / (case["case_id"] + ".svg")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(drawing_svg(case), encoding="utf-8")
    tests = [c for c in cases if c["split"] == "test"]
    ids = {c["case_id"] for c in tests}
    test_gold = [g for g in gold if g["case_id"] in ids]
    reports = {}
    for mode in ("naive", "evidence"):
        predictions = [review(c, mode) for c in tests]
        save(output / f"predictions_{mode}.json", predictions)
        reports[mode] = evaluate(tests, test_gold, predictions)
        save(output / f"report_{mode}.json", reports[mode])
    save(output / "reproducibility.json", {"generator_sha256": digest(Path(__file__)),
         "inputs_sha256": digest(output / "inputs.json"), "gold_sha256": digest(output / "gold.json"),
         "families": 16, "dev_families": 8, "test_families": 8, "samples": 96,
         "model_used": None, "note": "确定性基线；无OCR、CAD解析或大模型请求。"})
    print(json.dumps({k: {"metrics": v["metrics"], "demo_gate_pass": v["demo_gate_pass"]} for k, v in reports.items()}, ensure_ascii=False, indent=2))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("run")
    demo.add_argument("--output", type=Path, default=Path("build"))
    scorer = sub.add_parser("evaluate")
    scorer.add_argument("--inputs", type=Path, required=True)
    scorer.add_argument("--gold", type=Path, required=True)
    scorer.add_argument("--predictions", type=Path, required=True)
    scorer.add_argument("--split", choices=("dev", "test"), default="test")
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    else:
        load_json = lambda p: json.loads(p.read_text(encoding="utf-8"))
        cases = [c for c in load_json(args.inputs) if c["split"] == args.split]
        ids = {c["case_id"] for c in cases}
        if not cases:
            raise ValueError("划分不能为空。")
        report = evaluate(cases, [g for g in load_json(args.gold) if g["case_id"] in ids], load_json(args.predictions))
        save(args.output, report)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
