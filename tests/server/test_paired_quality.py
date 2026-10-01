"""Exact paired statistics, identity evidence and complete versus partial coverage."""

import math

import pytest

from core.model_comparator import mcnemar_test
from core.quality_evaluator import QualityEvaluator, QualityTestConfig
from evaluators.ceval_evaluator import CEvalEvaluator
from server.checkpoints import JobJournal
from server.paired_quality import PairingConflict, adjust_family, compare_dataset
from server.settings import Endpoint
from server.store import JobStore


def dataset(outcomes=(True, False), **updates):
    config = QualityTestConfig(use_cache=False).to_dict()
    config.update(
        scoring_contract="a" * 64,
        dataset_provenance={
            "sample_sha256": "b" * 64,
            "few_shot_sha256": "c" * 64,
            "selection_seed": 42,
            "evaluation_split": "test",
            "few_shot_split": "dev",
        },
    )
    return {
        "total_samples": len(outcomes),
        "accuracy": 0.999,
        "config": config,
        "details": [
            {
                "sample_id": str(i),
                "question": f"q{i}",
                "correct_answer": "a",
                "prompt": f"prompt{i}",
                "is_correct": outcome,
            }
            for i, outcome in enumerate(outcomes)
        ],
        **updates,
    }


@pytest.mark.parametrize(
    "n,k", [(1, 0), (2, 0), (5, 0), (6, 0), (10, 1), (20, 3), (20, 10), (50, 11)]
)
def test_exact_binomial_matches_integer_combinatorial_reference(n, k):
    result = mcnemar_test([True] * k + [False] * (n - k), [False] * k + [True] * (n - k))
    expected = min(1, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    assert result["p_value"] == pytest.approx(expected, rel=1e-12)
    assert result["method"] == "mcnemar-exact-binomial-two-sided-v1"
    assert result["b01_count"] == k and result["b10_count"] == n - k


def test_identical_and_large_discordant_grades_have_complete_and_finite_results():
    equal = mcnemar_test([True, False], [True, False])
    assert equal["p_value"] == 1 and equal["both_correct"] == equal["both_wrong"] == 1
    assert equal["b01_count"] == equal["b10_count"] == 0
    extreme = mcnemar_test([True] * 2000, [False] * 2000)
    assert extreme["p_value"] > 0 and extreme["p_value_label"] == "<5e-324"
    assert extreme["log_p_value"] == pytest.approx((1 - 2000) * math.log(2))


def test_accuracies_use_valid_paired_denominator_and_show_missing_coverage():
    a, b = dataset((True, False, True)), dataset((False, True))
    result = compare_dataset(a, b)
    assert result["accuracy_a"] == result["accuracy_b"] == 0.5
    assert result["samples"] == 2 and result["unpaired_a"] == 1
    assert not result["verified"] and result["p_value"] is None and result["significant"] is None


@pytest.mark.parametrize("field", ["question", "correct_answer"])
def test_same_id_with_changed_identity_is_rejected(field):
    a, b = dataset(), dataset()
    b["details"][0][field] = "different"
    with pytest.raises(PairingConflict, match=field):
        compare_dataset(a, b)


def test_duplicate_ids_cannot_silently_overwrite_grades():
    a = dataset()
    a["details"][1]["sample_id"] = "0"
    with pytest.raises(PairingConflict, match="duplicate"):
        compare_dataset(a, dataset())


@pytest.mark.parametrize(
    "reason", ["prompt", "few_shot", "scoring", "temperature", "missing_conditions"]
)
def test_unverified_conditions_keep_description_but_never_significance(reason):
    a, b = dataset(), dataset()
    if reason == "prompt":
        b["details"][0]["prompt"] = "changed"
    elif reason == "few_shot":
        b["config"]["dataset_provenance"]["few_shot_sha256"] = "d" * 64
    elif reason == "scoring":
        b["config"].pop("scoring_contract")
    elif reason == "temperature":
        b["config"]["temperature"] = 1
    else:
        b["config"] = {}
    result = compare_dataset(a, b)
    assert result["p_value"] is None and result["warnings"] and result["samples"] == 2
    assert result["method"] == "descriptive-pairs-v1" and result["statistic"] is None


def test_rule_and_final_scores_distinguish_judge_corrections():
    a, b = dataset(), dataset()
    a["details"][0].update(is_judge_corrected=True, judge_verdict="YES")
    rule = compare_dataset(a, b, "standard")
    final = compare_dataset(a, b, "final")
    assert rule["accuracy_a"] == 0 and final["accuracy_a"] == 0.5
    assert not rule["verified"] and not final["verified"]
    assert rule["paired_sha256"] == final["paired_sha256"]


def test_strings_invalid_grades_and_request_errors_do_not_become_true_grades():
    a, b = dataset((True, False, True, True)), dataset((False, True, False, True))
    a["details"][0]["is_correct"] = "false"
    a["details"][1]["error"] = "request failed"
    a["details"][2].pop("sample_id")
    result = compare_dataset(a, b)
    assert result["samples"] == 1 and result["excluded_a"] == {
        "invalid_grade": 1,
        "request_error": 1,
        "missing_id": 1,
    }
    assert result["p_value"] is None


def test_complete_verified_grades_get_exact_test_and_same_pair_fingerprint_after_reordering():
    a, b = dataset((True,) * 6), dataset((False,) * 6)
    result = compare_dataset(a, b)
    assert result["verified"] and result["p_value"] == pytest.approx(0.03125)
    b["details"].reverse()
    assert compare_dataset(a, b)["paired_sha256"] == result["paired_sha256"]


def test_ceval_split_is_checked_only_for_ceval_and_effective_overrides_are_compared():
    a, b = dataset(), dataset()
    b["config"]["ceval_split"] = "val"
    assert compare_dataset(a, b, dataset_name="gsm8k")["verified"]
    assert not compare_dataset(a, b, dataset_name="ceval")["verified"]
    a["config"]["temperature"] = 1
    a["config"]["dataset_overrides"] = {"gsm8k": {"temperature": 0}}
    assert compare_dataset(a, b, dataset_name="gsm8k")["verified"]


def test_holm_is_step_down_and_never_implies_unverified_groups_are_non_significant():
    family = {
        name: {"verified": True, "log_p_value": math.log(p)}
        for name, p in zip(("a", "b", "c"), (0.01, 0.04, 0.2), strict=True)
    }
    family["legacy"] = {"verified": False}
    adjust_family(family)
    assert [family[n]["adjusted_p_value"] for n in ("a", "b", "c")] == pytest.approx(
        [0.03, 0.08, 0.2]
    )
    assert (
        family["a"]["adjusted_significant"] is True and family["b"]["adjusted_significant"] is False
    )
    assert family["legacy"]["adjusted_significant"] is None and family["a"]["test_family_size"] == 3


@pytest.mark.parametrize("field,value", [("details", None), ("config", "invalid")])
def test_malformed_report_has_actionable_conflict(field, value):
    a = dataset()
    a[field] = value
    with pytest.raises(PairingConflict):
        compare_dataset(a, dataset())


def test_core_statistics_do_not_coerce_boolean_strings():
    assert "error" in mcnemar_test(["false"], [False])
    assert "error" in mcnemar_test([], [])


def test_code_grading_needs_shared_sandbox_identity_for_inference():
    from tests.sandbox_fixtures import sandbox_identity

    a, b = dataset(), dataset()
    a["config"]["requires_code_execution"] = b["config"]["requires_code_execution"] = True
    assert compare_dataset(a, b)["p_value"] is None
    a["config"]["sandbox_contract"] = b["config"]["sandbox_contract"] = "d" * 64
    assert not compare_dataset(a, b)["verified"]
    proof = sandbox_identity()
    for side in (a, b):
        side["config"].update(sandbox_identity=proof, sandbox_contract=proof["sha256"])
    assert compare_dataset(a, b)["verified"]


def test_scoring_source_is_frozen_in_real_sample_journal_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(QualityEvaluator, "_init_tokenizer", lambda *args: None)
    monkeypatch.setattr("core.quality_evaluator.scoring_fingerprint", lambda evaluator: "a" * 64)
    store = JobStore(tmp_path / "isolated.db")
    store.submit(test_type="quality", endpoint_id="lab", model_id="synthetic", parameters={})
    job = store.claim("fixture")
    endpoint = Endpoint(
        "lab", "Synthetic", "OpenAI", "http://127.0.0.1:9/v1", "synthetic", "UNUSED"
    )
    journal = JobJournal(store, job, "fixture", endpoint)
    engine = QualityEvaluator(
        api_base_url=endpoint.api_base_url,
        model_id="synthetic",
        output_dir=str(tmp_path / "artifacts"),
        enable_cache=False,
        journal=journal,
    )
    evaluator = CEvalEvaluator(num_shots=0)
    evaluator.load_dataset = lambda subset=None: [{"id": "one", "question": "original"}]
    original = engine._load_plan(evaluator, None, "quality:ceval")
    monkeypatch.setattr("core.quality_evaluator.scoring_fingerprint", lambda evaluator: "b" * 64)
    changed = CEvalEvaluator(num_shots=0)
    changed.load_dataset = lambda subset=None: [{"id": "one", "question": "changed"}]
    assert engine._load_plan(changed, None, "quality:ceval") == original
    assert evaluator.scoring_contract == changed.scoring_contract == "a" * 64
