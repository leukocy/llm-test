"""Offline tool behavior, denominator integrity and authenticated boundaries."""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from core.robustness_tester import PerturbationType, TextPerturber
from server.advanced import ConsistencyBody, ParseBody, analyze_consistency, parse_answer
from server.api import create_app
from server.settings import Settings
from server.store import JobStore

TOKEN = "a" * 48
DATASETS = ["auto", "mmlu", "gsm8k", "math500", "humaneval", "gpqa", "truthfulqa", "longbench"]
KINDS = [
    "synonym",
    "typo",
    "reorder",
    "case",
    "punctuation",
    "whitespace",
    "number_format",
    "paraphrase",
    "context_add",
    "rephrase",
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    from core.database.connection import Database
    from core.database.manager import DatabaseManager

    Database._instance = None
    DatabaseManager._instance = None
    db_path = tmp_path / "advanced.db"
    monkeypatch.setenv("LLM_TEST_DB_PATH", str(db_path))
    settings = Settings(
        api_token=TOKEN, db_path=db_path, artifact_root=tmp_path / "artifacts", endpoints={}
    )
    store = JobStore(db_path)
    with TestClient(create_app(settings, store)) as client:
        yield client, store
    Database._instance = None
    DatabaseManager._instance = None


def post(client, path, body):
    response = client.post(
        "/api/v1/advanced/" + path, json=body, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_catalog_and_authentication(env):
    client, _ = env
    response = client.get("/api/v1/advanced/catalog", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["datasets"] == DATASETS
    assert [item["id"] for item in catalog["perturbations"]] == KINDS
    assert {item["id"] for item in catalog["perturbations"] if not item["supported"]} == {
        "reorder",
        "paraphrase",
        "rephrase",
    }
    for path in ["parse", "perturb", "consistency", "reasoning"]:
        response = client.post("/api/v1/advanced/" + path, json={})
        assert response.status_code == 401
    assert client.get("/api/v1/advanced/catalog").status_code == 401


@pytest.mark.parametrize(
    ("dataset", "response", "expected", "parser"),
    [
        ("auto", r"The answer is \boxed{42}.", "42", "MathAnswerParser"),
        ("mmlu", "The answer is B.", "B", "MultiChoiceParser"),
        ("gpqa", "The answer is B.", "B", "MultiChoiceParser"),
        ("truthfulqa", "The answer is B.", "B", "MultiChoiceParser"),
        ("gsm8k", "Working...\n#### 42", "42", "MathAnswerParser"),
        ("math500", r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", "MathAnswerParser"),
        (
            "humaneval",
            "Here:\n```python\ndef solve():\n    return 42\n```",
            "def solve():\n    return 42",
            "CodeAnswerParser",
        ),
        (
            "longbench",
            "  A long answer with punctuation!  ",
            "A long answer with punctuation!",
            "TextAnswerParser",
        ),
    ],
)
def test_all_original_dataset_modes_extract_without_reference_hints(
    env, dataset, response, expected, parser
):
    client, _ = env
    base = {"response": response, "dataset_type": dataset}
    original = post(client, "parse", base)
    with_reference = post(client, "parse", {**base, "expected_answer": "wrong reference"})
    assert original["extracted_answer"] == expected
    assert original["method"] == parser
    assert original["extracted_answer"] == with_reference["extracted_answer"]
    assert original["confidence"] is None


@pytest.mark.parametrize("dataset", ["auto", "gsm8k", "math500"])
def test_bare_fractions_and_percent_keep_their_value(dataset):
    result = parse_answer(ParseBody(response="1/2", dataset_type=dataset, expected_answer="0.5"))
    assert result["extracted_answer"] == "1/2"
    assert result["is_correct"] is True
    assert result["normalized_value"] == "1/2"


@pytest.mark.parametrize(
    ("kind", "response", "reference", "correct"),
    [
        ("number", "The answer is 0.4", "0.49", False),
        ("number", "0.5", "50", False),
        ("math", "50%", "0.5", True),
        ("math", "sin(2)", "2", None),
        ("math", "1/0", "0", None),
        ("math", r"\boxed{__import__('os').getcwd()}", "0", None),
        ("boolean", "Unclear", "no", None),
        ("boolean", "Yes", "true", True),
        ("text", "not 42", "42", False),
        ("text", "A B!", "a   b!", True),
        ("text", "A B!", "A B", False),
        (
            "code",
            "```python\nraise RuntimeError('never execute')\n```",
            "raise RuntimeError('never execute')",
            True,
        ),
    ],
)
def test_type_comparison_is_strict_and_unknown_is_preserved(
    env, kind, response, reference, correct
):
    client, _ = env
    result = post(
        client, "parse", {"response": response, "answer_type": kind, "expected_answer": reference}
    )
    assert result["is_correct"] is correct
    assert result["score"] == (None if correct is None else float(correct))


def test_snapshots_bind_full_input_and_results_do_not_create_jobs(env):
    client, store = env
    body = {"text": "Buy 1000 apples", "perturbation_type": "number_format", "seed": 42}
    first = post(client, "perturb", body)
    second = post(client, "perturb", body)
    encoded = json.dumps(
        first["input"], sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()
    assert first == second
    assert first["input"] == body
    assert first["tool_version"] == "offline-tools-v1"
    assert first["input_sha256"] == hashlib.sha256(encoded).hexdigest()
    jobs = client.get("/api/v1/jobs", headers={"Authorization": f"Bearer {TOKEN}"})
    assert jobs.status_code == 200
    assert jobs.json()["items"] == []


def consistency(responses, *, reference="42", threshold=0.8, dataset="auto"):
    return analyze_consistency(
        ConsistencyBody(
            question="15 + 27?",
            runs=len(responses),
            threshold=threshold,
            dataset_type=dataset,
            expected_answer=reference,
            responses=responses,
        )
    )


def test_consistency_uses_all_attempts_and_normalized_groups():
    result = consistency(["42", r"\boxed{42}", "42.0", "43", ""])
    assert result["consistency_rate"] == 3 / 5
    assert result["parsed"] == 4 and result["unparsed"] == 1
    assert result["stable"] is False
    assert result["reference_match_rate"] == 3 / 5
    assert result["reference_undetermined"] == 1
    assert result["groups"][0]["count"] == 3


def test_consistency_does_not_confuse_agreement_with_correctness():
    result = consistency(["43"] * 5)
    assert result["stable"] is True
    assert result["consistency_rate"] == 1
    assert result["reference_match_rate"] == 0
    fractional = consistency(["1/2", "0.5", "50%"], reference="0.5")
    assert fractional["consistency_rate"] == 1
    assert len(fractional["groups"]) == 1


def test_ties_and_unparsed_answers_do_not_become_stable():
    tied = consistency(["42", "43"], threshold=0.5)
    assert tied["tied_majority"] is True and tied["stable"] is False
    absent = consistency(["", "   "])
    assert absent["parsed"] == 0 and absent["unparsed"] == 2
    assert absent["consistency_rate"] is None and absent["stable"] is None
    invalid = consistency(["sin(2)", "1/0"], dataset="math500")
    assert invalid["stable"] is None and invalid["parsed"] == 0
    assert all(item["error"] for item in invalid["results"])


@pytest.mark.parametrize("kind", KINDS)
def test_perturbation_capabilities_report_actual_change(env, kind):
    client, _ = env
    result = post(
        client, "perturb", {"text": "How to Calculate 1000 apples?", "perturbation_type": kind}
    )
    unsupported = kind in {"reorder", "paraphrase", "rephrase"}
    assert result["supported"] is not unsupported
    assert result["changed"] is (result["original_text"] != result["perturbed_text"])
    if unsupported:
        assert result["status"] == "unsupported" and result["changed"] is False
    else:
        assert result["status"] == ("applied" if result["changed"] else "unchanged")


def test_no_matching_synonym_is_unchanged_and_typo_keeps_other_factors(env):
    client, _ = env
    result = post(client, "perturb", {"text": "calculations", "perturbation_type": "synonym"})
    assert result["status"] == "unchanged" and result["details"] == "No synonyms found"
    source = "Hello\t world\n  apples"
    result = TextPerturber(42).perturb(source, PerturbationType.TYPO_INSERT)
    assert [char for char in source if char.isspace()] == [
        char for char in result.perturbed_question if char.isspace()
    ]


def test_rng_isolated_and_concurrent_requests_reproducible():
    before = random.getstate()

    def generate(seed):
        return (
            TextPerturber(seed)
            .perturb("Please calculate the big number", PerturbationType.CONTEXT_ADD)
            .perturbed_question
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(generate, [42, 7, 42, 7, 42, 7]))
    assert random.getstate() == before
    assert outputs[0] == outputs[2] == outputs[4]
    assert outputs[1] == outputs[3] == outputs[5]


def test_number_format_preserves_decimal_sign_leading_zero_and_exponent():
    source = "1000.5000 -2000.25 +3,000.75 01234 1e100 2E+8"
    result = TextPerturber(42).perturb(source, PerturbationType.NUMBER_FORMAT)
    assert result.perturbed_question == "1,000.5000 -2,000.25 +3000.75 01234 1e100 2E+8"
    large = TextPerturber(42).perturb("9" * 5000, PerturbationType.NUMBER_FORMAT)
    assert large.perturbed_question.replace(",", "") == "9" * 5000


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("parse", {"response": " ", "dataset_type": "auto"}),
        ("parse", {"response": "42", "dataset_type": "auto", "answer_type": "number"}),
        ("parse", {"response": "42", "dataset_type": "unknown"}),
        ("parse", {"response": "42", "file": "/tmp/data"}),
        ("perturb", {"text": "hello", "perturbation_type": "case", "seed": -1}),
        ("perturb", {"text": "hello", "perturbation_type": "case", "seed": 2**32}),
        ("perturb", {"text": " ", "perturbation_type": "case"}),
        ("consistency", {"question": "test", "runs": 5, "responses": ["42", "42"]}),
        (
            "consistency",
            {"question": "test", "runs": 2, "threshold": 0.49, "responses": ["42", "42"]},
        ),
        (
            "consistency",
            {"question": "test", "runs": 2, "threshold": "NaN", "responses": ["42", "42"]},
        ),
        ("consistency", {"question": "test", "runs": 10, "responses": ["x" * 10000] * 10}),
        ("reasoning", {"question": " "}),
    ],
)
def test_invalid_bounds_and_extra_fields_are_rejected(env, path, body):
    client, _ = env
    response = client.post(
        "/api/v1/advanced/" + path, json=body, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 422


def test_reasoning_reference_missing_is_unknown_and_numeric_grading_strict(env):
    client, _ = env
    missing = post(
        client,
        "reasoning",
        {"question": "test", "final_answer": "42", "reasoning": "First solve. Then answer."},
    )
    assert missing["final_answer_correct"] is None
    assert missing["quality_score"]["correctness"] is None
    assert missing["quality_score"]["overall"] is None
    assert missing["failure_analysis"] == ""
    incorrect = post(
        client, "reasoning", {"question": "test", "final_answer": "0.5", "correct_answer": "50"}
    )
    assert incorrect["final_answer_correct"] is False


def test_default_answer_type_remains_text(env):
    client, _ = env
    result = post(client, "parse", {"response": "Full response"})
    assert result["answer_type"] == "text"
    assert "is_correct" not in result
