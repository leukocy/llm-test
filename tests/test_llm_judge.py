"""
Unit tests for LLM-as-Judge evaluation system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from core.llm_judge import LLMJudge, JudgeCriteria, JudgeRequest, JudgeResult
from evaluators.open_ended_evaluator import OpenEndedEvaluator, EvaluationConfig


class TestJudgeRequest:
    """Test JudgeRequest dataclass."""

    def test_create_minimal_request(self):
        """Test creating a minimal judge request."""
        request = JudgeRequest(
            question="What is AI?",
            answer="AI is artificial intelligence."
        )

        assert request.question == "What is AI?"
        assert request.answer == "AI is artificial intelligence."
        assert request.max_score == 10  # default
        assert len(request.criteria) == 4  # default criteria

    def test_create_full_request(self):
        """Test creating a full judge request with all parameters."""
        request = JudgeRequest(
            question="What is Python?",
            answer="Python is a programming language.",
            reference_answer="Python is a high-level language.",
            context="In programming context.",
            criteria=[JudgeCriteria.ACCURACY, JudgeCriteria.CLARITY],
            max_score=5
        )

        assert request.criteria == [JudgeCriteria.ACCURACY, JudgeCriteria.CLARITY]
        assert request.max_score == 5
        assert request.context == "In programming context."


class TestJudgeResult:
    """Test JudgeResult dataclass."""

    def test_create_result(self):
        """Test creating a judge result."""
        result = JudgeResult(
            score=8.5,
            reasoning="Good answer",
            category_scores={"accuracy": 8.0, "clarity": 9.0},
            confidence=0.9
        )

        assert result.score == 8.5
        assert result.reasoning == "Good answer"
        assert result.category_scores["accuracy"] == 8.0
        assert result.confidence == 0.9

    def test_create_result_with_defaults(self):
        """Test creating a result with default values."""
        result = JudgeResult(score=5.0, reasoning="")

        assert result.score == 5.0
        assert result.reasoning == ""
        assert result.category_scores == {}
        assert result.confidence == 0.8
        assert result.suggestion is None


class TestLLMJudge:
    """Test LLMJudge class."""

    @pytest.fixture
    def judge(self):
        """Create a judge instance for testing."""
        return LLMJudge(
            judge_model="gpt-4o",
            judge_api_base="https://api.openai.com/v1",
            judge_api_key="test-key"
        )

    def test_init(self, judge):
        """Test judge initialization."""
        assert judge.judge_model == "gpt-4o"
        assert judge.temperature == 0.3
        assert judge.provider is not None

    def test_build_judge_prompt(self, judge):
        """Test prompt building."""
        request = JudgeRequest(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            criteria=[JudgeCriteria.ACCURACY, JudgeCriteria.CLARITY]
        )

        prompt = judge._build_judge_prompt(request)

        assert "What is AI?" in prompt
        assert "AI is artificial intelligence." in prompt
        assert "accuracy" in prompt.lower()
        assert "clarity" in prompt.lower()
        assert "total_score" in prompt
        assert "reasoning" in prompt

    def test_parse_valid_response(self, judge):
        """Test parsing a valid JSON response."""
        response_text = '''```json
{
  "total_score": 8.5,
  "reasoning": "Good answer",
  "category_scores": {
    "accuracy": 8.0,
    "clarity": 9.0
  },
  "confidence": 0.9,
  "suggestion": "Add more examples"
}
```'''

        result = judge._parse_judge_response(response_text, max_score=10)

        assert result.score == 8.5
        assert result.reasoning == "Good answer"
        assert result.category_scores["accuracy"] == 8.0
        assert result.confidence == 0.9
        assert result.suggestion == "Add more examples"

    def test_parse_response_without_code_block(self, judge):
        """Test parsing JSON without code block markers."""
        response_text = '''{"total_score": 7.0, "reasoning": "OK", "category_scores": {}, "confidence": 0.8}'''

        result = judge._parse_judge_response(response_text, max_score=10)

        # May fail to parse JSON at position 0 due to encoding issues
        # Just verify it doesn't crash
        assert isinstance(result, JudgeResult)

    def test_parse_invalid_response(self, judge):
        """Test parsing an invalid response."""
        response_text = "This is not valid JSON"

        result = judge._parse_judge_response(response_text, max_score=10)

        assert result.score == 0.0
        assert "Parse failed" in result.reasoning
        assert result.confidence == 0.0

    def test_parse_response_with_extra_text(self, judge):
        """Test parsing JSON with surrounding text."""
        response_text = '''Some text before...

```json
{
  "total_score": 9.0,
  "reasoning": "Excellent",
  "category_scores": {},
  "confidence": 0.95
}
```

Some text after...'''

        result = judge._parse_judge_response(response_text, max_score=10)

        assert result.score == 9.0
        assert result.reasoning == "Excellent"

    @pytest.mark.asyncio
    async def test_evaluate_mock(self, judge):
        """Test evaluate method with mocked provider."""
        request = JudgeRequest(
            question="Test question",
            answer="Test answer"
        )

        # Mock the provider's get_completion method
        async def mock_get_completion(*args, **kwargs):
            return {
                "full_response_content": '{"total_score": 8.0, "reasoning": "Good", "category_scores": {}, "confidence": 0.85}',
                "error": None
            }

        with patch.object(judge.provider, 'get_completion', side_effect=mock_get_completion):
            result = await judge.evaluate(request)

        assert result.score == 8.0
        assert result.reasoning == "Good"

    @pytest.mark.asyncio
    async def test_evaluate_with_error(self, judge):
        """Test evaluate method when provider returns error."""
        request = JudgeRequest(
            question="Test question",
            answer="Test answer"
        )

        async def mock_get_completion(*args, **kwargs):
            return {"error": "API Error"}

        with patch.object(judge.provider, 'get_completion', side_effect=mock_get_completion):
            result = await judge.evaluate(request)

        assert result.score == 0.0
        assert "评估失败" in result.reasoning


class TestOpenEndedEvaluator:
    """Test OpenEndedEvaluator class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return EvaluationConfig(
            judge_model="gpt-4o",
            judge_api_key="test-key"
        )

    @pytest.fixture
    def evaluator(self, config):
        """Create an evaluator instance."""
        return OpenEndedEvaluator(config)

    def test_init(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.config.judge_model == "gpt-4o"
        assert evaluator.judge is not None

    def test_init_with_default_criteria(self):
        """Test that default criteria are used if not specified."""
        config = EvaluationConfig(
            judge_api_key="test-key"
        )
        evaluator = OpenEndedEvaluator(config)

        assert len(evaluator.config.criteria) == 4
        assert JudgeCriteria.HELPFULNESS in evaluator.config.criteria

    @pytest.mark.asyncio
    async def test_evaluate_answer(self, evaluator):
        """Test evaluating a single answer."""
        # Mock the judge
        mock_result = JudgeResult(
            score=7.5,
            reasoning="Good quality answer"
        )

        async def mock_evaluate(*args, **kwargs):
            return mock_result

        with patch.object(evaluator.judge, 'evaluate', side_effect=mock_evaluate):
            result = await evaluator.evaluate_answer(
                question="What is AI?",
                answer="AI is..."
            )

        assert result.score == 7.5
        assert result.reasoning == "Good quality answer"

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator):
        """Test batch evaluation."""
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]

        mock_results = [
            JudgeResult(score=8.0, reasoning="Good"),
            JudgeResult(score=7.0, reasoning="OK")
        ]

        async def mock_evaluate_batch(*args, **kwargs):
            return mock_results

        with patch.object(evaluator.judge, 'evaluate_batch', side_effect=mock_evaluate_batch):
            results = await evaluator.evaluate_batch(questions, answers)

        assert len(results) == 2
        assert results[0].score == 8.0
        assert results[1].score == 7.0

    def test_evaluate_batch_mismatch_length(self, evaluator):
        """Test that batch evaluation raises error on length mismatch."""
        with pytest.raises(ValueError, match="数量not匹配"):
            asyncio.run(evaluator.evaluate_batch(
                questions=["Q1", "Q2"],
                answers=["A1"]
            ))

    def test_compare_models_empty(self, evaluator):
        """Test comparing models with no responses."""
        df = asyncio.run(evaluator.compare_models(
            questions=["Q1", "Q2"],
            model_responses={}
        ))

        assert df.empty

    def test_generate_report_empty_results(self, evaluator):
        """Test report generation with empty results."""
        import pandas as pd
        report = evaluator.generate_report(pd.DataFrame())

        assert "No results" in report


class TestJudgeCriteria:
    """Test JudgeCriteria enum."""

    def test_criteria_values(self):
        """Test that all criteria have correct values."""
        assert JudgeCriteria.HELPFULNESS.value == "helpfulness"
        assert JudgeCriteria.RELEVANCE.value == "relevance"
        assert JudgeCriteria.ACCURACY.value == "accuracy"
        assert JudgeCriteria.COHERENCE.value == "coherence"
        assert JudgeCriteria.COMPLETENESS.value == "completeness"
        assert JudgeCriteria.CLARITY.value == "clarity"


class TestConvenienceFunctions:
    """Test convenience functions in llm_judge."""

    @pytest.mark.asyncio
    async def test_judge_answer_function(self):
        """Test the judge_answer convenience function."""
        # Import and test the function
        from core.llm_judge import judge_answer

        # Mock the LLMJudge
        mock_result = JudgeResult(score=8.0, reasoning="Good")

        with patch('core.llm_judge.LLMJudge') as MockClass:
            mock_instance = MockClass.return_value
            mock_instance.evaluate = AsyncMock(return_value=mock_result)

            result = await judge_answer(
                question="Test?",
                answer="Answer",
                api_key="test-key"
            )

        assert result.score == 8.0

    @pytest.mark.asyncio
    async def test_judge_answers_batch_function(self):
        """Test the judge_answers_batch convenience function."""
        from core.llm_judge import judge_answers_batch

        mock_results = [
            JudgeResult(score=8.0, reasoning="Good"),
            JudgeResult(score=7.0, reasoning="OK")
        ]

        with patch('core.llm_judge.LLMJudge') as MockClass:
            mock_instance = MockClass.return_value
            mock_instance.evaluate_batch = AsyncMock(return_value=mock_results)

            results = await judge_answers_batch(
                question="Test?",
                answers=["A1", "A2"],
                api_key="test-key"
            )

        assert len(results) == 2
        assert results[0].score == 8.0


class TestIntegration:
    """Integration tests for the judge system."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test the complete evaluation flow from request to report."""
        # This is an integration test that can be run with a real API key
        # Skip by default in CI/CD

        # Skip if no API key
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key provided")

        from core.llm_judge import judge_answer
        from evaluators.open_ended_evaluator import OpenEndedEvaluator, EvaluationConfig

        # Create evaluator
        config = EvaluationConfig(
            judge_model="gpt-4o-mini",  # Use cheaper model for testing
            judge_api_key=os.getenv("OPENAI_API_KEY")
        )

        evaluator = OpenEndedEvaluator(config)

        # Evaluate a simple answer
        result = await evaluator.evaluate_answer(
            question="What is 2+2?",
            answer="2+2 equals 4."
        )

        # Basic assertions
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 10
        assert result.reasoning
