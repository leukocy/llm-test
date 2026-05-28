"""
单元Test: core/metrics.py - ThinkingMetrics

Test推理ModelMetric calculation，包括:
- TTFT (Time To First Token)
- TTUT (Time To User Text)
- 推理 Token 占比
- 成本估算
"""

import time

import pytest

from core.metrics import (
    ThinkingMetrics,
    ThinkingMetricsResult,
    format_metrics_report,
)


class TestThinkingMetricsResult:
    """Test推理指标Result"""

    def test_default_values(self):
        """default值"""
        result = ThinkingMetricsResult()
        assert result.ttft_ms is None
        assert result.ttut_ms is None
        assert result.reasoning_tokens == 0
        assert result.content_tokens == 0

    def test_with_values(self):
        """带值Initialize"""
        result = ThinkingMetricsResult(
            ttft_ms=100.0,
            ttut_ms=200.0,
            reasoning_tokens=500,
            content_tokens=200,
            total_tokens=700,
            platform="mimo",
            model_id="mimo-v2-flash"
        )
        assert result.ttft_ms == 100.0
        assert result.platform == "mimo"
        assert result.model_id == "mimo-v2-flash"

    def test_reasoning_ratio_field(self):
        """Test推理占比字段"""
        result = ThinkingMetricsResult(
            ttft_ms=100.0,
            reasoning_tokens=100,
            content_tokens=100,
            total_tokens=200,
            reasoning_ratio=0.5
        )
        assert result.reasoning_ratio == 0.5


class TestThinkingMetrics:
    """Test推理Metric calculation器"""

    def test_init(self):
        """Initialize"""
        metrics = ThinkingMetrics(platform="mimo", model_id="mimo-v2-flash")
        assert metrics.platform == "mimo"
        assert metrics.model_id == "mimo-v2-flash"

    def test_record_request_start_end(self):
        """记录请求开始and结束"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        time.sleep(0.01)
        metrics.record_request_end()

        result = metrics.calculate()
        assert result.ttft_ms is None  # 没has记录 token
        assert result.total_time_ms > 0

    def test_record_first_token(self):
        """记录首 token"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        time.sleep(0.01)
        metrics.record_first_token()

        result = metrics.calculate()
        assert result.ttft_ms > 0

    def test_record_reasoning_chunk(self):
        """记录推理内容块"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        time.sleep(0.01)
        metrics.record_reasoning_chunk("Thinking part 1")
        time.sleep(0.01)
        metrics.record_reasoning_chunk("Thinking part 2")

        result = metrics.calculate()
        assert result.reasoning_chars == len("Thinking part 1") + len("Thinking part 2")
        assert result.ttr_ms > 0  # Time to reasoning
        assert result.reasoning_time_ms > 0

    def test_record_content_chunk(self):
        """记录正文内容块"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        time.sleep(0.01)
        metrics.record_content_chunk("Content part 1")
        metrics.record_content_chunk("Content part 2")

        result = metrics.calculate()
        assert result.content_chars == len("Content part 1") + len("Content part 2")
        assert result.ttut_ms > 0

    def test_first_token_set_by_reasoning(self):
        """首 token 由推理内容Trigger"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        time.sleep(0.01)  # 确保has足够时间差
        metrics.record_reasoning_chunk("First")

        result = metrics.calculate()
        assert result.ttft_ms >= 0  # 可能is0or接近0

    def test_first_token_set_by_content(self):
        """首 token 由正文内容Trigger"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        time.sleep(0.01)  # 确保has足够时间差
        metrics.record_content_chunk("First")

        result = metrics.calculate()
        assert result.ttft_ms >= 0  # 可能is0or接近0

    def test_set_usage_with_reasoning_tokens(self):
        """Set包含 reasoning_tokens  usage"""
        metrics = ThinkingMetrics()
        metrics.set_usage({
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "reasoning_tokens": 50
        })

        result = metrics.calculate()
        assert result.reasoning_tokens == 50
        assert result.content_tokens == 150  # 200 - 50
        assert result.total_tokens == 300

    def test_set_usage_with_nested_reasoning(self):
        """Set嵌套结构 reasoning_tokens"""
        metrics = ThinkingMetrics()
        metrics.set_usage({
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "completion_tokens_details": {
                "reasoning_tokens": 80
            }
        })

        result = metrics.calculate()
        assert result.reasoning_tokens == 80
        assert result.content_tokens == 120

    def test_set_usage_estimates_reasoning_by_char_ratio(self):
        """based on字符比例估算推理 token"""
        metrics = ThinkingMetrics()
        metrics.record_reasoning_chunk("AAAA")  # 4 字符
        metrics.record_content_chunk("BB")      # 2 字符
        metrics.set_usage({
            "prompt_tokens": 10,
            "completion_tokens": 60,  # 6 tokens
            "total_tokens": 70
        })

        result = metrics.calculate()
        # 推理占比 = 4/6，therefore reasoning_tokens = 60 * 4/6 = 40
        assert result.reasoning_tokens == 40
        assert result.content_tokens == 20

    def test_calculate_all_metrics(self):
        """Calculate所has指标"""
        metrics = ThinkingMetrics(platform="deepseek", model_id="deepseek-v3")

        # 模拟完整流程
        metrics.record_request_start()
        time.sleep(0.01)
        metrics.record_reasoning_chunk("Thinking")
        time.sleep(0.01)
        metrics.record_reasoning_chunk("More thinking")  # 二推理块用于Calculate reasoning_time
        time.sleep(0.01)
        metrics.record_content_chunk("Answer")
        time.sleep(0.01)
        metrics.record_request_end()

        metrics.set_usage({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "reasoning_tokens": 30
        })

        result = metrics.calculate()
        assert result.ttft_ms >= 0
        assert result.ttr_ms >= 0
        assert result.ttut_ms >= 0
        assert result.total_time_ms > 0
        assert result.reasoning_time_ms > 0  # 现inhas两推理块，应该has时间差
        assert result.reasoning_tokens == 30
        assert result.content_tokens == 20
        assert result.reasoning_ratio == pytest.approx(30/150)  # reasoning / total
        assert result.reasoning_density > 0

    def test_calculate_with_quality_score(self):
        """Calculate带质量分数指标"""
        metrics = ThinkingMetrics(platform="deepseek")

        metrics.record_request_start()
        metrics.record_content_chunk("Answer")
        metrics.record_request_end()

        metrics.set_usage({
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        })

        # quality_score = 8/10
        result = metrics.calculate(quality_score=8.0)

        # 预估成本
        # input: 1000/1M * 0.14 = 0.00014
        # output: 500/1M * 0.28 = 0.00014
        # total: 0.00028
        # quality/$: 8 / 0.00028 ≈ 28571
        assert result.estimated_cost_usd > 0
        assert result.quality_per_dollar > 0

    def test_calculate_free_platform(self):
        """免费平台成本is0"""
        metrics = ThinkingMetrics(platform="mimo")

        metrics.set_usage({
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        })

        result = metrics.calculate()
        # MiMo 免费
        assert result.estimated_cost_usd == 0.0
        assert result.quality_per_dollar is None

    def test_reset(self):
        """ResetStatus"""
        metrics = ThinkingMetrics()
        metrics.record_request_start()
        metrics.record_reasoning_chunk("Test")
        metrics.set_usage({"total_tokens": 100})

        metrics.reset()

        assert metrics._request_start is None
        assert metrics._reasoning_chars == 0
        assert metrics._usage is None

        result = metrics.calculate()
        assert result.ttft_ms is None
        assert result.reasoning_chars == 0

    def test_calculate_without_request_start(self):
        """没has记录请求开始时Calculate"""
        metrics = ThinkingMetrics()
        metrics.record_content_chunk("Content")

        result = metrics.calculate()
        # 没hasStart time，no法Calculate延迟
        assert result.ttft_ms is None
        assert result.ttut_ms is None
        assert result.total_time_ms is None
        # but字符Statistics仍然has效
        assert result.content_chars > 0


class TestPricing:
    """Test定价Calculate"""

    def test_deepseek_pricing(self):
        """DeepSeek 定价"""
        assert ThinkingMetrics.PRICING["deepseek"]["input"] == 0.14
        assert ThinkingMetrics.PRICING["deepseek"]["output"] == 0.28

    def test_mimo_free(self):
        """MiMo 免费"""
        assert ThinkingMetrics.PRICING["mimo"]["input"] == 0.0
        assert ThinkingMetrics.PRICING["mimo"]["output"] == 0.0

    def test_unknown_platform_default_pricing(self):
        """未知平台usedefault定价"""
        metrics = ThinkingMetrics(platform="unknown")

        metrics.set_usage({
            "prompt_tokens": 1000,
            "completion_tokens": 500
        })

        result = metrics.calculate()
        # 应该usedefault定价 (0.5, 0.5)
        assert result.estimated_cost_usd > 0


class TestFormatMetricsReport:
    """Test指标报告Format"""

    def test_format_complete_report(self):
        """Format完整报告"""
        result = ThinkingMetricsResult(
            platform="mimo",
            model_id="mimo-v2-flash",
            ttft_ms=100.0,
            ttr_ms=120.0,
            ttut_ms=500.0,
            total_time_ms=1000.0,
            reasoning_time_ms=300.0,
            reasoning_tokens=100,
            content_tokens=50,
            total_tokens=150,
            reasoning_ratio=2/3,
            reasoning_chars=500,
            content_chars=250,
            reasoning_density=2.0,
            estimated_cost_usd=0.001,
            quality_per_dollar=5000.0
        )

        report = format_metrics_report(result)
        assert "mimo" in report
        assert "mimo-v2-flash" in report
        assert "TTFT" in report
        assert "100ms" in report or "100" in report
        assert "推理 Token" in report

    def test_format_partial_report(self):
        """Format部分Data报告"""
        result = ThinkingMetricsResult(
            platform="test",
            model_id="test-model",
            reasoning_tokens=100,
            content_tokens=50
        )

        report = format_metrics_report(result)
        assert "test" in report
        assert "N/A" in report  # 缺失延迟指标

    def test_format_report_with_none_values(self):
        """Process None 值"""
        result = ThinkingMetricsResult(
            ttft_ms=None,
            estimated_cost_usd=None
        )

        report = format_metrics_report(result)
        assert "N/A" in report


class TestReasoningRatio:
    """Test推理占比Calculate"""

    def test_reasoning_ratio_calculation(self):
        """推理占比Calculate"""
        metrics = ThinkingMetrics()

        metrics.set_usage({
            "total_tokens": 100,
            "completion_tokens": 60,
            "reasoning_tokens": 40
        })

        result = metrics.calculate()
        # reasoning_ratio = reasoning_tokens / total_tokens
        assert result.reasoning_ratio == pytest.approx(40/100)

    def test_reasoning_ratio_zero_tokens(self):
        """零 token 时占比"""
        metrics = ThinkingMetrics()
        result = metrics.calculate()
        assert result.reasoning_ratio == 0.0

    def test_reasoning_density_calculation(self):
        """推理密度Calculate"""
        metrics = ThinkingMetrics()
        metrics.record_reasoning_chunk("AAAA BBBB")  # 9 字符
        metrics.record_content_chunk("CC")            # 2 字符

        result = metrics.calculate()
        assert result.reasoning_density == pytest.approx(9/2)
