from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.benchmark_runner import BenchmarkRunner


class TestBenchmarkRunner:
    @pytest.fixture
    def runner(self):
        # Mock dependencies
        placeholder = MagicMock()
        progress_bar = MagicMock()
        status_text = MagicMock()
        log_placeholder = MagicMock()

        return BenchmarkRunner(
            placeholder=placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="字符数 (Fallback)",
            csv_filename="test.csv",
            api_key="test-key",
            log_placeholder=log_placeholder,
            provider="TestProvider"
        )

    def test_calculate_metrics_normal(self, runner):
        start_time = 100.0
        first_token_time = 100.5
        end_time = 102.0
        completion_tokens = 10

        ttft, tps, tpot, p95, p99, gen_time = runner._calculate_metrics(start_time, first_token_time, end_time, completion_tokens)

        expected_ttft_raw = 0.5
        expected_ttft = expected_ttft_raw

        # Total time = 2.0
        # Generation time = 2.0 - 0.5 = 1.5
        # TPS = 10 / 1.5
        expected_tps = 10 / (2.0 - expected_ttft)

        assert ttft == pytest.approx(expected_ttft)
        assert tps == pytest.approx(expected_tps)

    def test_calculate_metrics_single_token(self, runner):
        start_time = 100.0
        first_token_time = 100.5
        end_time = 100.5 # Instant finish after first token
        completion_tokens = 1

        ttft, tps, tpot, p95, p99, gen_time = runner._calculate_metrics(start_time, first_token_time, end_time, completion_tokens)

        expected_ttft_raw = 0.5
        expected_ttft = expected_ttft_raw

        # Generation time = 100.5 - 100.5 = 0
        # TPS = 0 when generation_time is 0
        expected_tps = 0

        assert ttft == pytest.approx(expected_ttft)
        assert tps == pytest.approx(expected_tps)

    def test_calculate_metrics_no_first_token(self, runner):
        # Case where first_token_time is None (e.g. error or empty response)
        ttft, tps, _, _, _, _ = runner._calculate_metrics(100.0, None, 102.0, 0)
        assert ttft == 0
        assert tps == 0

    def test_get_empty_metrics(self, runner):
        metrics = runner._get_empty_metrics()
        assert metrics['ttft'] == 0
        assert metrics['tps'] == 0
        assert metrics['error'] is None
        # Checking implementation: return {"ttft": 0, ..., "token_calc_method": "Error"}
        assert metrics['token_calc_method'] == "Error"

    @pytest.mark.asyncio
    async def test_get_completion_decode_time_uses_skip_first_token_window(self, runner):
        class FakeProvider:
            async def get_completion(self, *args, **kwargs):
                return {
                    "created_at": 1000.0,
                    "start_time": 100.0,
                    "first_token_time": 100.4,
                    "end_time": 101.6,
                    "full_response_content": "one two three four",
                    "usage_info": {"prompt_tokens": 16, "completion_tokens": 4},
                    "token_timestamps": [100.4, 100.8, 101.2, 101.6],
                }

        runner.provider = FakeProvider()
        runner.skip_first_token_for_tps = True
        runner._update_log = MagicMock()

        result = await runner.get_completion(None, 1, "prompt", 4)

        assert result["decode_time"] == pytest.approx(0.8)
        assert result["decode_tokens_for_tps"] == 3
        assert result["tps"] == pytest.approx(3 / 0.8)

    @pytest.mark.asyncio
    async def test_long_context_system_output_throughput_uses_skip_adjusted_decode_tokens(self, runner, tmp_path):
        runner.csv_file = str(tmp_path / "long_context.csv")
        runner._start_db_run = MagicMock()
        runner._batch_save_results_to_db = MagicMock()
        runner._complete_db_run = MagicMock()
        runner.update_ui = MagicMock()
        runner._calibrate_prompt = MagicMock(return_value="prompt")
        runner._run_long_context_request = AsyncMock(
            return_value={
                "session_id": 0,
                "error": None,
                "ttft": 0.4,
                "tps": 3.75,
                "tpot": 0.4,
                "tpot_p95": 0.4,
                "tpot_p99": 0.4,
                "prefill_tokens": 16,
                "decode_tokens": 4,
                "decode_tokens_for_tps": 3,
                "decode_time": 0.8,
                "total_time": 1.2,
                "cache_hit_tokens": 0,
                "token_calc_method": "API (usage field)",
            }
        )

        df = await runner.run_long_context_test([64], rounds_per_level=1, max_tokens=4)

        assert df.loc[0, "system_output_throughput"] == pytest.approx(3.75)

    @pytest.mark.asyncio
    async def test_concurrency_prompt_generation_subtracts_template_tokens(self, tmp_path):
        runner = BenchmarkRunner(
            placeholder=MagicMock(),
            progress_bar=MagicMock(),
            status_text=MagicMock(),
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="Character Count (Fallback)",
            csv_filename=str(tmp_path / "concurrency.csv"),
            api_key="test-key",
            log_placeholder=MagicMock(),
            provider="TestProvider",
            template_tokens=5,
        )
        runner._start_db_run = MagicMock()
        runner._batch_save_results_to_db = MagicMock()
        runner._complete_db_run = MagicMock()
        runner.update_ui = MagicMock()
        runner._get_tokenizer = MagicMock(return_value=object())
        runner._calibrate_prompt = MagicMock(return_value="prompt")
        runner._run_concurrency_batch = AsyncMock(return_value=[])

        await runner.run_concurrency_test([2], rounds_per_level=1, max_tokens=4, input_tokens_target=20)

        calibrated_targets = [call.args[0] for call in runner._calibrate_prompt.call_args_list]
        assert calibrated_targets == [15, 15]


class TestBenchmarkRunnerSystemInfo:
    """Test系统信息Get"""

    @pytest.fixture
    def runner(self):
        placeholder = MagicMock()
        progress_bar = MagicMock()
        status_text = MagicMock()
        log_placeholder = MagicMock()

        return BenchmarkRunner(
            placeholder=placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="字符数 (Fallback)",
            csv_filename="test.csv",
            api_key="test-key",
            log_placeholder=log_placeholder,
            provider="TestProvider"
        )

    def test_get_system_info_returns_required_fields(self, runner):
        """Test系统信息包含必需字段"""
        info = runner.get_system_info()

        required_fields = [
            "system", "processor", "python", "hostname",
            "memory", "cpu_count", "gpu", "mainboard",
            "model_name", "engine_name"
        ]
        for field in required_fields:
            assert field in info

    def test_get_system_info_with_custom_overrides(self, runner):
        """TestCustom系统信息覆盖"""
        import streamlit as st
        st.session_state.custom_sys_info = {
            'processor': 'Custom CPU',
            'gpu': 'Custom GPU',
            'memory': '64GB RAM',
            'engine_name': 'vLLM'
        }

        info = runner.get_system_info()

        assert info['processor'] == 'Custom CPU'
        assert info['gpu'] == 'Custom GPU'
        assert info['memory'] == '64GB RAM'
        assert info['engine_name'] == 'vLLM'

    def test_get_system_info_model_name_from_config(self, runner):
        """TestModel name来自Configure"""
        info = runner.get_system_info()
        assert info['model_name'] == 'test-model'

    def test_get_system_info_engine_name_empty_without_override(self, runner):
        """Test引擎名称来自 provider"""
        import streamlit as st
        st.session_state.custom_sys_info = {}

        info = runner.get_system_info()
        # Engine refers to the inference backend, so provider names are not a fallback.
        assert info['engine_name'] == ''


class TestBenchmarkRunnerInitialization:
    """Test BenchmarkRunner Initialize"""

    @pytest.fixture
    def mock_dependencies(self):
        return {
            'placeholder': MagicMock(),
            'progress_bar': MagicMock(),
            'status_text': MagicMock(),
            'log_placeholder': MagicMock()
        }

    def test_initialization_basic_params(self, mock_dependencies):
        """Test基本参数Initialize"""
        runner = BenchmarkRunner(
            api_base_url="http://test.api",
            model_id="test-model",
            tokenizer_option="API (usage field)",
            csv_filename="test.csv",
            api_key="test-key",
            provider="openai",
            **mock_dependencies
        )

        assert runner.api_base_url == "http://test.api"
        assert runner.model_id == "test-model"
        assert runner.tokenizer_option == "API (usage field)"
        assert runner.csv_file == "test.csv"
        assert runner.api_key == "test-key"

    def test_initialization_thinking_params(self, mock_dependencies):
        """TestThinking parametersInitialize"""
        runner = BenchmarkRunner(
            api_base_url="http://test",
            model_id="thinking-model",
            tokenizer_option="API",
            csv_filename="test.csv",
            api_key="test-key",
            provider="openai",
            thinking_enabled=True,
            thinking_budget=8192,
            reasoning_effort="high",
            **mock_dependencies
        )

        assert runner.thinking_enabled == True
        assert runner.thinking_budget == 8192
        assert runner.reasoning_effort == "high"

    def test_initialization_counters(self, mock_dependencies):
        """Test计数器Initialize"""
        runner = BenchmarkRunner(
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="API",
            csv_filename="test.csv",
            api_key="test-key",
            provider="openai",
            **mock_dependencies
        )

        assert runner.completed_requests == 0
        assert runner.total_requests == 0
        assert runner.results_list == []
        assert runner.all_outputs == []

    def test_initialization_combined_csv_columns(self, mock_dependencies):
        """Test CSV 列定义"""
        runner = BenchmarkRunner(
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="API",
            csv_filename="test.csv",
            api_key="test-key",
            provider="openai",
            **mock_dependencies
        )

        expected_columns = [
            "test_type", "concurrency", "round",
            "input_tokens_target",
            "context_length_target",
            "session_id", "ttft", "tps", "prefill_speed",
            "prefill_tokens", "decode_tokens", "api_prefill", "api_decode",
            "cache_hit_tokens",
            "token_calc_method", "error", "system_output_throughput",
            "system_input_throughput", "rps", "tpot_p95", "tpot_p99"
        ]

        assert runner.combined_csv_columns == expected_columns


class TestBenchmarkRunnerMetrics:
    """TestMetric calculation边缘情况"""

    @pytest.fixture
    def runner(self):
        placeholder = MagicMock()
        progress_bar = MagicMock()
        status_text = MagicMock()
        log_placeholder = MagicMock()

        return BenchmarkRunner(
            placeholder=placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="字符数 (Fallback)",
            csv_filename="test.csv",
            api_key="test-key",
            log_placeholder=log_placeholder,
            provider="TestProvider"
        )

    def test_calculate_metrics_with_latency_offset(self, runner):
        """Test延迟偏移Calculate"""
        runner.latency_offset = 0.1

        start_time = 100.0
        first_token_time = 100.6
        end_time = 102.0
        completion_tokens = 10

        ttft, tps, _, _, _, _ = runner._calculate_metrics(
            start_time, first_token_time, end_time, completion_tokens
        )

        # TTFT raw = 0.6, offset = 0.1, adjusted = 0.5
        expected_ttft = 0.6 - 0.1
        assert ttft == pytest.approx(expected_ttft)

    def test_calculate_metrics_zero_completion_tokens(self, runner):
        """Test零完成 tokens 情况"""
        start_time = 100.0
        first_token_time = 100.5
        end_time = 101.0
        completion_tokens = 0

        ttft, tps, _, _, _, _ = runner._calculate_metrics(
            start_time, first_token_time, end_time, completion_tokens
        )

        # TTFT should still be calculated
        assert ttft == pytest.approx(0.5)
        # TPS should be 0 since no tokens generated
        assert tps == 0

    def test_calculate_metrics_negative_times(self, runner):
        """Test异常时间顺序（not应该发生，butneed防御）"""
        start_time = 100.0
        first_token_time = 99.0  # Before start!
        end_time = 98.0  # Before first token!

        ttft, tps, _, _, _, _ = runner._calculate_metrics(
            start_time, first_token_time, end_time, 10
        )

        # Should handle gracefully
        assert ttft >= 0
        assert tps >= 0

    def test_calculate_metrics_large_token_count(self, runner):
        """Test大量 tokens Calculate"""
        start_time = 100.0
        first_token_time = 101.0
        end_time = 110.0
        completion_tokens = 10000

        ttft, tps, _, _, _, _ = runner._calculate_metrics(
            start_time, first_token_time, end_time, completion_tokens
        )

        expected_tps = 10000 / (10.0 - 1.0)
        assert tps == pytest.approx(expected_tps)


class TestBenchmarkRunnerEdgeCases:
    """Test边缘情况andErrorProcess"""

    @pytest.fixture
    def runner(self):
        placeholder = MagicMock()
        progress_bar = MagicMock()
        status_text = MagicMock()
        log_placeholder = MagicMock()

        return BenchmarkRunner(
            placeholder=placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
            api_base_url="http://test",
            model_id="test-model",
            tokenizer_option="字符数 (Fallback)",
            csv_filename="test.csv",
            api_key="test-key",
            log_placeholder=log_placeholder,
            provider="TestProvider"
        )

    def test_concurrent_requests_tracking(self, runner):
        """Test并发请求跟踪"""
        runner.total_requests = 100
        runner.completed_requests = 50

        assert runner.total_requests == 100
        assert runner.completed_requests == 50

    def test_results_storage(self, runner):
        """Result storage"""
        result1 = {"ttft": 0.5, "tps": 10.0}
        result2 = {"ttft": 0.6, "tps": 15.0}

        runner.results_list.append(result1)
        runner.results_list.append(result2)

        assert len(runner.results_list) == 2
        assert runner.results_list[0]["tps"] == 10.0
        assert runner.results_list[1]["tps"] == 15.0

    def test_all_outputs_storage(self, runner):
        """Test输出存储"""
        output1 = "Response 1"
        output2 = "Response 2"

        runner.all_outputs.append(output1)
        runner.all_outputs.append(output2)

        assert len(runner.all_outputs) == 2
        assert runner.all_outputs[0] == "Response 1"

    def test_last_output_tracking(self, runner):
        """Test最后输出跟踪"""
        runner.last_output = "Latest response"
        assert runner.last_output == "Latest response"

    @patch('core.benchmark_runner.get_provider')
    def test_provider_initialization(self, mock_get_provider):
        """Test Provider Initialize"""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        placeholder = MagicMock()
        progress_bar = MagicMock()
        status_text = MagicMock()
        log_placeholder = MagicMock()

        runner = BenchmarkRunner(
            placeholder=placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
            api_base_url="http://custom.api",
            model_id="custom-model",
            tokenizer_option="API",
            csv_filename="test.csv",
            api_key="custom-key",
            log_placeholder=log_placeholder,
            provider="custom_provider"
        )

        # Verify get_provider was called with correct arguments
        mock_get_provider.assert_called_once_with(
            "custom_provider",
            "http://custom.api",
            "custom-key",
            "custom-model"
        )
        assert runner.provider == mock_provider
