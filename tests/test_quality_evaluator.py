"""
QualityEvaluator 模块单元Test

TestQuality Assessment引擎核心功能：
- QualityTestConfig Configure类
- QualityEvaluator Initialize
- Token 计数
- 缓存功能
- EvaluatorRegisterandGet
- Result汇总
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from core.quality_evaluator import (
    QualityEvaluator,
    QualityTestConfig,
    quick_evaluate
)


class TestQualityTestConfig:
    """Test QualityTestConfig Data类"""

    def test_create_default_config(self):
        """TestCreatedefaultConfigure"""
        config = QualityTestConfig()
        assert config.datasets == ["mmlu"]
        assert config.num_shots == 5
        assert config.max_samples is None
        assert config.temperature == 0.0
        assert config.max_tokens == 256
        assert config.concurrency == 4
        assert config.thinking_enabled is False
        assert config.use_cache is True

    def test_create_custom_config(self):
        """TestCreateCustom Configuration"""
        config = QualityTestConfig(
            datasets=["gsm8k", "math500"],
            num_shots=3,
            max_samples=50,
            temperature=0.5,
            max_tokens=512,
            concurrency=8,
            thinking_enabled=True,
            thinking_budget=2048,
            reasoning_effort="high"
        )
        assert config.datasets == ["gsm8k", "math500"]
        assert config.num_shots == 3
        assert config.max_samples == 50
        assert config.temperature == 0.5
        assert config.thinking_enabled is True
        assert config.thinking_budget == 2048
        assert config.reasoning_effort == "high"

    def test_config_to_dict(self):
        """TestConfigureConvertis字典"""
        config = QualityTestConfig(
            datasets=["mmlu"],
            num_shots=5,
            temperature=0.0
        )
        d = config.to_dict()
        assert d["datasets"] == ["mmlu"]
        assert d["num_shots"] == 5
        assert d["temperature"] == 0.0
        assert "use_cache" in d

    def test_config_with_subsets(self):
        """Test带子集Configure"""
        config = QualityTestConfig(
            datasets=["mmlu", "ceval"],
            subsets={"mmlu": ["stem", "humanities"], "ceval": ["computer"]}
        )
        assert config.subsets is not None
        assert "stem" in config.subsets["mmlu"]
        assert "computer" in config.subsets["ceval"]

    def test_config_with_dataset_overrides(self):
        """test data集覆盖Configure"""
        config = QualityTestConfig(
            datasets=["mmlu", "gsm8k"],
            dataset_overrides={
                "mmlu": {"max_tokens": 1024, "temperature": 0.5},
                "gsm8k": {"max_tokens": 512}
            }
        )
        assert config.dataset_overrides["mmlu"]["max_tokens"] == 1024
        assert config.dataset_overrides["gsm8k"]["max_tokens"] == 512


class TestQualityEvaluator:
    """Test QualityEvaluator 类"""

    @pytest.fixture
    def mock_provider(self):
        """Mock Provider"""
        provider = MagicMock()
        provider.get_completion = AsyncMock(return_value={
            'full_response_content': 'Test response',
            'start_time': 0,
            'first_token_time': 0.1,
            'end_time': 0.5,
            'usage_info': {'prompt_tokens': 10, 'completion_tokens': 5}
        })
        return provider

    @pytest.fixture
    def evaluator(self, mock_provider):
        """Create QualityEvaluator 实例"""
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    api_key="test-key",
                    enable_cache=False
                )
                # Mock provider
                evaluator.provider = mock_provider
                return evaluator

    # ==================== InitializeTest ====================

    def test_init_basic(self, mock_provider):
        """Test基本Initialize"""
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    api_key="test-key"
                )
                assert evaluator.api_base_url == "http://test.com"
                assert evaluator.model_id == "test-model"
                assert evaluator.api_key == "test-key"
                assert evaluator.results == {}
                assert evaluator.is_running is False

    def test_init_with_custom_output_dir(self, mock_provider):
        """TestCustom输出目录"""
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    output_dir="custom_output"
                )
                assert evaluator.output_dir == "custom_output"

    def test_init_with_log_callback(self, mock_provider):
        """Test带LogCallbackInitialize"""
        callback = MagicMock()
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    log_callback=callback
                )
                assert evaluator.log_callback == callback

    # ==================== Token 计数Test ====================

    def test_count_tokens_empty_string(self, evaluator):
        """Test空字符串 token 计数"""
        assert evaluator.count_tokens("") == 0

    def test_count_tokens_without_tokenizer(self, evaluator):
        """Testno tokenizer 时字符估算"""
        evaluator.tokenizer = None
        count = evaluator.count_tokens("hello world")
        # 字符估算：约 len // 3
        assert count > 0

    def test_count_tokens_with_tokenizer(self, evaluator):
        """Testhas tokenizer 时计数"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        evaluator.tokenizer = mock_tokenizer
        count = evaluator.count_tokens("test")
        assert count == 5

    def test_count_tokens_tokenizer_error_fallback(self, evaluator):
        """Test tokenizer 出错时回退到字符估算"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(side_effect=Exception("Error"))
        evaluator.tokenizer = mock_tokenizer
        count = evaluator.count_tokens("hello world")
        # 回退到字符估算
        assert count > 0

    # ==================== 缓存Test ====================

    def test_get_cache_stats_disabled(self, evaluator):
        """Test禁用缓存Statistics"""
        evaluator.cache = None
        stats = evaluator.get_cache_stats()
        assert stats["enabled"] is False

    def test_get_cache_stats_enabled(self, evaluator):
        """Test启用缓存Statistics"""
        mock_cache = MagicMock()
        mock_cache_stats = MagicMock()
        mock_cache_stats.total_entries = 100
        mock_cache_stats.total_bytes = 1024000
        mock_cache.get_stats = MagicMock(return_value=mock_cache_stats)

        evaluator.cache = mock_cache
        evaluator._cache_stats = {"hits": 10, "misses": 5}

        stats = evaluator.get_cache_stats()
        assert stats["enabled"] is True
        assert stats["session_hits"] == 10
        assert stats["session_misses"] == 5
        assert stats["session_hit_rate"] == 10 / 15

    def test_clear_cache_model_only(self, evaluator):
        """Test清除单 models缓存"""
        mock_cache = MagicMock()
        evaluator.cache = mock_cache
        evaluator.clear_cache(model_only=True)
        mock_cache.clear.assert_called_once_with(model_id="test-model")

    def test_clear_cache_all(self, evaluator):
        """Test清除所has缓存"""
        mock_cache = MagicMock()
        evaluator.cache = mock_cache
        evaluator.clear_cache(model_only=False)
        mock_cache.clear.assert_called_once_with()

    # ==================== LogTest ====================

    def test_log_with_callback(self, evaluator):
        """Test带CallbackLog"""
        callback = MagicMock()
        evaluator.log_callback = callback

        evaluator._log("Test message")
        callback.assert_called_once()

    def test_log_without_callback(self, evaluator, capsys):
        """Testnot带CallbackLog"""
        evaluator.log_callback = None
        evaluator._log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    # ==================== EvaluatorRegisterTest ====================

    def test_register_evaluator(self, evaluator):
        """TestRegisterEvaluator"""
        mock_evaluator_class = MagicMock()
        evaluator.register_evaluator("test_dataset", mock_evaluator_class)
        assert evaluator.EVALUATOR_CLASSES["test_dataset"] == mock_evaluator_class

    def test_get_evaluator_not_registered(self, evaluator):
        """TestGet未RegisterEvaluator"""
        config = QualityTestConfig()
        result = evaluator.get_evaluator("unknown_dataset", config)
        assert result is None

    def test_get_evaluator_registered(self, evaluator):
        """TestGet已RegisterEvaluator"""
        mock_evaluator_class = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator_instance

        evaluator.register_evaluator("test_dataset", mock_evaluator_class)

        config = QualityTestConfig(num_shots=3, max_samples=10)
        result = evaluator.get_evaluator("test_dataset", config)

        assert result is not None
        mock_evaluator_class.assert_called_once()

    def test_get_evaluator_custom_needle_with_filter(self, evaluator):
        """TestCustom大海捞针Evaluator带Filter器"""
        mock_evaluator_class = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator_instance

        evaluator.register_evaluator("custom_needle", mock_evaluator_class)

        config = QualityTestConfig()
        result = evaluator.get_evaluator("custom_needle_frankenstein", config, test_filter="frankenstein")

        assert result is not None
        # Checkis否传递 test_filter
        call_kwargs = mock_evaluator_class.call_args[1]
        assert call_kwargs["test_filter"] == "frankenstein"

    # ==================== 响应GetTest ====================

    @pytest.mark.asyncio
    async def test_get_response_success(self, evaluator):
        """Test成功Get响应"""
        result = await evaluator._get_response_with_metrics(
            prompt="Test prompt",
            temperature=0.0,
            max_tokens=256
        )
        assert result["content"] == "Test response"
        assert result["error"] is None
        assert result["input_tokens"] >= 0
        assert result["output_tokens"] >= 0

    @pytest.mark.asyncio
    async def test_get_response_with_error(self, evaluator):
        """TestGet响应时出错"""
        evaluator.provider.get_completion = AsyncMock(return_value={
            'error': 'API Error',
            'full_response_content': ''
        })

        result = await evaluator._get_response_with_metrics(
            prompt="Test prompt"
        )
        assert result["content"] == ""
        assert result["error"] == "API Error"

    # ==================== 停止andStatusTest ====================

    def test_stop_evaluation(self, evaluator):
        """Test停止评估"""
        evaluator.stop()
        assert evaluator.should_stop is True

    # ==================== Dataset路径Test ====================

    def test_dataset_paths_mapping(self):
        """test data集路径映射"""
        assert "mmlu" in QualityEvaluator.DATASET_PATHS
        assert "gsm8k" in QualityEvaluator.DATASET_PATHS
        assert "math500" in QualityEvaluator.DATASET_PATHS
        assert QualityEvaluator.DATASET_PATHS["mmlu"] == "datasets/mmlu"


class TestQualityEvaluatorResults:
    """Test QualityEvaluator ResultProcess功能"""

    @pytest.fixture
    def evaluator_with_results(self):
        """Create带ResultEvaluator"""
        mock_provider = MagicMock()
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    enable_cache=False
                )

        # Mock Result
        mock_result = MagicMock()
        mock_result.model_id = "test-model"
        mock_result.accuracy = 0.85
        mock_result.correct_samples = 85
        mock_result.total_samples = 100
        mock_result.duration_seconds = 10.5
        mock_result.timestamp = "2024-01-01 12:00:00"
        mock_result.config = {
            "thinking_enabled": False,
            "thinking_budget": 0,
            "reasoning_effort": "N/A"
        }
        mock_result.by_category = {}

        evaluator.results = {"mmlu": mock_result}
        return evaluator, mock_result

    def test_get_summary_df(self, evaluator_with_results):
        """TestGet汇总 DataFrame"""
        evaluator, _ = evaluator_with_results
        df = evaluator.get_summary_df()
        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["Dataset"] == "mmlu"
        assert df.iloc[0]["Model"] == "test-model"
        assert df.iloc[0]["Accuracy"] == 0.85

    def test_get_summary_df_empty(self):
        """Test空Result汇总 DataFrame"""
        mock_provider = MagicMock()
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    enable_cache=False
                )

        df = evaluator.get_summary_df()
        assert df.empty

    def test_get_category_breakdown(self, evaluator_with_results):
        """TestGet分类别Statistics"""
        evaluator, mock_result = evaluator_with_results
        mock_result.by_category = {
            "math": {"accuracy": 0.9, "count": 50},
            "history": {"accuracy": 0.8, "count": 30}
        }

        df = evaluator.get_category_breakdown("mmlu")
        assert not df.empty
        assert len(df) == 2

    def test_get_category_breakdown_unknown_dataset(self, evaluator_with_results):
        """TestGet未知Dataset分类别Statistics"""
        evaluator, _ = evaluator_with_results
        df = evaluator.get_category_breakdown("unknown")
        assert df.empty

    def test_summary_df_thinking_mode_enabled(self):
        """TestThinking mode启用汇总Display"""
        mock_provider = MagicMock()
        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                evaluator = QualityEvaluator(
                    api_base_url="http://test.com",
                    model_id="test-model",
                    enable_cache=False
                )

        mock_result = MagicMock()
        mock_result.model_id = "test-model"
        mock_result.accuracy = 0.85
        mock_result.correct_samples = 85
        mock_result.total_samples = 100
        mock_result.duration_seconds = 10.5
        mock_result.config = {
            "thinking_enabled": True,
            "thinking_budget": 2048,
            "reasoning_effort": "high"
        }
        mock_result.by_category = {}

        evaluator.results = {"mmlu": mock_result}

        df = evaluator.get_summary_df()
        assert not df.empty
        # CheckThinking mode列包含正确信息
        assert "✅" in df.iloc[0]["Thinking mode"]


class TestConvenienceFunctions:
    """Test便捷函数"""

    @pytest.mark.asyncio
    async def test_quick_evaluate(self):
        """Test quick_evaluate 便捷函数"""
        mock_provider = MagicMock()
        mock_provider.get_completion = AsyncMock(return_value={
            'full_response_content': 'Test',
            'start_time': 0,
            'first_token_time': 0.1,
            'end_time': 0.5,
            'usage_info': {'prompt_tokens': 10, 'completion_tokens': 5}
        })

        with patch('core.quality_evaluator.get_provider', return_value=mock_provider):
            with patch('core.quality_evaluator.os.makedirs'):
                with patch.object(QualityEvaluator, 'run_evaluation', new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = {}

                    results = await quick_evaluate(
                        api_base_url="http://test.com",
                        model_id="test-model",
                        datasets=["mmlu"],
                        num_shots=5,
                        max_samples=100
                    )

                    mock_run.assert_called_once()
