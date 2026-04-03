"""
Test Runner 模块单元Test

TestTest执行器类核心功能

注意: 依赖 conftest.py in全局 streamlit mock
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, call, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_config():
    """CreateTestConfigure"""
    return {
        'api_base_url': 'https://api.test.com',
        'model_id': 'test-model',
        'api_key': 'test-key',
        'tokenizer_option': 'API (usage field)',
        'hf_tokenizer_model_id': '',
        'provider': 'openai'
    }


@pytest.fixture
def clean_session_state():
    """Cleanup session state"""
    import streamlit as st
    st.session_state._data.clear()
    return st.session_state


@pytest.fixture
def session_state_module():
    """动态Import session_state 模块"""
    if 'config.session_state' in sys.modules:
        del sys.modules['config.session_state']
    from config import session_state
    # Initialize the session state
    session_state.init_session_state()
    return session_state


class TestTestExecutor:
    """Test TestExecutor 类"""

    @pytest.fixture
    def executor(self, mock_config, clean_session_state, session_state_module):
        """Create TestExecutor 实例"""
        from ui.test_runner import TestExecutor
        return TestExecutor(mock_config)

    def test_init(self, executor, mock_config):
        """TestInitialize"""
        assert executor.config == mock_config
        assert executor.test_running == False

    @patch('core.benchmark_runner.BenchmarkRunner')
    def test_capture_system_info(self, mock_runner_class, executor, clean_session_state):
        """TestSystem info capture"""
        import streamlit as st

        # Set mock runner
        mock_runner = MagicMock()
        mock_runner.get_system_info.return_value = {
            'model_name': 'test-model',
            'engine_name': 'test-engine'
        }
        mock_runner_class.return_value = mock_runner

        # 调用方法
        executor._capture_system_info(mock_runner)

        # ValidateResult
        assert 'system_info' in st.session_state._data
        sys_info = st.session_state._data['system_info']
        assert sys_info['model_name'] == 'test-model'
        assert sys_info['engine_name'] == 'test-engine'

    @patch('core.benchmark_runner.BenchmarkRunner')
    def test_capture_system_info_with_custom_overrides(self, mock_runner_class, executor, clean_session_state):
        """TestSystem info capture时Merge用户Custom信息"""
        import streamlit as st

        # SetCustom系统信息
        st.session_state._data['custom_sys_info'] = {
            'processor': 'Custom CPU',
            'gpu': 'Custom GPU'
        }

        # Set mock runner
        mock_runner = MagicMock()
        mock_runner.get_system_info.return_value = {
            'model_name': 'test-model',
            'engine_name': 'test-engine',
            'processor': 'Auto CPU'
        }
        mock_runner_class.return_value = mock_runner

        # 调用方法
        executor._capture_system_info(mock_runner)

        # ValidateCustom信息覆盖自动检测信息
        sys_info = st.session_state._data['system_info']
        assert sys_info['processor'] == 'Custom CPU'
        assert sys_info['gpu'] == 'Custom GPU'
        # other信息保留
        assert sys_info['model_name'] == 'test-model'

    @patch('core.benchmark_runner.BenchmarkRunner')
    def test_capture_system_info_fallbacks(self, mock_runner_class, executor, clean_session_state):
        """TestSystem info capture时回退值"""
        import streamlit as st

        # Set mock runner Return未知值
        mock_runner = MagicMock()
        mock_runner.get_system_info.return_value = {
            'model_name': 'Unknown',
            'engine_name': 'Unknown'
        }
        mock_runner_class.return_value = mock_runner

        # 调用方法
        executor._capture_system_info(mock_runner)

        # Validate回退值
        sys_info = st.session_state._data['system_info']
        assert sys_info['model_name'] == 'test-model'  # use config in model_id
        assert sys_info['engine_name'] == 'openai'  # use config in provider

    def test_calculate_total_requests_concurrency_test(self, executor):
        """Test并发TestTotal requestsCalculate"""
        # use简单对象来捕获 total_requests 赋值
        class MockRunner:
            def __init__(self):
                self.total_requests = 0

        mock_runner = MockRunner()
        selected_concurrencies = [1, 2, 4]
        rounds_per_level = 3

        # Create具has正确 __name__ 函数
        def dummy_func(r, *args):
            pass
        dummy_func.__name__ = 'run_concurrency_test'

        executor._calculate_total_requests(
            dummy_func,
            mock_runner,
            (selected_concurrencies, rounds_per_level, 100, 2000)
        )

        # (1 + 2 + 4) * 3 = 21
        assert mock_runner.total_requests == 21

    def test_calculate_total_requests_prefill_test(self, executor):
        """Test Prefill TestTotal requestsCalculate"""
        class MockRunner:
            def __init__(self):
                self.total_requests = 0

        mock_runner = MockRunner()
        token_levels = [100, 500, 1000]
        requests_per_level = 10

        def dummy_func(r, *args):
            pass
        dummy_func.__name__ = 'run_prefill_test'

        executor._calculate_total_requests(
            dummy_func,
            mock_runner,
            (token_levels, requests_per_level, 500)
        )

        # 3 * 10 = 30
        assert mock_runner.total_requests == 30

    def test_calculate_total_requests_long_context_test(self, executor):
        """Test长onunder文TestTotal requestsCalculate"""
        class MockRunner:
            def __init__(self):
                self.total_requests = 0

        mock_runner = MockRunner()
        context_lengths = [2000, 4000, 8000]
        rounds = 5

        def dummy_func(r, *args):
            pass
        dummy_func.__name__ = 'run_long_context_test'

        executor._calculate_total_requests(
            dummy_func,
            mock_runner,
            (context_lengths, rounds, 1000)
        )

        # 3 * 5 = 15
        assert mock_runner.total_requests == 15

    def test_calculate_total_requests_throughput_matrix_test(self, executor):
        """Test吞吐量矩阵TestTotal requestsCalculate"""
        class MockRunner:
            def __init__(self):
                self.total_requests = 0

        mock_runner = MockRunner()
        selected_concurrencies = [1, 2]
        context_lengths = [1000, 2000]
        rounds_per_level = 3

        def dummy_func(r, *args):
            pass
        dummy_func.__name__ = 'run_throughput_matrix_test'

        executor._calculate_total_requests(
            dummy_func,
            mock_runner,
            (selected_concurrencies, context_lengths, rounds_per_level, 500, 100)
        )

        # (1*2 + 2*2) * 3 = 18
        assert mock_runner.total_requests == 18

    def test_capture_test_config_concurrency(self, executor, clean_session_state, session_state_module):
        """Test并发TestConfigure捕获"""
        import streamlit as st
        session_state_module.init_session_state()

        test_function = lambda r, *args: None
        test_function.__name__ = 'run_concurrency_test'

        executor._capture_test_config(
            test_function,
            '并发性能Test',
            ([1, 2], 3, 100, 2000),
            '20250130_120000'
        )

        assert 'test_config' in st.session_state._data
        config = st.session_state._data['test_config']
        assert config['Test Type'] == '并发性能Test'
        assert config['Model ID'] == 'test-model'
        assert config['Input Tokens'] == '2000'
        assert config['Concurrency Levels'] == '[1, 2]'
        assert config['Rounds per Level'] == '3'
        assert config['Max Tokens'] == '100'

    def test_capture_test_config_long_context(self, executor, clean_session_state, session_state_module):
        """Test长onunder文TestConfigure捕获"""
        import streamlit as st
        session_state_module.init_session_state()

        test_function = lambda r, *args: None
        test_function.__name__ = 'run_long_context_test'

        executor._capture_test_config(
            test_function,
            '长onunder文Test',
            ([2000, 4000], 5, 500),
            '20250130_120000'
        )

        config = st.session_state._data['test_config']
        assert config['Test Type'] == '长onunder文Test'
        assert config['Context Lengths'] == '[2000, 4000]'
        assert config['Rounds'] == '5'


class TestCreateTestExecutor:
    """TestFactory函数"""

    def test_create_test_executor(self, mock_config):
        """TestCreateTest执行器Factory函数"""
        from ui.test_runner import create_test_executor

        executor = create_test_executor(mock_config)

        from ui.test_runner import TestExecutor
        assert isinstance(executor, TestExecutor)
        assert executor.config == mock_config
