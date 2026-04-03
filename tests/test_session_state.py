"""
Session State 管理模块单元Test

use conftest.py in全局 streamlit mock
"""

import sys

import pandas as pd
import pytest


@pytest.fixture
def mock_session_state():
    """Create干净 session_state mock"""
    import streamlit as st
    st.session_state._data.clear()
    return st.session_state


@pytest.fixture
def session_state_module():
    """动态Import session_state 模块"""
    if 'config.session_state' in sys.modules:
        del sys.modules['config.session_state']
    from config import session_state
    return session_state


class TestInitSessionState:
    """Test init_session_state 函数"""

    def test_init_creates_test_control_states(self, mock_session_state, session_state_module):
        """TestInitializeCreateTest控制Status变量"""
        session_state_module.init_session_state()

        assert 'test_running' in mock_session_state._data
        assert 'stop_requested' in mock_session_state._data
        assert mock_session_state._data['test_running'] == False
        assert mock_session_state._data['stop_requested'] == False

    def test_init_creates_results_data_states(self, mock_session_state, session_state_module):
        """TestInitializeCreateResultDataStatus变量"""
        session_state_module.init_session_state()

        assert 'results_df' in mock_session_state._data
        assert 'report' in mock_session_state._data
        assert 'logger' in mock_session_state._data
        assert isinstance(mock_session_state._data['results_df'], pd.DataFrame)
        assert mock_session_state._data['report'] == ""
        assert mock_session_state._data['logger'] is None

    def test_init_creates_test_config_states(self, mock_session_state, session_state_module):
        """TestInitializeCreateTestConfigureStatus变量"""
        session_state_module.init_session_state()

        assert 'test_duration' in mock_session_state._data
        assert 'test_config' in mock_session_state._data
        assert mock_session_state._data['test_duration'] == 0.0
        assert mock_session_state._data['test_config'] == {}

    def test_init_creates_latency_offset_state(self, mock_session_state, session_state_module):
        """TestInitializeCreate延迟校准Status变量"""
        session_state_module.init_session_state()

        assert 'latency_offset' in mock_session_state._data
        assert mock_session_state._data['latency_offset'] == 0.0

    def test_init_creates_file_states(self, mock_session_state, session_state_module):
        """TestInitializeCreateData文件Status变量"""
        session_state_module.init_session_state()

        assert 'current_csv_file' in mock_session_state._data
        assert 'current_log_file' in mock_session_state._data
        assert mock_session_state._data['current_csv_file'] == ""
        assert mock_session_state._data['current_log_file'] == ""

    def test_init_creates_system_info_states(self, mock_session_state, session_state_module):
        """TestInitializeCreate系统信息Status变量"""
        session_state_module.init_session_state()

        assert 'system_info' in mock_session_state._data
        assert mock_session_state._data['system_info'] == {}

    def test_init_creates_model_states(self, mock_session_state, session_state_module):
        """TestInitializeCreateModel相关Status变量"""
        session_state_module.init_session_state()

        assert 'fetched_models' in mock_session_state._data
        assert 'custom_sys_info' in mock_session_state._data
        assert mock_session_state._data['fetched_models'] == []
        assert mock_session_state._data['custom_sys_info'] == {}

    def test_init_creates_quality_test_states(self, mock_session_state, session_state_module):
        """TestInitializeCreate质量TestStatus变量"""
        session_state_module.init_session_state()

        assert 'quality_results' in mock_session_state._data
        assert 'quality_test_running' in mock_session_state._data
        assert mock_session_state._data['quality_results'] == {}
        assert mock_session_state._data['quality_test_running'] == False

    def test_init_creates_ab_comparison_states(self, mock_session_state, session_state_module):
        """TestInitializeCreate A/B 对比Status变量"""
        session_state_module.init_session_state()

        assert 'ab_result' in mock_session_state._data
        assert 'ab_comparison_running' in mock_session_state._data
        assert mock_session_state._data['ab_result'] == {}
        assert mock_session_state._data['ab_comparison_running'] == False

    def test_init_creates_test_outputs_state(self, mock_session_state, session_state_module):
        """TestInitializeCreateTest输出Status变量"""
        session_state_module.init_session_state()

        assert 'test_outputs' in mock_session_state._data
        assert mock_session_state._data['test_outputs'] == []

    def test_init_preserves_existing_values(self, mock_session_state, session_state_module):
        """TestInitialize保留已has值"""
        mock_session_state._data['test_running'] = True
        mock_session_state._data['latency_offset'] = 0.5

        session_state_module.init_session_state()

        # 已has值应该被保留
        assert mock_session_state._data['test_running'] == True
        assert mock_session_state._data['latency_offset'] == 0.5


class TestGetState:
    """Test get_state 函数"""

    def test_get_existing_key(self, mock_session_state, session_state_module):
        """TestGet存in键"""
        mock_session_state._data['existing_key'] = 'test_value'
        result = session_state_module.get_state('existing_key')
        assert result == 'test_value'

    def test_get_nonexistent_key_with_default(self, mock_session_state, session_state_module):
        """TestGetnot存in键，Returndefault值"""
        result = session_state_module.get_state('nonexistent_key', default='default_value')
        assert result == 'default_value'

    def test_get_nonexistent_key_without_default(self, mock_session_state, session_state_module):
        """TestGetnot存in键，nodefault值Return None"""
        result = session_state_module.get_state('nonexistent_key')
        assert result is None


class TestSetState:
    """Test set_state 函数"""

    def test_set_string_value(self, mock_session_state, session_state_module):
        """TestSet字符串值"""
        session_state_module.set_state('string_key', 'string_value')
        assert mock_session_state._data['string_key'] == 'string_value'

    def test_set_numeric_value(self, mock_session_state, session_state_module):
        """TestSet数值"""
        session_state_module.set_state('number_key', 42)
        assert mock_session_state._data['number_key'] == 42

    def test_set_dict_value(self, mock_session_state, session_state_module):
        """TestSet字典值"""
        test_dict = {'a': 1, 'b': 2}
        session_state_module.set_state('dict_key', test_dict)
        assert mock_session_state._data['dict_key'] == test_dict

    def test_set_list_value(self, mock_session_state, session_state_module):
        """TestSet列表值"""
        test_list = [1, 2, 3]
        session_state_module.set_state('list_key', test_list)
        assert mock_session_state._data['list_key'] == test_list

    def test_set_none_value(self, mock_session_state, session_state_module):
        """TestSet None 值"""
        session_state_module.set_state('none_key', None)
        assert mock_session_state._data['none_key'] is None

    def test_set_overwrites_existing(self, mock_session_state, session_state_module):
        """Test覆盖已has值"""
        mock_session_state._data['key'] = 'old_value'
        session_state_module.set_state('key', 'new_value')
        assert mock_session_state._data['key'] == 'new_value'


class TestResetTestState:
    """Test reset_test_state 函数"""

    def test_reset_test_running(self, mock_session_state, session_state_module):
        """TestReset test_running"""
        mock_session_state._data['test_running'] = True
        session_state_module.reset_test_state()
        assert mock_session_state._data['test_running'] == False

    def test_reset_stop_requested(self, mock_session_state, session_state_module):
        """TestReset stop_requested"""
        mock_session_state._data['stop_requested'] = True
        session_state_module.reset_test_state()
        assert mock_session_state._data['stop_requested'] == False

    def test_reset_test_duration(self, mock_session_state, session_state_module):
        """TestReset test_duration"""
        mock_session_state._data['test_duration'] = 123.45
        session_state_module.reset_test_state()
        assert mock_session_state._data['test_duration'] == 0.0

    def test_reset_does_not_affect_other_states(self, mock_session_state, session_state_module):
        """TestResetnot影响otherStatus变量"""
        mock_session_state._data['test_running'] = True
        mock_session_state._data['stop_requested'] = True
        mock_session_state._data['other_key'] = 'other_value'
        mock_session_state._data['latency_offset'] = 0.5

        session_state_module.reset_test_state()

        assert mock_session_state._data['test_running'] == False
        assert mock_session_state._data['stop_requested'] == False
        assert mock_session_state._data['other_key'] == 'other_value'
        assert mock_session_state._data['latency_offset'] == 0.5


class TestResetResults:
    """Test reset_results 函数"""

    def test_reset_results_df(self, mock_session_state, session_state_module):
        """TestReset results_df"""
        mock_session_state._data['results_df'] = pd.DataFrame({'a': [1, 2, 3]})
        session_state_module.reset_results()
        assert mock_session_state._data['results_df'].empty

    def test_reset_report(self, mock_session_state, session_state_module):
        """TestReset report"""
        mock_session_state._data['report'] = 'some report content'
        session_state_module.reset_results()
        assert mock_session_state._data['report'] == ""

    def test_reset_test_outputs(self, mock_session_state, session_state_module):
        """TestReset test_outputs"""
        mock_session_state._data['test_outputs'] = ['output1', 'output2']
        session_state_module.reset_results()
        assert mock_session_state._data['test_outputs'] == []

    def test_reset_does_not_affect_other_states(self, mock_session_state, session_state_module):
        """TestResetnot影响otherStatus变量"""
        mock_session_state._data['results_df'] = pd.DataFrame({'a': [1]})
        mock_session_state._data['report'] = 'report'
        mock_session_state._data['test_running'] = True
        mock_session_state._data['latency_offset'] = 0.5

        session_state_module.reset_results()

        assert mock_session_state._data['results_df'].empty
        assert mock_session_state._data['report'] == ""
        assert mock_session_state._data['test_running'] == True
        assert mock_session_state._data['latency_offset'] == 0.5


class TestIsTestRunning:
    """Test is_test_running 函数"""

    def test_returns_true_when_running(self, mock_session_state, session_state_module):
        """TestTest运行时Return True"""
        mock_session_state._data['test_running'] = True
        assert session_state_module.is_test_running() == True

    def test_returns_false_when_not_running(self, mock_session_state, session_state_module):
        """TestTest未运行时Return False"""
        mock_session_state._data['test_running'] = False
        assert session_state_module.is_test_running() == False


class TestIsStopRequested:
    """Test is_stop_requested 函数"""

    def test_returns_true_when_requested(self, mock_session_state, session_state_module):
        """Test请求停止时Return True"""
        mock_session_state._data['stop_requested'] = True
        assert session_state_module.is_stop_requested() == True

    def test_returns_false_when_not_requested(self, mock_session_state, session_state_module):
        """Test未请求停止时Return False"""
        mock_session_state._data['stop_requested'] = False
        assert session_state_module.is_stop_requested() == False


class TestRequestStop:
    """Test request_stop 函数"""

    def test_sets_stop_requested_to_true(self, mock_session_state, session_state_module):
        """TestSet stop_requested is True"""
        mock_session_state._data['stop_requested'] = False
        session_state_module.request_stop()
        assert mock_session_state._data['stop_requested'] == True

    def test_sets_test_running_to_false(self, mock_session_state, session_state_module):
        """TestSet test_running is False"""
        mock_session_state._data['test_running'] = True
        session_state_module.request_stop()
        assert mock_session_state._data['test_running'] == False

    def test_can_be_called_multiple_times(self, mock_session_state, session_state_module):
        """Testcan多次调用"""
        mock_session_state._data['stop_requested'] = False
        mock_session_state._data['test_running'] = True

        session_state_module.request_stop()
        session_state_module.request_stop()
        session_state_module.request_stop()

        assert mock_session_state._data['stop_requested'] == True
        assert mock_session_state._data['test_running'] == False
