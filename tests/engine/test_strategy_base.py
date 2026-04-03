# tests/engine/test_strategy_base.py
import pytest
from engine.strategies.base import TestStrategy, register_strategy, get_strategy, list_strategies, _STRATEGY_REGISTRY
from engine.events import EventBus

@pytest.fixture(autouse=True)
def clean_registry():
    # Save original
    original = _STRATEGY_REGISTRY.copy()
    _STRATEGY_REGISTRY.clear()
    yield
    # Restore
    _STRATEGY_REGISTRY.clear()
    _STRATEGY_REGISTRY.update(original)

def test_strategy_registration():
    @register_strategy("test_dummy")
    class DummyStrategy(TestStrategy):
        display_name = "Dummy"
        description = "Test"
        icon = ":)"
        
        @classmethod
        def param_schema(cls):
            return {}
        
        def calculate_total_requests(self, params):
            return 1
            
        async def execute(self, *args, **kwargs):
            return []
            
        def csv_columns(self):
            return []

    assert "test_dummy" in _STRATEGY_REGISTRY
    assert get_strategy("test_dummy") == DummyStrategy
    
    strategies = list_strategies()
    assert "test_dummy" in strategies
    assert strategies["test_dummy"]["name"] == "Dummy"

def test_get_strategy_unknown():
    with pytest.raises(ValueError):
        get_strategy("unknown")

def test_strategy_abstract_methods():
    # Should enforce implementation
    class Incomplete(TestStrategy):
        pass
        
    with pytest.raises(TypeError):
        # Allow instantiation? No, ABC prevents it if abstract methods missing?
        # ABC mechanism works on instantiation.
        Incomplete(EventBus()) 
