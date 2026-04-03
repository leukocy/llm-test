# tests/engine/test_events.py
import pytest
from engine.events import EventBus, Event, EventType


def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.LOG, lambda e: received.append(e))
    bus.emit(Event(type=EventType.LOG, data={"message": "hello"}))
    assert len(received) == 1
    assert received[0].data["message"] == "hello"


def test_progress_event():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.PROGRESS, lambda e: received.append(e))
    bus.emit(Event(type=EventType.PROGRESS, data={"completed": 5, "total": 10}))
    assert received[0].data["completed"] == 5


def test_unsubscribe():
    bus = EventBus()
    received = []
    handler = lambda e: received.append(e)
    bus.subscribe(EventType.LOG, handler)
    bus.unsubscribe(EventType.LOG, handler)
    bus.emit(Event(type=EventType.LOG, data={"message": "ignored"}))
    assert len(received) == 0


def test_control_signal():
    bus = EventBus()
    bus.request_stop()
    assert bus.is_stop_requested()
    bus.clear_control()
    assert not bus.is_stop_requested()


def test_request_pause():
    bus = EventBus()
    bus.request_pause()
    assert bus.is_pause_requested()


def test_multiple_subscribers():
    bus = EventBus()
    received_a = []
    received_b = []
    bus.subscribe(EventType.RESULT, lambda e: received_a.append(e))
    bus.subscribe(EventType.RESULT, lambda e: received_b.append(e))
    bus.emit(Event(type=EventType.RESULT, data={"value": 42}))
    assert len(received_a) == 1
    assert len(received_b) == 1


def test_subscriber_error_does_not_crash():
    """A failing subscriber should not prevent others from receiving events."""
    bus = EventBus()
    received = []

    def bad_handler(e):
        raise RuntimeError("boom")

    bus.subscribe(EventType.LOG, bad_handler)
    bus.subscribe(EventType.LOG, lambda e: received.append(e))
    bus.emit(Event(type=EventType.LOG, data={"message": "test"}))
    assert len(received) == 1


def test_emit_without_subscribers():
    """Emitting with no subscribers should not raise."""
    bus = EventBus()
    bus.emit(Event(type=EventType.ERROR, data={"error": "test"}))
