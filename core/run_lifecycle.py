"""Validated lifecycle for one benchmark run.

The state belongs to a run, not to a browser session. Callers can use the
pure transition function before persisting a change, while repositories must
also compare the stored state when they write it.
"""

from enum import Enum


class RunStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class RunEvent(str, Enum):
    ENQUEUE = "enqueue"
    START = "start"
    REQUEST_PAUSE = "request_pause"
    PAUSE = "pause"
    RESUME = "resume"
    REQUEST_CANCEL = "request_cancel"
    CANCEL = "cancel"
    COMPLETE = "complete"
    FAIL = "fail"
    RECOVER = "recover"


class InvalidRunTransition(ValueError):
    """An event is not valid for the run's current state."""


_TRANSITIONS: dict[RunStatus, dict[RunEvent, RunStatus]] = {
    RunStatus.CREATED: {
        RunEvent.ENQUEUE: RunStatus.QUEUED,
        RunEvent.START: RunStatus.RUNNING,
        RunEvent.CANCEL: RunStatus.CANCELLED,
        RunEvent.FAIL: RunStatus.FAILED,
    },
    RunStatus.QUEUED: {
        RunEvent.START: RunStatus.RUNNING,
        RunEvent.CANCEL: RunStatus.CANCELLED,
        RunEvent.FAIL: RunStatus.FAILED,
    },
    RunStatus.RUNNING: {
        RunEvent.REQUEST_PAUSE: RunStatus.PAUSING,
        RunEvent.PAUSE: RunStatus.PAUSED,
        RunEvent.REQUEST_CANCEL: RunStatus.CANCELLING,
        RunEvent.CANCEL: RunStatus.CANCELLED,
        RunEvent.COMPLETE: RunStatus.COMPLETED,
        RunEvent.FAIL: RunStatus.FAILED,
    },
    RunStatus.PAUSING: {
        RunEvent.PAUSE: RunStatus.PAUSED,
        RunEvent.REQUEST_CANCEL: RunStatus.CANCELLING,
        RunEvent.CANCEL: RunStatus.CANCELLED,
        RunEvent.COMPLETE: RunStatus.COMPLETED,
        RunEvent.FAIL: RunStatus.FAILED,
    },
    RunStatus.PAUSED: {
        RunEvent.RESUME: RunStatus.RUNNING,
        RunEvent.REQUEST_CANCEL: RunStatus.CANCELLING,
        RunEvent.CANCEL: RunStatus.CANCELLED,
        RunEvent.FAIL: RunStatus.FAILED,
    },
    RunStatus.CANCELLING: {
        RunEvent.CANCEL: RunStatus.CANCELLED,
        RunEvent.FAIL: RunStatus.FAILED,
    },
    RunStatus.CANCELLED: {},
    RunStatus.COMPLETED: {},
    RunStatus.FAILED: {RunEvent.RECOVER: RunStatus.QUEUED},
}


def advance_run(status: RunStatus | str, event: RunEvent | str) -> RunStatus:
    """Apply an event or reject an invalid transition without changing state."""
    current = RunStatus(status)
    action = RunEvent(event)
    try:
        return _TRANSITIONS[current][action]
    except KeyError as exc:
        raise InvalidRunTransition(f"Cannot {action.value} a {current.value} run") from exc


def event_for_transition(status: RunStatus | str, target: RunStatus | str) -> RunEvent:
    """Find the event for a requested status change, raising if it is illegal."""
    current = RunStatus(status)
    next_status = RunStatus(target)
    for event, destination in _TRANSITIONS[current].items():
        if destination == next_status:
            return event
    raise InvalidRunTransition(f"Cannot move a run from {current.value} to {next_status.value}")
