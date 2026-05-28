"""Compatibility exports for saved benchmark result history."""

from core.results.history_service import (
    delete_saved_result,
    list_saved_results,
    restore_last_results_to_session,
    restore_result_file_to_session,
    save_last_result_snapshot,
)

__all__ = [
    "delete_saved_result",
    "list_saved_results",
    "restore_last_results_to_session",
    "restore_result_file_to_session",
    "save_last_result_snapshot",
]
