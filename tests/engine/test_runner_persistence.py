"""A failed per-request write must not be replaced by a duplicate later result."""

from types import SimpleNamespace
from unittest.mock import Mock

from core.benchmark_runner import BenchmarkRunner


def test_batch_flush_retries_the_failed_middle_observation():
    manager = Mock()
    manager.save_result.side_effect = [
        SimpleNamespace(id=11),
        RuntimeError("transient write failure"),
        SimpleNamespace(id=13),
    ]
    manager.results.count.return_value = 2
    manager.save_results_batch.return_value = 1

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner._db_run = SimpleNamespace(id=1)
    runner._persisted_result_ids = set()
    runner.results_list = [{"request_index": index} for index in range(3)]
    runner._get_db_manager = lambda: manager

    for result in runner.results_list:
        runner._save_result_to_db(result)
    runner._batch_save_results_to_db()

    manager.save_results_batch.assert_called_once_with(runner._db_run, [runner.results_list[1]])
    assert len(runner._persisted_result_ids) == 3
