import json

import pandas as pd


def test_save_and_restore_last_result_snapshot(tmp_path):
    from core.result_persistence import restore_last_results_to_session, save_last_result_snapshot

    csv_path = tmp_path / "raw_data" / "model-a" / "benchmark_results_model-a_Prefill_Stress_Test_20260503_120000.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({"session_id": ["s1"], "ttft": [1.23], "tps": [45.6]}).to_csv(
        csv_path, index=False
    )

    save_last_result_snapshot(
        csv_path=csv_path,
        test_type="Prefill Stress Test",
        model_id="model-a",
        provider="OpenAI Compatible",
        duration=12.5,
        test_config={"Token Levels": "[1024]"},
        system_info={"gpu": "RTX"},
        base_dir=tmp_path / "raw_data",
    )

    state = {}
    restored = restore_last_results_to_session(state, base_dir=tmp_path / "raw_data")

    assert restored is True
    assert state["current_csv_file"] == str(csv_path)
    assert state["current_test_type"] == "Prefill Stress Test"
    assert state["current_model_id"] == "model-a"
    assert state["current_provider"] == "OpenAI Compatible"
    assert state["test_duration"] == 12.5
    assert state["test_config"] == {"Token Levels": "[1024]"}
    assert state["system_info"] == {"gpu": "RTX"}
    assert state["results_df"].loc[0, "ttft"] == 1.23
    assert state["restored_from_csv"] is True
    assert state["restored_result_context"]["test_type"] == "Prefill Stress Test"
    assert csv_path.with_suffix(".csv.meta.json").exists()


def test_restore_latest_csv_when_last_snapshot_is_missing(tmp_path):
    from core.result_persistence import restore_last_results_to_session

    old_csv = tmp_path / "raw_data" / "old" / "benchmark_results_old_Concurrency_Test_20260503_110000.csv"
    new_csv = tmp_path / "raw_data" / "model-b" / "benchmark_results_model-b_Long_Context_Test_20260503_120000.csv"
    old_csv.parent.mkdir(parents=True)
    new_csv.parent.mkdir(parents=True)
    pd.DataFrame({"session_id": ["old"], "ttft": [9.9]}).to_csv(old_csv, index=False)
    pd.DataFrame({"session_id": ["new"], "ttft": [2.5]}).to_csv(new_csv, index=False)

    old_time = 1_700_000_000
    new_time = old_time + 100
    old_csv.touch()
    new_csv.touch()
    import os

    os.utime(old_csv, (old_time, old_time))
    os.utime(new_csv, (new_time, new_time))

    state = {}
    restored = restore_last_results_to_session(state, base_dir=tmp_path / "raw_data")

    assert restored is True
    assert state["current_csv_file"] == str(new_csv)
    assert state["current_test_type"] == "Long Context Test"
    assert state["current_model_id"] == "model-b"
    assert state["results_df"].loc[0, "session_id"] == "new"


def test_list_and_restore_specific_saved_result_with_metadata(tmp_path):
    from core.result_persistence import (
        list_saved_results,
        restore_result_file_to_session,
        save_last_result_snapshot,
    )

    base_dir = tmp_path / "raw_data"
    first_csv = base_dir / "model-a" / "benchmark_results_model-a_Concurrency_Test_20260503_120000.csv"
    second_csv = base_dir / "model-b" / "benchmark_results_model-b_Segmented_Context_Test_20260503_130000.csv"
    first_csv.parent.mkdir(parents=True)
    second_csv.parent.mkdir(parents=True)
    pd.DataFrame({"session_id": ["a"], "ttft": [1.0]}).to_csv(first_csv, index=False)
    pd.DataFrame({"session_id": ["b"], "ttft": [2.0]}).to_csv(second_csv, index=False)

    save_last_result_snapshot(
        csv_path=second_csv,
        test_type="Segmented Context Test",
        model_id="model-b",
        provider="Provider B",
        duration=33.0,
        test_config={"Segment Levels": "[1024, 2048]"},
        system_info={"engine_name": "vLLM"},
        base_dir=base_dir,
    )

    results = list_saved_results(base_dir=base_dir)

    assert [item["path"] for item in results] == [str(second_csv), str(first_csv)]
    assert results[0]["test_type"] == "Segmented Context Test"
    assert results[0]["model_id"] == "model-b"

    state = {}
    restored = restore_result_file_to_session(state, second_csv, base_dir=base_dir)

    assert restored is True
    assert state["current_test_type"] == "Segmented Context Test"
    assert state["current_model_id"] == "model-b"
    assert state["current_provider"] == "Provider B"
    assert state["test_config"] == {"Segment Levels": "[1024, 2048]"}
    assert state["system_info"] == {"engine_name": "vLLM"}
    assert state["results_df"].loc[0, "session_id"] == "b"


def test_delete_saved_result_removes_csv_metadata_and_repairs_last_pointer(tmp_path):
    from core.result_persistence import (
        delete_saved_result,
        restore_last_results_to_session,
        save_last_result_snapshot,
    )

    base_dir = tmp_path / "raw_data"
    first_csv = base_dir / "model-a" / "benchmark_results_model-a_Concurrency_Test_20260503_120000.csv"
    second_csv = base_dir / "model-b" / "benchmark_results_model-b_Prefill_Stress_Test_20260503_130000.csv"
    first_csv.parent.mkdir(parents=True)
    second_csv.parent.mkdir(parents=True)
    pd.DataFrame({"session_id": ["a"], "ttft": [1.0]}).to_csv(first_csv, index=False)
    pd.DataFrame({"session_id": ["b"], "ttft": [2.0]}).to_csv(second_csv, index=False)

    save_last_result_snapshot(
        csv_path=first_csv,
        test_type="Concurrency Test",
        model_id="model-a",
        provider="Provider A",
        duration=11.0,
        base_dir=base_dir,
    )
    save_last_result_snapshot(
        csv_path=second_csv,
        test_type="Prefill Stress Test",
        model_id="model-b",
        provider="Provider B",
        duration=22.0,
        base_dir=base_dir,
    )

    deleted = delete_saved_result(second_csv, base_dir=base_dir)

    assert deleted is True
    assert not second_csv.exists()
    assert not second_csv.with_suffix(".csv.meta.json").exists()

    state = {}
    restored = restore_last_results_to_session(state, base_dir=base_dir)

    assert restored is True
    assert state["current_csv_file"] == str(first_csv)
    assert state["current_test_type"] == "Concurrency Test"


def test_delete_last_remaining_saved_result_removes_last_pointer(tmp_path):
    from core.result_persistence import delete_saved_result, restore_last_results_to_session

    base_dir = tmp_path / "raw_data"
    csv_path = base_dir / "model-a" / "benchmark_results_model-a_Concurrency_Test_20260503_120000.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({"session_id": ["a"], "ttft": [1.0]}).to_csv(csv_path, index=False)

    deleted = delete_saved_result(csv_path, base_dir=base_dir)

    assert deleted is True
    assert not csv_path.exists()
    assert not (base_dir / ".last_result.json").exists()
    assert restore_last_results_to_session({}, base_dir=base_dir) is False


def test_restore_ignores_missing_or_empty_saved_csv(tmp_path):
    from core.result_persistence import restore_last_results_to_session

    base_dir = tmp_path / "raw_data"
    base_dir.mkdir()
    (base_dir / ".last_result.json").write_text(
        json.dumps({"csv_path": str(base_dir / "missing.csv")}),
        encoding="utf-8",
    )

    state = {}
    restored = restore_last_results_to_session(state, base_dir=base_dir)

    assert restored is False
    assert "results_df" not in state


def test_history_service_is_available_as_canonical_module(tmp_path):
    from core.results.history_service import list_saved_results

    base_dir = tmp_path / "raw_data"
    base_dir.mkdir()

    assert list_saved_results(base_dir=base_dir) == []
