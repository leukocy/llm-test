"""Tests for CSV persistence."""

import os
import csv
import tempfile
import pytest
from engine.persistence.csv_writer import CSVWriter


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def writer(tmp_dir):
    return CSVWriter(output_dir=tmp_dir)


def test_write_results_basic(writer, tmp_dir):
    results = [
        {"session_id": 1, "ttft": 0.1, "tps": 50.0, "error": None},
        {"session_id": 2, "ttft": 0.2, "tps": 45.0, "error": None},
    ]
    
    filepath = os.path.join(tmp_dir, "test_output.csv")
    writer.write_results(filepath, results)
    
    assert os.path.exists(filepath)
    
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 2
    assert rows[0]["session_id"] == "1"
    assert rows[0]["ttft"] == "0.1"


def test_write_results_with_column_ordering(writer, tmp_dir):
    results = [
        {"tps": 50.0, "session_id": 1, "ttft": 0.1},
    ]
    columns = ["session_id", "ttft", "tps"]
    
    filepath = os.path.join(tmp_dir, "ordered.csv")
    writer.write_results(filepath, results, columns=columns)
    
    with open(filepath, encoding="utf-8") as f:
        header = f.readline().strip()
    
    assert header == "session_id,ttft,tps"


def test_write_results_empty(writer, tmp_dir):
    filepath = os.path.join(tmp_dir, "empty.csv")
    writer.write_results(filepath, [])
    # Should not create a file for empty results
    assert not os.path.exists(filepath)


def test_append_result(writer, tmp_dir):
    run_id = "test_run"
    
    writer.append_result(run_id, {"session_id": 1, "ttft": 0.1})
    writer.append_result(run_id, {"session_id": 2, "ttft": 0.2})
    writer.close(run_id)
    
    # Find the output file
    files = os.listdir(tmp_dir)
    assert len(files) == 1
    
    filepath = os.path.join(tmp_dir, files[0])
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 2
    assert rows[0]["session_id"] == "1"
    assert rows[1]["session_id"] == "2"


def test_append_result_with_columns(writer, tmp_dir):
    run_id = "col_test"
    columns = ["session_id", "ttft", "tps"]
    
    writer.append_result(run_id, {"session_id": 1, "ttft": 0.1, "tps": 50.0, "extra": "ignored"}, columns=columns)
    writer.close(run_id)
    
    files = os.listdir(tmp_dir)
    filepath = os.path.join(tmp_dir, files[0])
    
    with open(filepath, encoding="utf-8") as f:
        header = f.readline().strip()
    
    assert header == "session_id,ttft,tps"


def test_get_filepath(writer, tmp_dir):
    fp = writer.get_filepath("abc123", "concurrency")
    assert "concurrency_abc123_" in fp
    assert fp.endswith(".csv")
    assert fp.startswith(tmp_dir)
