"""ui.warehouse_charts 数据准备纯函数测试（不依赖 Streamlit 渲染）。"""

from __future__ import annotations

from ui.warehouse_charts import (
    build_box_figure,
    build_compare_table,
    build_engine_timeline_figure,
    build_histogram_figure,
    build_trend_data,
    build_trend_figure,
    collect_distribution_values,
    run_label,
)


def _row(**kw):
    base = {
        "date": "2026-06-01",
        "test_id": "t-1234567890",
        "model_name": "m1",
        "machine_id": "host1",
        "engine": "vllm",
        "decode_tps": 55.0,
        "ttft_s": 0.12,
        "gpu_vram_peak_gb": 70.0,
        "external_level": "internal",
    }
    base.update(kw)
    return base


# ---------- build_trend_data ----------


def test_trend_data_groups_and_sorts_by_date():
    rows = [
        _row(date="2026-06-03", decode_tps=53.0),
        _row(date="2026-06-01", decode_tps=55.0),
        _row(date="2026-06-02", decode_tps=51.0, model_name="m2"),
        _row(date="2026-06-02", decode_tps=None),  # 缺指标 → 跳过
        _row(date="", decode_tps=99.0),  # 缺日期 → 跳过
    ]
    data = build_trend_data(rows, "decode_tps", "model_name")
    assert [p[1] for p in data["m1"]] == [55.0, 53.0]  # 按日期升序
    assert [p[1] for p in data["m2"]] == [51.0]
    assert len(data) == 2


def test_trend_data_falls_back_to_unlabeled_group():
    data = build_trend_data([_row(machine_id="")], "decode_tps", "machine_id")
    assert list(data) == ["未标注"]


def test_trend_figure_has_one_trace_per_group():
    rows = [_row(model_name="a"), _row(model_name="b", decode_tps=60.0)]
    fig = build_trend_figure(rows, "decode_tps", "model_name")
    assert len(fig.data) == 2


# ---------- 对比 ----------


def test_run_label_contains_date_model_machine():
    label = run_label(_row())
    assert "2026-06-01" in label and "m1" in label and "host1" in label
    assert "t-123456" in label  # test_id 前 8 位


def test_compare_table_rows_are_metrics_cols_are_runs():
    rows = [_row(), _row(date="2026-06-02", decode_tps=60.0)]
    table = build_compare_table(rows)
    assert "decode TPS" in table
    values = list(table["decode TPS"].values())
    assert values == [55.0, 60.0]


# ---------- 分布 ----------


class _Res:
    def __init__(self, ttft=None, tps=None, concurrency_level=None, error=None):
        self.ttft = ttft
        self.tps = tps
        self.tpot = None
        self.concurrency_level = concurrency_level
        self.error = error


def test_collect_distribution_skips_errors_and_none():
    results = [
        _Res(ttft=0.1, concurrency_level=1),
        _Res(ttft=0.2, concurrency_level=1),
        _Res(ttft=None, concurrency_level=1),  # 缺值跳过
        _Res(ttft=9.9, concurrency_level=1, error="boom"),  # 错误跳过
        _Res(ttft=0.4, concurrency_level=8),
    ]
    groups = collect_distribution_values(results, "ttft")
    assert groups["并发 1"] == [0.1, 0.2]
    assert groups["并发 8"] == [0.4]


def test_histogram_and_box_return_none_on_empty():
    assert build_histogram_figure([], "t", "x") is None
    assert build_box_figure({}, "t", "y") is None
    assert build_histogram_figure([0.1, 0.2], "t", "x") is not None
    assert build_box_figure({"并发 1": [0.1]}, "t", "y") is not None


# ---------- 引擎时间线 ----------


def test_engine_timeline_none_on_empty_or_all_null():
    assert build_engine_timeline_figure([]) is None
    assert build_engine_timeline_figure([{"t": 1.0}]) is None
    fig = build_engine_timeline_figure(
        [{"t": 1.0, "gpu_cache_usage_perc": 0.5, "num_requests_running": 8}]
    )
    assert fig is not None
    assert len(fig.data) == 3  # KV / running / waiting
