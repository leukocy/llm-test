"""ui.warehouse_report.build_single_test_report 单元测试（纯函数，无 Streamlit）。"""

from __future__ import annotations

from ui.warehouse_report import build_single_test_report


def _ctx(**overrides):
    base = {
        "test_type": "concurrency",
        "model_id": "DeepSeek-V3.1",
        "tester": "alice",
        "machine_id": "abc123",
        "status_detail": "passed",
        "bottleneck": "memory_bandwidth",
        "hardware_fingerprint": {
            "machine_id": "abc123",
            "cpu": {"model_name": "EPYC 9355", "sockets": 1, "cores_per_socket": 32},
            "memory": {"total_gb": 1133, "type": "DDR5", "channels": 24},
            "gpus": [
                {
                    "name": "RTX PRO 6000",
                    "vram_gb": 96,
                    "nominal_bandwidth_gbps": 1792,
                    "pcie_gen": 5,
                    "pcie_width": 16,
                }
            ],
            "cuda": {"cuda_version": "13.0", "driver": "580.159.03"},
        },
        "test_config": {"tp_size": 8, "random_seed": 42},
        "resource_monitor": {
            "peaks": {
                "gpu_util_percent": 95.0,
                "gpu_vram_gb": 70.5,
                "system_memory_gb": 120.0,
                "gpu_power_w": 1800.0,
                "gpu_temp_c": 72,
            }
        },
        "engine_metrics": {
            "engine_family": "vllm",
            "sample_count": 5,
            "peaks": {"gpu_cache_usage_perc": 0.82, "num_requests_running": 8},
            "engine_means": {"ttft_s": 0.3, "tpot_s": 0.025},
            "preemption_total": 3,
            "cache_config": {"kv_capacity_tokens": 1468000, "block_size": 16},
        },
        "client_vs_engine": {
            "client_ttft_s": 0.7,
            "engine_ttft_s": 0.3,
            "ttft_overhead_pct": 133.3,
            "verdict": "开销显著，延迟在引擎之外",
        },
        "effective_bandwidth": {
            "effective_bandwidth_gbps": 1850.0,
            "nominal_bandwidth_gbps": 1792.0,
            "bandwidth_utilization_pct": 103.2,
        },
        "gate": {
            "level": "internal",
            "gates": {
                "config_complete": True,
                "reproducible": True,
                "metrics_trustworthy": True,
                "external_reviewed": False,
            },
            "reasons": ["未经人工复核"],
        },
        "insights": ["🚀 良好", "⚠️ 偶发波动"],
        "notes": "8卡 TP",
        "next_action": "补测 128K 长上下文",
    }
    base.update(overrides)
    return base


def test_report_has_nine_sections():
    md = build_single_test_report(_ctx())
    for section in [
        "硬件指纹",
        "配置快照",
        "性能摘要",
        "资源峰值",
        "推理引擎运行时",
        "客户端 vs 引擎侧延迟对照",
        "等效带宽偏差分析",
        "可对外闸门",
        "洞察",
        "备注 / 下一步",
    ]:
        assert section in md, f"缺章节: {section}"
    assert "DeepSeek-V3.1" in md
    assert "abc123" in md
    assert "1850" in md  # 等效带宽
    assert "RTX PRO 6000" in md


def test_report_handles_missing_fields_gracefully():
    md = build_single_test_report({"test_type": "concurrency"})
    # 不抛异常，且仍包含章节
    assert "硬件指纹" in md
    assert "—" in md


def test_report_includes_gate_and_bottleneck():
    md = build_single_test_report(_ctx(bottleneck="compute_prefill"))
    assert "compute_prefill" in md
    assert "internal" in md
    assert "未经人工复核" in md  # reasons


def test_report_bandwidth_summary_uses_gap_function():
    md = build_single_test_report(_ctx())
    # summarize_gap 对高利用率会给出“上界”类描述
    assert "GB/s" in md
