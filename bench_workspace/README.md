# bench_workspace — 一次性测试脚本与产出归档

主目录（仓库根）只保留正式入口与文档化 CLI：

| 保留在根目录 | 作用 |
| --- | --- |
| `app.py` | Streamlit 主界面 |
| `live_bench.py` | 在线压测 CLI（测试时采引擎/硬件指标，`docs/环境信息采集指南.md` 记录） |
| `hw_snapshot.py` | 硬件盘点快照采集 CLI（`docs/硬件快照采集与导入.md` 记录） |
| `import_hw_snapshot.py` | 快照导入 CLI |

其余一次性测试脚本、生成的报告、数据归档、运行日志统一收纳在本目录下。

## 目录结构

```
bench_workspace/
├── scripts/   # 一次性测试/构建脚本（GLM-5.2 各轮压测、报告生成、decode steady 探针等）
├── reports/   # 生成的测试报告目录 + 对应 zip 包
├── archives/  # 原始数据归档（raw_data.zip、test-standard*、results_nvfp4 等）
├── logs/      # 运行日志（logs_*.out）
└── README.md
```

## 输出路径

脚本产出统一写回本目录,**不再散落到仓库根**:

| 脚本 | 写入位置 |
| --- | --- |
| `build_glm52_*.py`、`build_glm52_single.py` | `bench_workspace/reports/<报告名>/`（HTML + 同目录附件） |
| `run_nvfp4_benchmark.py`（`--output` 默认值） | `bench_workspace/archives/results_nvfp4/` |
| `run_nvfp4_benchmark_v2.py`、`run_gpqa_only.py` | `bench_workspace/archives/results_nvfp4_v2/` |

说明：

- 上述路径均为脚本内写死的相对路径（相对仓库根），所以**仍须从仓库根运行**（见下节）。
- `raw_data/` 是仓库既有的共享数据目录（`live_bench.py` / `core` / `ui` 共用，且被 `.gitignore` 忽略），
  保留在仓库根**未改动**：脚本仍从 `raw_data/*.csv` 读源数据，部分 `_live_*` 探针仍往 `raw_data/` 写 CSV。
  这是项目既有约定，不属于「散落到根的产物」，故不在本次整理范围内。

## 运行 scripts/ 下的脚本

这些脚本以 `from core import ...` / `from evaluators import ...` 方式引用仓库根的一方包，
并且部分脚本之间存在平级互引（如 `_mtp_cont.py` → `import _live_glm52_mtp`）。
因此**必须从仓库根目录、并显式把仓库根加入 `PYTHONPATH` 运行**：

```bash
# 从仓库根执行（推荐）
PYTHONPATH=. python bench_workspace/scripts/<name>.py

# 例
PYTHONPATH=. python bench_workspace/scripts/build_glm52_report.py
PYTHONPATH=. python bench_workspace/scripts/_live_glm52_nvfp4.py
```

说明：

- `PYTHONPATH=.` 让 `from core ...` / `from evaluators ...` 可解析（仓库根在搜索路径上）。
- 直接 `python bench_workspace/scripts/<name>.py` 时，Python 会把 `bench_workspace/scripts/`
  自动加入 `sys.path[0]`，故平级互引（`import _live_glm52_mtp`）天然可用。
- 不要用 `python -m bench_workspace.scripts.xxx`：这些脚本不是包，且会破坏平级互引。

## 归档说明

- `reports/`、`archives/`、`logs/` 均为生成物/大文件，已在 `.gitignore` 中忽略，不纳入版本管理。
- `scripts/` 可跟踪：其中 `build_report.py`、`build_showcase.py`、`export_hmtest.py`、
  `export_standard.py`、`_live_test*.py`、`_watch_full.py`、`_decode_steady_full.py` 是
  历史已跟踪的一次性脚本（经 `git mv` 迁入）；其余 `_live_glm52*`、`build_glm52*`、
  `run_*` 等为未跟踪的一次性脚本，按需自行决定是否提交。
