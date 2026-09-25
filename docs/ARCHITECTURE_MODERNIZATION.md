<!-- markdownlint-disable MD013 -->

# LLM 测试平台架构升级方案

适用场景：单租户内网部署。优先保证安全、评测可信度和可恢复的测试运行，再提升吞吐和展示体验。

## 1. 现状与可验证的问题

| 方向 | 当前实现 | 主要问题 |
| --- | --- | --- |
| 执行 | `core/benchmark_runner.py` 约 4900 行，异步 `httpx` 发请求；Streamlit 创建后台线程执行 | 场景编排、结果写入、数据库、监控和 UI 回调集中在同一个类；难以独立压测或替换执行器 |
| 状态 | `config/session_state.py`、`core/cancel_state.py`、`ui/test_control_panel.py`、`test_runs.status` 各自保存状态 | 暂停、停止、完成可在不同层直接改写；浏览器刷新后的会话状态不能充当持久运行状态 |
| 数据 | 运行时逐条追加 CSV、逐条提交 SQLite；历史查询由仓库模块提供 | 磁盘写入位于请求处理路径；需要测量高并发下的写入延迟、锁等待和丢失风险 |
| 展示 | Streamlit 多模块页面，结果刷新时将整个列表重建成 DataFrame | 结果增长时每次全量重建和渲染；交互、导航、响应式布局的控制空间有限 |

`pyproject.toml` 已声明 FastAPI 和 SQLAlchemy 依赖，但当前主程序没有 FastAPI 服务，实际数据库连接层使用 `sqlite3`。项目已有 `ui/`、`core/` 和 `evaluators/` 等模块；缺口在依赖边界和单一状态来源，并非缺少目录。

## 2. 目标结构

```text
React + TypeScript 控制台 ── HTTP / 事件流 ── FastAPI 查询与控制层
                                                  │
                                                  v
                                      RunService + 持久状态机
                                                  │
                                      任务表 / 租约 / Worker
                                                  │
                                                  v
                 场景规划 → 异步请求引擎 → 批量结果写入 → 统计投影
                                                   │
                                    SQLite WAL / 后续可选 PostgreSQL
                                                   │
                                      版本化报告与原始产物
```

各层职责：

1. **领域层**：运行状态、场景定义、结果与指标口径。纯 Python，不导入 Streamlit、数据库或 HTTP 框架。
2. **执行层**：规划负载、调用 provider、收集逐请求事件；通过 `ResultSink` 写入，不直接操作页面。
3. **应用层**：创建、排队、取消、暂停、恢复运行；以运行 ID 保证命令幂等并记录状态事件。
4. **持久层**：条件更新状态、批量保存结果、查询分页、版本化报告产物。状态变更使用比较并交换，防止旧 worker 覆盖新状态。
5. **接口层**：只接受请求、鉴权、校验和返回结果；长时间评测由独立 worker 执行。
6. **展示层**：消费接口提供的摘要、分页明细和报告，不直接导入 runner。设计系统统一颜色、字号、图表尺度、单位、异常与置信区间标记。

已有的 [数据展示分层设计](数据展示平台-分层架构设计.md) 对 Probe、Driver、Warehouse、Display 的划分可继续复用。本方案补齐控制面、运行状态和前后端边界。

## 3. 运行状态约束

新运行的目标路径为 `created → queued → running → completed/failed`。控制路径包括 `running → pausing → paused → running` 和 `running/paused → cancelling → cancelled`。终态不再转回 `running`；重测必须创建新的运行 ID，并记录其与原运行的关系。

本次已在 `core/run_lifecycle.py` 增加事件驱动的合法转换，在 `TestRun` 模型和 `TestRunRepository` 接入。仓库状态写入带上旧状态条件，旧 worker 不能在取消后把运行改为完成。当前 UI 的会话标志和进程级取消信号仍是过渡适配，下一步需要统一经 `RunService` 发命令。现有 runner 的暂停/停止路径还可能把持久记录记为完成；迁移时必须以此作为阻断验收项。

## 4. 性能改造顺序

1. **先测量**：固定模型、并发、输入输出 token、结果条数，记录请求吞吐、事件循环阻塞、SQLite 写入耗时、UI 刷新耗时和报告生成耗时。区分服务端模型性能与平台自身开销。
2. **控制实时刷新**：本次将全量 DataFrame 重建限制在最多每 0.5 秒一次，结束时强制刷新。下一步改为追加式摘要和有界预览，避免长测试不断重传全部记录。
3. **移走同步写入**：结果进入有界队列，由专用写入器批量提交；写入失败要有可恢复的落盘缓冲与明确错误状态。不能以无限队列换吞吐。
4. **拆分执行进程**：API 进程不执行评测；worker 按租约领取任务，定期心跳，超时任务可恢复。按端点和全局并发设置上限。
5. **查询与报告**：历史明细服务端分页、筛选、聚合；报告在完成时生成并按数据版本缓存。数据量和并发测量超过 SQLite 适用范围时再迁移 PostgreSQL。

任何“高性能”声明都应附测量条件、基线和回归阈值。目前没有足够证据承诺具体吞吐倍数。

## 5. 界面与报告

新控制台建议使用 React + TypeScript，以组件和设计令牌统一概览、运行详情、模型比较、数据质量、报告五类视图。图表同时显示单位、样本数、P50/P95/P99、错误率、置信区间或“不足以推断”的说明；所有比较先检查模型配置、硬件、数据集版本和口径可比性。页面需覆盖桌面与窄屏，并允许键盘操作和导出可复核的原始数据。既有 Streamlit 页面可在 API 建立期间继续使用。

## 6. 交付顺序与验收

| 阶段 | 交付 | 验收依据 |
| --- | --- | --- |
| A：基础约束 | 状态机、条件状态写入、UI 刷新限频 | 非法终态回退被拒绝；旧 worker 无法覆盖取消；结束展示完整结果 |
| B：执行解耦 | 场景规划、请求引擎、批量 `ResultSink`、独立 worker | 无 Streamlit 导入的 headless 测试；中断恢复；相同数据得到相同指标 |
| C：控制接口 | FastAPI 运行命令、查询、事件流、内网认证 | 刷新浏览器后按运行 ID 恢复；重复命令不会创建重复任务；权限测试通过 |
| D：新展示 | React 控制台、科学图表、报告与导出 | 视觉评审、窄屏与键盘操作、图表口径测试、报告与原始数据一致 |

上线前还要进行固定工作负载的吞吐与延迟回归、异常注入、数据库备份恢复演练，以及暂停/取消/重试的端到端测试。阶段 A 是本分支的实现范围，其余阶段为后续工程任务。

## 参考文档

- [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)：会话状态依赖 WebSocket，刷新页面会重置。
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)：重计算任务适合独立进程/任务队列。
- [FastAPI Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)：接口模块化组织。
- [React Managing State](https://react.dev/learn/managing-state)：减少重复 UI 状态并明确状态归属。
