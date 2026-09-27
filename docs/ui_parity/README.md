# 首次提交界面功能账本

<!-- markdownlint-disable MD013 -->

## 基线与证据

本账本的唯一历史基线是仓库首次提交
[`2e1ff546194027d4469d183498217ef309c43c97`](https://github.com/leukocy/llm-test/tree/2e1ff546194027d4469d183498217ef309c43c97)
（2026-05-28）。对照对象是编写本账本时的 `master`：`4f605b5`。原版入口为
`app.py`，现版入口为 `frontend/src/App.tsx`。这里记录的是源码证据；未在旧版
依赖环境中重放每一次点击，不能把“源码有控件”直接视为“当时可正常操作”。

| 证据文件 | 内容 | 条目数 |
|---|---|---:|
| [`first_commit_controls.csv`](first_commit_controls.csv) | 每个原始控件调用点一行，含原始标签、选项表达式、默认值、边界、显示条件、处理代码位置与源码链接 | 297 |
| [`first_commit_dynamic_choices.json`](first_commit_dynamic_choices.json) | 从首次提交源码展开的服务商、模型、Tokenizer、测试类型、数据集组、分段策略、扰动类型与日志级别 | — |
| [`first_commit_outputs.csv`](first_commit_outputs.csv) | 每个图表、表格、指标、代码/JSON 输出调用点一行 | 202 |
| [`first_commit_html_actions.csv`](first_commit_html_actions.csv) | `st.*` 控件之外的 HTML 下载/帮助链接 | 12 |
| [`extract_legacy_controls.py`](extract_legacy_controls.py) | 从上述固定提交重新生成四份机器可读文件 | — |

运行 `python docs/ui_parity/extract_legacy_controls.py` 可重新生成账本。所有 CSV
的 `source_url` 指向固定提交中的精确行；`id` 是本账本内的稳定检索号。标签或
选项若由变量生成，CSV 保留原始表达式与最近的赋值位置，静态来源展开在 JSON；
用户创建的预设、API 返回的模型、历史结果等运行时选项没有预先固定的完整值。

控件调用点中，194 个位于主入口接入的源码路径，68 个只写在未接入主入口的
模块/函数中，35 个位于主入口有路由但首次提交被代码错误阻断的质量评估路径。
其中有 80 个普通按钮、10 个下载按钮；80 个普通按钮都记录了对应处理代码行。
这三类分别标记为 `app_routed_source`、`source_only_not_routed_from_app`、
`routed_but_blocked_in_first_commit`。一个循环中的调用点可能渲染多个选项，
一个条件调用点也可能根本不显示，因此这些数字不是屏幕上同时出现的按钮数。

## 首次提交的实际导航

首次提交侧栏的基础测试类型按顺序为：Concurrency Test、Prefill Stress Test、
Segmented Context Test、Long Context Test、Concurrency-Context Matrix Test、
Custom Text Test、All Tests、Stability Test、Batch Test。若对应模块能导入，
再追加 Model Quality Test、A/B Model Comparison、Advanced Evaluation。选择器在
运行或暂停时禁用。[原始选择器](https://github.com/leukocy/llm-test/blob/2e1ff546194027d4469d183498217ef309c43c97/ui/sidebar.py#L464)

主流程是：侧栏设置受测接口与测量参数 → 选择测试类型 → 侧栏或主区启动测试 →
进度控制 → 结果表、图表、文字报告与导出 → 日志查看。侧栏底部另有历史结果、
配置预设和报告硬件信息。下表给出每一类功能的原始控件 ID 与现版源码对照。

## 功能逐项对照

“部分”表示现版存在相关能力，但原始入口、选项、默认值、操作步骤或导出格式
不等价；“有”只表示对应操作的主要能力在现版源码中存在，不表示视觉体验已验收。

| 原始功能与控件 ID | 首次提交中的行为 | 现版位置与结论 |
|---|---|---|
| 初次使用引导 `IC-0123`～`IC-0132` | 快速说明、逐步上一步/下一步/完成/跳过、帮助入口、功能亮点与 FAQ | [App](../../frontend/src/App.tsx) 没有等价引导；**缺失** |
| 服务商与模型 `IC-0157`～`IC-0162` | 11 个内置服务商、自定义 URL/密钥；刷新 `/models`，从预设或返回列表快选模型，亦可手填覆盖 | [API 设置](../../frontend/src/pages/ApiSettings.tsx) 已补齐初版服务商模板、17 项模型快选、已保存端点的模型列表刷新及手填覆盖；新增端点需先保存才能在线刷新；**已恢复主流程** |
| 延迟与复现参数 `IC-0163`～`IC-0168` | 一键 Probe 写入 RTT 偏移；偏移 0～5 秒，模板 token 开销、跳过首 token、固定种子 42 | [API 设置](../../frontend/src/pages/ApiSettings.tsx) 把生成耗时与独立 GET `/models` 参考耗时分开；可选择后者带入 [新建测量](../../frontend/src/pages/NewRun.tsx) 的偏移，明确包含服务商处理时间；开销、跳过首 token、种子仍可编辑；**已恢复可信口径下主流程** |
| Tokenizer `IC-0153`～`IC-0155`、`IC-0169`～`IC-0173` | 三种计数方式；自动映射 HF 模型、缺失项批量/单项下载、文本 token 计数器 | [新建测量](../../frontend/src/pages/NewRun.tsx) 已可看自动映射/本地状态、快选已安装项、用本地 tokenizer、参考编码或字符数计数；容器内持久化下载尚未恢复；**部分** |
| 测试类型 `IC-0174` | 9 个基础项及 3 个条件项，单个下拉切换整页 | [场景卡](../../frontend/src/constants.ts) 与独立页面覆盖大部分；[批量测量](../../frontend/src/pages/Batch.tsx) 已有 All Tests 三阶段入口；**已恢复主流程** |
| 并发阶梯 `IC-0202`～`IC-0209` | 级别多选、自定义级别、每级轮数、目标输入长度、最大输出；侧栏 S 与主区 M 是同一启动动作 | [新建测量](../../frontend/src/pages/NewRun.tsx) 支持同类参数，现版并发上限 128，原选项含 256～1024；**部分** |
| Prefill `IC-0210`～`IC-0217` | 输入 token 长度多选/自填、每级请求数、`max_tokens=1` 隔离开关、最大输出、双位置启动 | [新建测量](../../frontend/src/pages/NewRun.tsx) 已有长度快选/自填与隔离开关；单级长度上限 131072；**已恢复安全范围内主流程** |
| 分段上下文 `IC-0218`～`IC-0228` | Custom/Progressive/Rapid Growth/Fine-grained 策略、每段请求、累计前缀、轮数、每轮独立提示、并发、最大输出 | [新建测量](../../frontend/src/pages/NewRun.tsx) 已恢复三组策略快选，自定义仍可直接编辑；**已恢复主流程** |
| 长上下文 `IC-0229`～`IC-0235` | 预设长度多选/自填、每级轮数、最大输出、双位置启动 | 现版有同类测量，但预设勾选与原范围未保留；**部分** |
| 并发×上下文矩阵 `IC-0236`～`IC-0245` | 两维预设多选/自填、每组合轮数、最大输出、预热开关、双位置启动 | 现版有矩阵测量及预热开关，原始预设和数值范围不同；**部分** |
| 自定义文本 `IC-0246`～`IC-0253` | 上传 TXT 作为正文、后缀指令、并发级别、轮数、最大输出、防缓存噪声、双位置启动 | [新建测量](../../frontend/src/pages/NewRun.tsx) 已恢复 UTF-8 TXT 导入，正文可继续编辑，并保留后缀/防缓存；**已恢复主流程** |
| 一键运行全部 `IC-0254`～`IC-0265` | 一次配置并依次跑并发、Prefill、长上下文三阶段 | [批量测量](../../frontend/src/pages/Batch.tsx) 已恢复三阶段一键载入、逐项调参/校验和串行提交；旧版超 131072 token 档位受当前安全上限限制；**已恢复安全范围内主流程** |
| 稳定性 `IC-0266`～`IC-0272` | 并发、时长（默认 60 秒）、最大输出、输入长度、双位置启动 | 现版有同类测量，默认 120 秒，参数入口不同；**部分** |
| 运行控制 `IC-0195`～`IC-0201` | 暂停、恢复、停止、进度历史、可恢复测试列表与恢复按钮 | [运行详情](../../frontend/src/pages/Detail.tsx) 可取消与查看状态/日志，没有暂停、恢复和旧式断点恢复入口；**部分** |
| 运行中的输出预览 `IC-0281`、`IO-0201`～`IO-0202` | 逐步更新结果表，展开查看最新一次模型输出 | [运行详情](../../frontend/src/pages/Detail.tsx) 可主动展开最近一次已落库输出，最多显示 4,000 字符；逐请求表继续滚动更新，不默认传输输出正文；**已恢复主流程** |
| 结果与报告 `IC-0133`～`IC-0152`、`IO-*` | 格式化表、原始表、完整 CSV、分测试类型图表与解释、Markdown 报告；另有 HTML/PNG 下载链接 | [运行详情](../../frontend/src/pages/Detail.tsx) 已有统计、CSV/JSON/HTML 与 Markdown；静态 PNG 入口尚缺；**部分** |
| 运行日志 `IC-0113`～`IC-0122` | 级别多选、搜索、显示指标、文本/表格/统计三页、JSON/TXT/CSV 下载 | [运行详情](../../frontend/src/pages/Detail.tsx) 已有级别筛选、搜索、指标显示、文本/表格/统计三视图与服务端完整 JSON/TXT/CSV 导出；新任务保留结构化指标，旧日志仅保留已有字段；**已恢复主流程** |
| 历史 CSV `IC-0175`～`IC-0179` | 选保存结果、加载并重绘报表、确认后删除原始 CSV/元数据 | [运行记录](../../frontend/src/App.tsx) 保存作业并查看详情；旧 CSV 回载到测试页面的入口没有对应；**部分** |
| 保存测试配置 `IC-0180`～`IC-0187` | 选择/应用/删除预设，给当前配置命名并保存说明 | [新建测量](../../frontend/src/pages/NewRun.tsx) 能保存、应用、删除方案，端点/参数含义与原版不同；**部分** |
| 手填报告环境信息 `IC-0188`～`IC-0194` | Processor、Mainboard、Memory、GPU、System、Engine 六项 | 现版有自动采集与仓库元数据，但没有同位置的六项报告手填表；**部分** |
| 批量测试 `IC-0053`～`IC-0080` | 批次名称/说明、并行开关/上限、失败即停；每项独立 API/模型、启停；JSON 配置导入/保存、运行/停止、结果及历史 | [批量测量](../../frontend/src/pages/Batch.tsx) 已支持名称/说明入库、子任务启停、逐项端点/模型、样本计划、本地 JSON 导入/保存与整批停止；提交在同一事务内完成，运行记录可浏览最近批次及其子任务；并行与失败即停尚缺；**部分** |
| 高级工具 `IC-0040`～`IC-0052` | 按数据集类型解析答案、重复答题稳定性面板、文本扰动演示 | [高级评估](../../frontend/src/pages/Advanced.tsx) 有规则解析与推理评分；旧稳定性按钮本来仅提示，文本扰动演示无同等入口；**部分** |
| 质量评估 `IC-0014`～`IC-0031`、`IC-0137`～`IC-0146` | 源码设计了 18 个数据集选择、抽样模式、few-shot、Judge/缓存、质量报告 | 首次提交的导入错误使该页面在进入参数面板前返回；现版有质量作业及样本分析，**不能把旧版源码选项当作曾可用的功能** |
| 双模型在线对比 `IC-0032`～`IC-0039` | 源码设计了直接设置模型 B 并运行同一数据集 | 首次提交受同一导入错误阻断；现版 [模型对比](../../frontend/src/pages/Compare.tsx) 对已完成作业做离线配对检验，工作流不同 |

## 原始选项与默认值索引

以下是用户最常改的配置。其余每个控件的原始 `min_value`、`max_value`、
`step`、`index`、`help`、`disabled` 和条件显示代码，均在控件 CSV 中逐行保存。

| 页面 | 首次提交中的主要选项与默认值 |
|---|---|
| API | 11 个内置服务商及 URL、17 个模型预设在动态选项 JSON；API 返回模型与用户自定义项会追加到选择器。Token 计数：HuggingFace / API usage / Character fallback。 |
| 并发 | 级别 `1,2,4,8,16,32,64,128,256,512,1024`，默认 `1,2`；轮数 `1`；输入 `64`；最大输出 `512`。 |
| Prefill | 长度 `1024,2048,4096,8192,16384,32768,65536,130000,260000,520000,1000000`，默认 `4096,8192,16384,32768,65536,130000`；每级 `1` 请求；隔离开关默认开；最大输出 `512`。 |
| 分段 | 策略 Custom、Progressive、Rapid Growth、Fine-grained（各级别见 JSON）；默认自填 `2000,8000,20000,40000,60000`；累计开、每轮独立关、轮数/并发各 `1`、最大输出 `512`。 |
| 长上下文 | 长度与 Prefill 相同，默认 `4096,8192,16384,32768,65536,130000`；每级 `1` 轮；最大输出 `512`。 |
| 矩阵 | 并发 `1,2,4,8,16,64,128,256,512,1024` 默认 `1,2`；上下文 `1024,2048,4096,8192,16384,32768,65536,260000,520000,1000000` 默认 `1024,4096,16384,65536`；每组合 `1` 轮；最大输出 `256`；预热默认开。 |
| 自定义文本 | TXT 文件、后缀默认 `Please summarize the above content.`；并发 `1,2,4,6,8,16` 默认 `1,2,4`；每级 `1` 轮；最大输出 `512`；防缓存默认开。 |
| All Tests | 并发 `1,2`、轮数 `1`、输入 `64`、输出 `512`；Prefill `20000,40000`、每级 `1`、输出 `1`；长上下文 `4096,8192,16384,32768,65536,130000,260000,520000,1000000`、每级 `1`、输出 `512`。 |
| 稳定性 | 并发 `1`、时长 `60` 秒（最小 `10`）、最大输出 `512`、输入 `64`。 |
| 质量评估源码 | 基础/推理/代码/特殊四组共 18 个数据集，MMLU 和 GSM8K 默认勾选；标准/思考/代码三种模型；100/500/自定义/全量抽样；few-shot `0～10`，普通模型默认 `5`、思考模型 `0`；温度 `0`、输出 `8192`、并发 `4`、Judge 关、缓存开。该入口首次提交时被阻断。 |
| 高级分析 | Parser 数据集类型 `auto,mmlu,gsm8k,math500,humaneval,gpqa,truthfulqa,longbench`；重复测试 `2～10` 次默认 `5`、稳定阈值 `0.5～1.0` 默认 `0.8`；文本扰动 10 种类型见 JSON。重复测试按钮在首次提交中仅显示提示，没有执行测试。 |
| 日志 | DEBUG、INFO、SUCCESS、WARNING、ERROR、CRITICAL；可搜索与切换文本/表格/统计视图。 |

## 源码存在但首次提交未接入或被阻断的项目

- `app.py` 中旧的“Config Presets”和“Custom Config Management”函数包含
  `IC-0001`～`IC-0013`，但 `main()` 没有调用；当时实际使用的是侧栏底部
  `IC-0180`～`IC-0194`。
- `ui/comparison_page.py`、`ui/dataset_manager.py`、
  `ui/evaluation_dashboard.py`、`ui/history_browser.py`、
  `ui/thinking_components.py`、`ui/dashboard_components.py` 的入口没有从
  `app.py` 或其已调用页面接入。它们的控件/输出仍保留在 CSV，并标为源码项。
- `config/auth.py` 中的登录页 `IC-0278`～`IC-0280`，以及
  `utils/test_config_manager.py` 中另一套预设/导入导出界面
  `IC-0282`～`IC-0297`，首次提交也没有接入 `app.py`；不能把它们当作旧版
  的可用功能。`utils/helpers.py` 两处 HTML 下载链接同样没有调用者。
- [原始高级面板](https://github.com/leukocy/llm-test/blob/2e1ff546194027d4469d183498217ef309c43c97/ui/advanced_panels.py#L31)
  尝试从 `evaluators` 导入 `EVALUATOR_REGISTRY`、`GSM8KEvaluator`、
  `MMLUEvaluator`，但[原始包入口](https://github.com/leukocy/llm-test/blob/2e1ff546194027d4469d183498217ef309c43c97/evaluators/__init__.py#L31)
  没有导出这些名字，质量评估无法进入参数面板；A/B 页面同样依赖该结果。
  此外[质量面板](https://github.com/leukocy/llm-test/blob/2e1ff546194027d4469d183498217ef309c43c97/ui/advanced_panels.py#L226)
  在给 `available_datasets` 赋值前先使用它。记录这些是为了保留最初设计，
  不是宣称这些按钮曾可正常使用。
- 旧“Consistency Test”的开始按钮只显示“先配置 API”的提示，未发起模型请求；
  旧批量页“Add from Presets”也只有标题，没有实际添加动作。两处均不能计为
  现版丢失的可运行功能。

## 后续恢复的验收方式

恢复功能时逐个引用 `IC-*` / `IH-*` / `IO-*`，记录现版入口、选项、默认值、
提交的数据、运行结果与导出格式；把“可点到”和“真正完成动作”分开验收。
优先补齐主入口实际可用而现版缺失的模型快选、Tokenizer 工具、All Tests、
暂停/恢复、TXT 上传、批量配置和日志筛选/导出，再处理仅在初版源码中设想的功能。
每批功能完整后再统一跑 CI，避免每恢复一个按钮就触发整套流水线。
