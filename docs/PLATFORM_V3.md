# LLM Test 平台控制面（单租户内网）

<!-- markdownlint-disable MD013 -->

## 技术结构

```text
浏览器 ── React + TypeScript 控制台
  │                 │
  │ Bearer token    │ 同源 HTTP
  v                 v
FastAPI 控制 API ── SQLite WAL（任务、状态、审计、逐请求结果）
                       ^
                       │ 租约领取与心跳
                  独立 worker ── BenchmarkRunner / QualityEvaluator ── 预设模型端点
                       │
                       └── results/jobs/<job-id>/（CSV、质量报告）
```

控制面位于 `server/`，界面位于 `frontend/`；任务执行不依赖浏览器会话。一个 worker 进程串行执行一个任务；多个 worker 容器可用同一台主机上的 SQLite WAL 文件抢占任务。SQLite 数据卷必须是本机可靠文件系统，不能放在不保证锁语义的网络共享目录。

### 已覆盖的测量

- 并发阶梯、输入长度、分段 prefill、长上下文、吞吐矩阵、稳定性、自定义提示词。
- 质量评估调用已有 evaluator 注册表；数据来源指纹和逐样本结果保留在质量报告中。
- 提交参数使用严格版本化校验，默认每个有限规模任务最多 1000 次请求，稳定性测试最长一小时。端点需先在认证后的“受测 API 设置”页面保存，再供任务选择；任务提交体不能携带任意 URL 或 API key。

### 已迁移的工作流

- **测试方案**：创建、更新、删除并复用参数方案。方案保存在 `control_presets`，包含名称、最多 500 字符的说明、测试参数和生成设置；说明可随方案应用、更新和另存为，旧方案的说明默认为空。沿用任务提交的严格参数校验；方案不含密钥。
- **使用引导**：总览首次使用提示和常驻“帮助与引导”入口，六步引导支持上一步、下一步、完成、跳过与重新开始，另有快速参考、功能亮点和 FAQ。完成状态保存在当前浏览器 `localStorage`，跳过仅保留当前标签页；引导不会阻止创建任务。
- **初版参数**：七种基础性能测试可一键载入首次提交的默认工作负载，保留现有标准与深入测量配置。旧版单样本设置适于复现操作，正式比较仍需增加样本和预热；并发与输入长度仍受当前 128 / 131,072 上限约束。切换自定义文本的测量强度会保留已填写或导入的正文。
- **报告环境信息**：创建测量和批量子任务均可填写 Processor、Mainboard、Memory、GPU、System、Engine 六项，每项最多 240 字符。明确选择信息对应受测模型服务器、测试客户端或未明确对象，内容随方案与批量 JSON 保存，并进入性能、质量和鲁棒性的 JSON / HTML / Markdown 报告。字段固定标记为 `user_reported`，不会覆盖执行端自动采集的硬件快照和机器身份；“环境信息”页面展示的是当前 API 主机，远程受测模型可能位于其他机器。
- **受测 API 设置**：内置常用服务商地址和初版模型快选，支持新增、编辑、删除、在线模型列表与一次低输出连接检测。参考网络耗时通过单独的 `/models` 响应首部测量，明确含服务商处理时间，不能视为纯网络 RTT。凭证经 Fernet 加密后存入 `control_endpoints`，列表和任务响应均不回传明文。待执行或运行中的任务引用端点时，禁止修改或删除该端点。
- **基础三项与批量**：可一键载入并发、Prefill、长上下文三阶段；每个子任务可选不同端点和模型、单独启停，提交前分别校验启用项的工作负载和样本预算。批次名称、说明、执行策略和子任务在同一事务中落库，重复提交受幂等摘要约束；默认按序串行执行，可开启 2～8 项并行和失败即停。并行上限由数据库事务控制，即使多个 worker 同时领取任务也不会超限；实际并行数取决于可用 worker。批量配置可导入/导出本地 JSON；批次历史可查看元数据并停止尚未结束的子任务。
- **Tokenizer 与文本**：创建测量可查看模型自动映射、本地安装状态，选择已登记的本地 tokenizer，或用本地、参考 tiktoken 编码及字符数对文本计数。可下载匹配项、单项或全部缺失项，并查看排队、字节/文件进度、取消和失败重试；下载任务保存在 `tokenizer_installs`，浏览器刷新不会丢失。自定义提示词可从 UTF-8 TXT 导入。
- **数据仓库**：按模型、硬件、类型、状态、对外等级和文本检索历史记录；复测链默认展示最新版本。历史表可展开完整测量字段，矩阵支持最新值或最大观测值，另有硬件盘点与扩展效率。`hwInventory` 和 `hmTest` 模板可导出 CSV/JSON。
- **历史 CSV**：独立入口 `/history` 可选择保存文件、加载并重绘统计与图表、预览逐请求记录、确认后删除 CSV 与配套元数据；删除保留回收副本。支持上传初版或当前平台原始性能 CSV 及可选 `.csv.meta.json`，刷新后保持已加载的文件。提供五种性能指标图形、PNG、HTML、Markdown、JSON 和安全 CSV；不创建测量任务，不调用模型 API。详见下文“历史 CSV 的保存与报告”。
- **质量诊断**：按数据集查看准确率与 Wilson 区间、类别表现、评分方式、失败归因、数据指纹和逐样本输入/响应。全部错误样本可导出带有电子表格公式防护的 CSV。

仓库 API 每次最多扫描最新的 1000 条匹配记录；界面同时显示匹配数、扫描数和无效记录数。查询超出窗口或包含无法解析的记录时，服务端返回 409 并拒绝导出，避免生成不完整的文件。历史指标中的非有限数值显示为空值。CSV 错误样本包含原始提示词与响应，仅通过已认证的控制 API 下载。

### 生命周期与失败处理

任务状态遵循 `created → queued → running → completed/failed`；运行中取消经 `cancelling → cancelled`。事件追加到 `job_events`。worker 领取任务时获得租约，每两秒续约，并同步数据库中的测量进度。单条写库失败后，runner 按未写入的具体观测值补写，避免用数据库总行数推断已写入前缀。完成性能任务前，worker 核对测量记录归属、终态和逐请求落库数量；不一致则把任务标记为失败，并保留已落库样本供诊断。失联且租约过期的测量标记为 `WORKER_LOST`，不会自动重试并混入新的样本；要重测，请创建新任务。

同一 `Idempotency-Key` 与相同参数重复提交只返回原任务；同键不同内容返回 409。任务 ID 贯穿 `control_jobs.job_id` 和 `test_runs.test_id`。任务完成或取消后，已有逐请求结果仍可查询。

有限性能测试支持 `running → pausing → paused → running`。在详情页点击“暂停运行”后，当前请求组会先完成并保存，再进入暂停；点击“继续运行”沿用当前执行位置，不重复请求或预热。支持并发、Prefill、分段上下文、长上下文、矩阵、自定义文本和数据集性能测试。矩阵和自定义文本保持一个连续负载单元完整，只有单元之间能暂停。最后一个请求组完成时，任务可能直接完成。按时长测试的稳定性、质量及鲁棒性评估暂不支持暂停。

暂停时 worker 保留租约并续约，占用一个批次执行槽位；关闭浏览器不影响继续或取消。暂停期间重启 worker 会丢失内存中的执行位置，过期租约会标为 `WORKER_LOST`，不能从该位置恢复。暂停、恢复、取消均通过认证后的 `POST /api/v1/jobs/{job_id}/pause|resume|cancel` 操作，非法状态或过期租约返回 409；暂停次数与累计时长随任务和测量配置保存。

“失败即停”由任务执行失败（含 worker 失联）或完成的性能测量中记录了正式请求错误触发。队列中的同批次任务直接取消，已运行的同批次任务进入取消流程，原因保留为 `BATCH_STOP_ON_ERROR`；不影响其他批次。质量题目答错不触发此策略；预热错误也不作为正式请求错误处理。

## 启动

1. 从 `config/endpoints.platform.example.json` 复制为 `config/endpoints.platform.json`。默认文件是空数组；如需由部署文件预设端点，可填写受信模型地址、模型 ID 与 API key 环境变量名，不要把密钥写入 JSON。
2. 从 `.env.platform.example` 复制为 `.env.platform`，设置长度至少 32 字符的随机 `LLM_TEST_API_TOKEN`。建议同时设置独立的 `LLM_TEST_ENDPOINT_ENCRYPTION_KEY`（至少 32 字符），并在备份和恢复后保持不变；未设置时端点加密密钥由控制令牌派生，轮换令牌会使已保存的端点凭证无法解密。自定义主机需加入 `LLM_TEST_TRUSTED_API_HOSTS` 精确主机名；私有网络端点还需显式设置 `LLM_TEST_ALLOW_PRIVATE_ENDPOINTS=1`。
3. 运行 `docker compose -p llm-test-platform --env-file .env.platform -f compose.platform.yml up -d --build`。浏览器打开 `http://127.0.0.1:8000`，粘贴访问令牌，在“受测 API 设置”中保存地址、模型 ID 和 API key。默认仅监听本机；内网访问应通过企业反向代理提供 TLS 与访问控制。

Compose 将 `${LLM_TEST_TOKENIZERS_DIR:-./tokenizers}` 只读挂入 API 和 worker；需要把已有 Tokenizer 放在项目外时，在 `.env.platform` 写入绝对目录，如 `/home/ai/llm-perf/tokenizers`。新下载保存到共享 `platform-cache` 卷的 `/app/cache/tokenizers`，容器重建仍保留，不覆盖已有文件。直接启动 API / worker 时，可把 `LLM_TEST_TOKENIZER_DOWNLOAD_ROOT` 设为双方可读写的绝对路径；默认使用 `$XDG_CACHE_HOME/llm-test/tokenizers` 或 `~/.cache/llm-test/tokenizers`。

下载只接受 `TOKENIZER_SOURCES` 登记的公开 Hugging Face 仓库，不接受任意 URL、仓库或路径，不使用受测 API 的凭证。需要 worker 能访问 `huggingface.co` 及其文件分发域名；需要授权的仓库会显示失败原因。当前入口不提供 ModelScope 下载切换。每项固定 Hub 提交版本，仅取明确允许的 Tokenizer / 配置文件，禁止 Python 和模型权重；文件清单最多 4 MiB，单文件最多 64 MiB、整项最多 128 MiB / 32 文件 / 20 分钟。校验实际大小、源清单中的 LFS SHA-256，并为所有文件记录 SHA-256；禁止远程代码、离线加载并成功编码后，在同一文件系统中原子启用。

本地安装清单最多读取 64 KiB。已登记的目录、安装清单和清单文件在解析真实路径后检查所属目录；越界的软链接或仅有相似前缀的目录不作为有效安装。被拒绝的已登记目录不会回退为任意本地路径加载。

全局同一时间只执行一个下载。数据库事务保证下载与所有性能/质量/鲁棒性作业互斥；排队的测量优先领取，已开始的下载完成或取消后才领取新测量。暂停的测量也会阻止下载。每两秒续约，worker 失联后任务标记为失败（取消中的任务标记为已取消），清理临时文件；重新下载会创建新任务。测量及本地 Tokenizer 计数只加载本地文件；tiktoken 参考计数首次使用可能下载其内置词表，不视为目标模型的精确计数。手动选择的 Tokenizer 加载失败会停止测量；自动匹配不可用时仍可能使用明确标注来源的参考编码。依赖仓库自定义 Python 的 Tokenizer 当前不启用。安装版本与文件清单随性能运行快照保存，并在详情和 JSON / HTML / Markdown 报告中显示；最终 Token 指标仍以逐请求来源为准。

Compose 先运行一次 `migrate` 服务完成数据库迁移，再启动 API 与 worker，避免两个进程首次启动时同时修改 schema。

默认部署只有一个 worker。需要批次并行时，完成上述启动后增加独立 worker 进程，例如：

```bash
docker compose -p llm-test-platform --env-file .env.platform -f compose.platform.yml up -d --no-build --no-deps --scale worker=2 worker
```

每个进程执行一个任务，不能在同一进程内并行运行多个 runner。串行批次在多个 worker 下仍保持串行；暂停任务会占用 worker。增加 worker 前核对受测模型的总并发预算，多个任务共享同一模型服务时会相互影响。

本地开发：先 `pip install -e ".[dev]"`，然后在项目根目录设置相同环境变量并运行 `uvicorn server.main:app --reload` 和 `python -m server.worker`；在 `frontend/` 执行 `npm ci && npm run dev`。Vite 将 `/api` 代理至 8000 端口。

## 报告口径

性能报告逐请求统计，不读取浏览器中的临时表格。失败请求计入请求数和失败率；延迟和 TPS 的均值、p50、p95、p99 只使用成功、有限且大于零的观测值，零为未采集哨兵值。分位数用相邻样本线性插值，成功率区间用双侧 95% Wilson score。报告按实际测试维度切片，暴露样本数量、token 来源与算法、实际指标契约版本；同一运行中配置与逐请求版本不一致时拒绝汇总。只有控制任务和测量记录均成功完成、逐请求数量与计划一致、指标契约为当前版本时，报告才标记为单次运行完整性核验通过；其余报告明确标为仅供诊断。跨运行对比还须核对硬件、模型配置、工作负载与 token 来源。HTML 可直接打印为 PDF，Markdown 便于审阅归档，JSON 适于机器处理，CSV 导出逐请求指标及其口径版本。逐请求 API 和 CSV 均不返回 prompt 与 output 正文。

质量报告沿用已有 evaluator 的数据来源、样本哈希和评分口径。质量 HTML 报告展示准确率、样本数及可用时的 Wilson 区间。不要跨数据集、硬件环境或 token 计算方法直接比较准确率和吞吐；先核对数据指纹与配置。

运行详情的性能 PNG 使用同一份服务端汇总绘制各条件的 TTFT p50 / p95，注明秒单位、有效样本数、失败数、完整性、指标契约、token 来源、预热、暂停与批次并行条件。没有有效 TTFT 样本或契约冲突时拒绝绘图；未通过完整性的运行仍明确标为诊断图。图中的较长手填环境值缩略显示，完整值见 HTML / JSON。仓库的趋势、对比、分布和引擎图表也可下载 PNG。图像由浏览器生成，按两倍像素导出；浏览器需要加载本地 Plotly 资源，不需要服务端安装 Chrome / Kaleido。CSP 仅为图像允许本地 `blob:` URL，脚本与网络连接仍限定同源。

### 历史 CSV 的保存与报告

Compose 使用结果卷中的 `/app/results/history`，与源码、作业 CSV 和 SQLite 队列分开。本地直接运行 API 时，将 `LLM_TEST_HISTORY_ROOT` 设置为项目外的绝对目录，例如 `/home/ai/llm-perf/runtime/raw_data`；默认是 `LLM_TEST_ARTIFACT_ROOT` 的父目录下的 `history`。目录必须允许 API 进程读取、上传和移动文件。导入不会修改原始归档，建议通过界面上传副本；配套文件名为 `<文件名>.csv.meta.json`。

列表显示模型、测试类型、时间与文件名，文件用不透明 ID 访问，不接受客户端路径。元数据中的模型、服务商、原配置与硬件说明按原记录展示，明确标为未经核验；API key、凭证及 URL 字段不进入报告。无元数据时，仅从已知命名方式推断类型或目录中的模型，不填造执行条件。相同 CSV、展示文件名及整理后的元数据重复上传复用现有记录。

支持原指标列 `ttft`、`tpot`、`tps`、`total_time`、`prefill_speed`，耗时单位为秒、速度为 token/s；至少有一项指标列。不将 `ttft_ms` 等其他列自动换算，指标契约互相冲突时拒绝汇总。`concurrency` 对应原版并发字段；与 `concurrency_level` 同时提供且不一致时拒绝。每项统计沿用当前平台的成功、有限且正值样本筛选、线性插值分位数和 Wilson 区间。显式 `success=0` 或错误字段计为失败；没有成功状态且没有错误列的请求标为未知，保留请求总数，但不进入成功率分母和成功请求的性能统计。缺失或无效指标留空，不补零。

CSV 最多 10 MiB / 20,000 行 / 80 列，JSON 元数据最多 64 KiB，条件切片最多 128 组，超限拒绝统计，不输出截断报告。目录最多扫描 5,000 个条目、向下四层，达到上限会明确提示列表不完整；软链接、隐藏回收目录和不可读条目不作为有效历史文件。上传先认证并检查请求大小，浏览器会自动提供 `Content-Length`。

每次加载保存原 CSV 与配套元数据的 SHA-256；哈希用于识别读取内容，不证明原测量有效。历史文件始终标为执行完整性未经核验，不据此推断显著性。HTML / Markdown / JSON 保留来源、条件与缺测说明，PNG 保留原 CSV 哈希和指标口径。安全 CSV 保留原列和记录，对可疑公式文本加前导单引号，隐去敏感字段；原始存储文件保持不变。预览每页 50 条，单格最多展示 2,000 字符，不影响完整统计和 CSV 导出。

导出和删除使用读取版本进行校验，文件或元数据变更后返回 409，需重新加载或确认。确认删除会将 CSV 与配套元数据移入历史目录的 `.trash/<回收ID>/`；元数据移动失败会回滚 CSV 的移动。回收副本不显示在列表中，应与结果卷一起备份；当前没有自动清空或界面恢复入口。

认证 API：`GET /api/v1/history` 列表、`POST /api/v1/history/uploads` 上传、`GET /api/v1/history/{id}` 加载与分页、`GET /api/v1/history/{id}/figure?metric=ttft|tps|tpot|total_time|prefill_speed` 绘图、`GET /api/v1/history/{id}/report?format=html|markdown|json|csv` 导出、`POST /api/v1/history/{id}/delete` 删除。上传使用 multipart 的 `file` 和可选 `metadata`；删除体必须包含 `confirmed: true` 与当前 `revision`。加载、绘图和导出可带 `revision` 防止使用过期内容。

## 运维与备份

- API 就绪检查：`GET /health/ready`；存活检查：`GET /health/live`。
- 备份时同时保存 `platform-data` 和 `platform-results` 卷；SQLite 推荐在运行中使用 `sqlite3.Connection.backup()` 生成一致性快照。质量详细 JSON 与性能 CSV 位于结果卷。
- 监控 worker 日志中的 `WORKER_LOST` 和 `EXECUTION_FAILED`；后者对客户端只显示通用信息，详细异常保留在受控日志。
- 部署升级前备份数据库。schema 迁移至 1.11.0 是幂等的：增加 `control_presets.description` 和 `tokenizer_installs`；保留批次执行策略、暂停计数和时长，旧测量和方案保持可读。数据库备份应与端点加密密钥一起保管；下载后的 Tokenizer 在独立缓存卷中，应随缓存卷备份。
- 控制令牌保存在浏览器当前标签页的 `sessionStorage`，关闭标签页后清除；API 响应标记 `no-store`。反向代理必须强制 HTTPS 并限制内网访问。
- HumanEval/MBPP 等代码评估继续使用既有隔离 sandbox 服务；需要时按原项目文档单独启用，不能让 API 或普通 worker 获得 Docker socket。

## 当前边界

这是单机、单租户控制面。任务、租约和报告已经与 Streamlit 会话分离；底层 `BenchmarkRunner` 仍是较大的模块，且部分策略使用旧 token 校准逻辑。worker 重启后的测量断点恢复、部分高级工具以及全部初版专用图形与解释尚未迁移至新控制台。历史 CSV 已支持独立回载和报告生成，其执行条件与测量口径无法仅从文件核验。生产推广前应为目标模型准备本地 tokenizer，并用固定版本的数据集、预热条件和硬件指纹核验指标。跨主机调度、租户隔离、集中认证与 PostgreSQL 存储需要下一次架构扩展。
