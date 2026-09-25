# 单租户内网部署升级路线与验收门槛

目标是把 llm-test 建成可重复运行、可审计、可运维的 LLM 性能与质量测试
基础设施。当前阶段优先消除安全和分数可信度的阻断问题。2026-09-25 的原始
评估和证据在开发机上归档于 `~/llm-perf/evaluation/2026-09-25/`，不属于
项目源码或运行数据。

## 当前交付的基础

<!-- markdownlint-disable MD013 -->

| 验收项 | 当前状态 | 证据或限制 |
|---|---|---|
| 不可信代码与应用凭据隔离 | 基础边界已实现 | HumanEval 请求发送到带鉴权的独立 worker；每次执行使用无网络、无挂载、只读、非 root、限资源的容器。Docker 实测覆盖环境变量、文件、网络和超时。worker 仍持有 Docker socket，应放专用主机。 |
| 缺数据集或 API 失败时不出正式分数 | 已实现 | 缺失或空子集、模型 API 请求失败会中止质量评估；演示样例需显式开启，结果和摘要标记 `embedded_demo`。A/B 排名拒绝演示、缺失或样本指纹不一致的结果。 |
| 可追溯的样本集合 | 部分实现 | 结果 JSON 记录有序评测及 few-shot 样本 SHA-256、来源、路径、子集、样本数和选择种子；尚缺下载 revision、数据集原始文件校验值及模型服务版本。 |
| 模型端点访问控制 | 部分实现 | 未知主机须管理员信任；私网还需显式启用；IP 别名被拒。DNS 重绑定需出口网络策略补强。 |
| CI 与主分支一致 | 基础门禁已修 | `master` 的 PR 和 push 触发 CI、CodeQL；Ruff、测试、容器隔离探针与已安装依赖审计运行。类型检查和 Bandit 仍为非阻断，依赖尚未锁定。 |
| 性能指标口径 | 基础契约已实现 | TPS/TPOT 使用同一解码 token 间隔；CSV 和数据库记录 `decode-interval-v2`，报表拒绝混用新旧版本。分块时间戳不能精确代表逐 token 延迟，仍需可控 SSE 金标准测试。 |
| 干净检出可构建镜像 | 部分实现 | 主 Dockerfile 不再强制要求未跟踪的 tokenizer bundle；完整镜像构建、镜像扫描和无网启动仍需在目标环境验收。 |
| 受控生产访问 | 部署前置条件 | 默认端口只绑定回环地址；管理员必须提供带认证和 TLS 的反向代理。项目尚无内置用户认证。 |

<!-- markdownlint-enable MD013 -->

## 分阶段验收

### 1. 安全和结果可信度

1. 在专用执行主机验证真实模型生成代码无法读取应用凭据、宿主文件和网络；worker 不对外暴露，应用没有 Docker socket。保存 canary 测试和部署配置作为证据。
2. 对每个质量数据集固定仓库、revision、split、下载文件哈希、清洗版本与样本哈希。缺失、空子集和解析错误均应停止评估；演示分数不得进入正式比较和排名。
3. 每次运行保存模型 ID、提供商端点、服务版本、参数、采样种子、tokenizer 指纹、项目提交、环境镜像 digest 与数据集指纹。密钥不写入结果。
4. 通过反向代理实施认证、TLS 和访问记录，并在主机/容器出口实施模型端点允许规则。针对 DNS 重绑定和重定向补充端到端检查。

### 2. 测量契约与统计

1. 按 [性能指标契约](METRICS_CONTRACT.md) 验收 TTFT、生成时长、TPS、TPOT、
   分块间隔和成功率的分母与缺失值处理；继续补齐运行结果 schema 的独立版本号。
2. 使用可控 SSE 服务做金标准测试，覆盖单 token、多个 token 同块、usage 缺失、重试、取消、超时与并发。真实模型每个配置重复运行并报告样本量、错误率、分位数及置信区间。
3. 比较历史结果前先校验完整性。旧运行中缺耗时、状态未结束或请求计数不符的记录应保留原始备份并标成不可比较，不直接修写旧库。

### 3. 交付与运维

1. 固定 Python 依赖和基础镜像 digest，生成 SBOM，建立更新与漏洞响应流程；让 Ruff、格式、类型检查、测试和镜像构建成为明确门禁。先清理现存格式和类型错误，再启用相应阻断。
2. 从干净检出构建应用与 worker 镜像；在无 GPU 和 NVIDIA GPU 环境各验收
   一次。GPU 环境叠加 `docker-compose.gpu.yml`。需要完全离线时先运行
   `python scripts/prepare_tokenizers.py`，再校验 bundle 与模型映射。
3. 增加迁移与回滚演练、备份恢复演练、运行队列与并发限额、任务取消、资源用量指标、健康检查和告警。SQLite 单实例写入上限应经真实负载验证；需要多用户并发时再迁移到服务型数据库。
4. 对镜像、配置和结果实施版本化发布，保留每次验收记录。只有全部生产门槛通过，才标记为工业级可用。

## 本地验证命令

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest tests/ -q
RUN_SANDBOX_INTEGRATION=1 python -m pytest tests/test_sandbox_worker.py -q
docker compose config -q
docker compose -f docker-compose.prod.yml config -q
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config -q
```

`RUN_SANDBOX_INTEGRATION=1` 需要 Docker 与预先加载的 `python:3.12-slim`。代码执行部署步骤和剩余威胁见 [SECURITY.md](SECURITY.md)。
