# 安全边界与部署要求

本说明面向单租户内网部署。应用会持有模型 API 凭据，质量评测可能运行
模型生成的代码。部署管理员需要控制谁能访问界面、哪些模型端点可被请求，
以及代码执行服务所在主机。

## 界面与模型端点

- Streamlit 界面没有内置用户认证。默认 Compose 只绑定
  `127.0.0.1:8501`。对其他用户开放时，应在前面部署带身份认证、TLS 和
  访问日志的反向代理；不要仅通过设置 `BIND_HOST=0.0.0.0` 暴露界面。
- 已知服务商 API 域名有内置允许列表。未知域名需由管理员加入
  `LLM_TEST_TRUSTED_API_HOSTS`（逗号分隔的精确主机名）。私网地址还需设置
  `LLM_TEST_ALLOW_PRIVATE_ENDPOINTS=1`；除回环地址外，私网 IP 也必须加入
  受信任主机列表。
- 本地回环模型端点可直接使用。链路会拒绝非 HTTP(S) URL、URL 中的凭据、
  非规范 IP 别名、链路本地地址和元数据主机名。
- URL 校验发生在连接前，不能单独防止 DNS 重绑定或被信任服务的重定向。
  生产环境应通过主机或容器出口防火墙限制可达网段；对外网模型使用 HTTPS。

## 生成代码执行

HumanEval 的代码只能交给独立 `sandbox-worker` 服务；缺少服务或鉴权失败
时，评测会报基础设施错误，不生成质量分数。普通应用进程不再挂载 Docker
socket。每个提交在单独的短生命周期容器中执行，该容器无网络、宿主挂载
和应用凭据，使用只读根文件系统、非 root 用户以及 CPU、内存、进程数、
运行时长和输出大小限制。

应用内的数学表达式计算只接受长度和复杂度受限的算术及少量数学函数；
通用 Python 代码必须走独立 worker。

启用代码评测前，在运行 Docker 的专用主机上准备镜像和令牌：

```bash
docker pull python:3.12-slim
export LLM_TEST_SANDBOX_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export DOCKER_GID="$(stat -c %g /var/run/docker.sock)"
docker compose --profile code-eval up -d --build
```

生产 Compose 使用预先加载的 `llm-benchmark:latest` 和
`llm-benchmark-sandbox-worker:latest`，以
`docker compose -f docker-compose.prod.yml --profile code-eval up -d` 启动。
令牌应由部署方的秘密管理机制提供，不要写入仓库。`sandbox-worker` 只接入
内部 Docker 网络，不映射宿主端口。执行镜像必须预先加载，因为 worker
固定使用 `--pull=never`。

**剩余风险：**worker 为创建隔离容器仍持有 Docker socket；攻陷 worker
可能导致宿主机被控制。因此代码执行应放在专用、可重建的主机或虚拟机上，
限制访问者，并及时更新 Docker 与内核。容器隔离不等同于针对内核漏洞的
强隔离。下一阶段可改用不暴露宿主 Docker socket 的专用执行后端。

## 数据与日志

- 正式质量评测缺少数据集时会失败。`LLM_TEST_ALLOW_EMBEDDED_SAMPLES=1`
  只用于演示；结果中的 `dataset_provenance.source=embedded_demo` 和报告警告
  表明其不能作为正式基准。
- 模型 API 请求失败会使当前数据集无分数。A/B 对比要求所有模型完成同一批
  正式样本，并验证样本指纹一致，才生成排名。
- 评测结果记录已使用评测及 few-shot 样本的顺序敏感 SHA-256 指纹、来源、
  路径、子集、样本数和选择种子。数据集 revision、模型服务版本、代码提交和
  tokenizer 指纹尚未全面写入，应在比较或发布分数前补齐。
- `.env`、API 日志、结果和数据目录可能含凭据或敏感提示词。由部署方设置
  访问权限、保留期限及备份策略。不要把这些目录提交到 Git。

## 验证

```bash
python -m pytest tests/test_security.py tests/test_dataset_provenance.py -q
RUN_SANDBOX_INTEGRATION=1 python -m pytest tests/test_sandbox_worker.py -q
ruff check .
```

第二条命令需要本机 Docker 与已加载的 `python:3.12-slim`。其集成用例
检查提交代码无法读取父进程变量、宿主文件、外部网络，并检查超时终止。
