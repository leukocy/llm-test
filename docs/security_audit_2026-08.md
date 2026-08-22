# LLM Test 平台安全审查报告

- **日期**: 2026-08-22
- **范围**: master @ 67421f7（含工作区），覆盖 `app.py`、
  `core/`、`ui/`、`utils/`、`evaluators/`、`config/`、Docker/compose 部署配置
- **方法**: 人工代码审计（沙箱逃逸/SSRF/SQL 注入/反序列化/XSS/
  密钥管理逐项核查）+ PoC 实际验证 + bandit 静态扫描 +
  pip-audit 依赖漏洞扫描
- **既有防线**: `tests/test_security.py` 21 项全部通过（沙箱基本用例、
  SSRF 校验器单测、路径遍历、日志清洗、限流）

## 总体结论

平台的输入校验类防线（路径遍历、preset 文件名清洗、请求日志头掩码）
是有效的；但**核心沙箱 `safe_exec_code` 可被绕过、超时保护未实现
（实测可挂死）、SSRF 校验器是死代码、部署形态（无认证 UI 绑定
0.0.0.0 + 挂载 docker socket）使上述问题可直接升级为宿主机接管**。
按内网基准测试工具的现实威胁模型，建议优先处理 P0 三项。

## 发现分级总览

| # | 等级 | 发现 | 位置 |
|---|------|------|------|
| 1 | P0 | 沙箱可绕过：dunder 黑名单仅扫源码文本，实测属性链逃逸 | `core/safe_executor.py` |
| 2 | P0 | `timeout_seconds` 未实现，实测挂死线程；无内存限制 | `safe_executor.py:139` |
| 3 | P0 | UI 无认证 + XSRF 关 + 0.0.0.0，叠加 #6 宿主机接管 | `.streamlit/config.toml` |
| 4 | 高 | SSRF 校验器为死代码，Base URL 可指向内网 | `core/url_validator.py` |
| 5 | 高 | 数据集字段 `eval()`，投毒可 RCE | `longbench_evaluator.py:255` |
| 6 | 高 | docker.sock 进容器可任意 exec，等价 root | compose, `engine_capture.py:824` |
| 7 | 中 | pickle 反序列化缓存/检查点 | `response_cache.py:471,484` |
| 8 | 中 | f-string 拼 SQL 标识符（纵深缺口） | `repositories/base.py` |
| 9 | 中 | 日志脱敏不成体系；掩码仅三种头 | `utils/log_sanitizer.py` 等 |
| 10 | 低 | Jinja2 autoescape=False（提示注入面） | `prompt_template.py:77` |
| 11 | 低 | 29 处 markdown f-string 插值外部数据（格式注入） | `ui/evaluation_dashboard.py` 等 |
| 12 | 低 | 弱哈希用于指纹（非安全用途）；url_validator 自身有绕过缺口 | 多处 |
| 13 | 低 | 依赖未锁定版本，环境 228 个已知漏洞（33 包） | `requirements.txt` |

## 详细发现与证据

### 1. [P0] safe_exec_code 沙箱绕过（实测）

`safe_exec_code` 的防线是「源码文本正则黑名单（`__class__`/`__mro__`/
`__subclasses__`/`__globals__`/`__builtins__` 等）+ 受限 builtins 白名单」。
黑名单在**源码字符串**上匹配，而 Python 允许运行时拼出属性名再交给
`str.format` 的字段语法解析——文本扫描完全失效：

```text
PoC（已在本仓库实测执行成功）:
u='_'; nc=u*2+'cl'+'ass'+u*2; nb=u*2+'ba'+'se'+u*2
ns=u*2+'su'+'bclasses'+u*2
A=type('A',(),{})
tpl='{0.'+nc+'.'+nb+'.'+ns+'}'
print(tpl.format(A()))
# 输出: <built-in method __subclasses__ of type object at 0x91c3e0>
```

白名单同时给了 `type`（三参动态建类，无需 class 语句）与
`map`/`filter`/`sorted(key=)`（调用原语）。本次测试未走通到任意代码
执行的最后一跳（format 返回值被 str 化，bound method 无法经它保留
对象类型），但「任意属性链读取 + 动态建类」已足以读取敏感对象、
探测解释器内部，且绕过手法会持续演化。

**威胁路径**: HumanEval/MBPP 评测的被测代码来自 LLM 输出 →
对抗性提示可让模型生成如上代码 → 在宿主机进程中执行。

**建议**: 放弃 AST 黑名单式自研沙箱，改为进程隔离执行
（子进程 + rlimit/seccomp，或容器/微 VM）；至少移除 `type` 出白名单、
拒绝一切 Attribute 访问（math 场景除外）、把 timeout 落实为进程级 kill。

### 2. [P0] 超时未实现（CPU/内存 DoS，实测）

`safe_exec_code(code, timeout_seconds=5)` 的 docstring 自认
"not enforced in this implementation"。实测：
`safe_exec_code('while True: pass', timeout_seconds=1)` 的工作线程
3 秒后仍存活（证据输出 `alive_after_3s = True`）；
`[0] * 10**9` 约 1.6s 完成 ~8GB 列表分配，无内存限制。
评测并发下单个失控样本即可拖垮整个测试进程。

### 3. [P0] 无认证 UI + XSRF 关闭 + 0.0.0.0

`.streamlit/config.toml`: `enableXsrfProtection = false`、
`address = "0.0.0.0"`；两个 compose 文件同样绑定 0.0.0.0 并把
8501 映射到宿主机。Streamlit 应用无任何登录/令牌。任何能路由到该
端口的主机都可：提交测试（触发 #1/#4/#6）、读结果、改配置。

**建议**: 反向代理层加 SSO/Basic Auth + TLS；开启 XSRF（若确有
关闭理由，写明并限制监听 127.0.0.1）；compose 补网络隔离说明。

### 4. [高] SSRF 校验器是死代码

`core/url_validator.py`（含 `is_safe_url`/`validate_and_normalize_url`/
`is_port_safe`）除自身单测外无任何生产调用方。侧边栏
「Custom (OpenAI Compatible)」的 base URL 直达 httpx 客户端，
可指向 `169.254.169.254` 云 metadata、内网管理面板等；响应还会进入
错误消息/日志回显给 UI 操作者。另注意校验器本身即便启用也存在缺口：
十进制/八进制 IP 表示、IPv6 字面量、DNS rebinding、30x 重定向均不拦截，
且未知域名策略是 warn-but-allow。

**建议**: 在 provider 工厂入口统一调用 URL 校验（私有部署场景可用
`allow_private=True` 显式放行）；补 DNS 解析后 IP 复核与重定向禁用。

### 5. [高] LongBench 评测 eval() 数据集字段

```python
# evaluators/longbench_evaluator.py:255
correct_list = eval(correct) if correct.startswith("[") else [correct]
```

`correct` 来自数据集文件（HF Hub 下载或本地上传，
`ui/dataset_manager.py` 支持 file_uploader）。恶意/被投毒的数据集行
即可在评测进程内执行任意代码。同型问题：`enhanced_parser.py:551` 与
`smart_answer_parser.py:339` 对 LLM 输出的受限字符 eval
（字符白名单后仍建议换 `ast.literal_eval`/手写求值器）。

**建议**: 该处用 `ast.literal_eval` + try 回退 `[correct]` 即可，行为不变。

### 6. [高] docker.sock 进容器 + 无约束 docker exec

dev/prod compose 均挂载 `/var/run/docker.sock` 与宿主 docker CLI
（供 engine_capture 采集引擎信息）。`core/engine_capture.py:824`
执行 `docker exec <container> python -c <script>`；container 参数
来自 find_vllm_container（docker ps 输出解析，非直接用户输入），
命令以 argv 数组传给 subprocess.run（无 shell 注入）。但组合 #3 后：
任何能打开 UI 的人可通过让平台请求任意 base URL 影响容器匹配逻辑，
进而借助 docker.sock 等价获得宿主机 root。这是部署形态问题。

**建议**: 生产改用只读 Docker API 代理（如 docker-socket-proxy，
endpoint 白名单）；或 engine_capture 仅在 dev profile 挂载。

### 7. [中] pickle 反序列化

`core/response_cache.py:471,484` pickle.load 加载
`cache/checkpoints/*.pkl.gz`。能写该目录即得代码执行。当前为本地
信任域内文件，风险取决于 cache 目录是否会被共享/挂载（compose 将
./cache 挂入容器）。建议改 JSON/msgpack 或加 HMAC 完整性校验。

### 8. [中] SQL 组装方式（纵深防御）

BaseRepository/Database 以 f-string 拼 table/WHERE/ORDER BY，
值一律 `?` 参数化（bandit B608 ×24 均为此模式）。审计了全部调用方：
表名为代码常量，WHERE/ORDER BY 当前均为字面量，**未发现可达的
用户输入注入路径**；presets 名称经 `_sanitize_preset_name` 清洗
（`.` 被剥除，`../` 无法存活）。但接口签名接受任意字符串，未来接入
排序/筛选 UI 时会立刻变成注入点。建议加标识符白名单校验。

### 9. [中] API Key 日志脱敏不体系

RequestLogger 默认掩码 authorization/api-key/x-api-key 三种头
（mask_api_key 默认开，好）；但 log_sanitizer.sanitize_api_key 无任何
生产调用方（仅测试引用），通用 logger 不经过清洗，Gemini provider 的
x-goog-api-key 头不在掩码名单。建议掩码名单补 Gemini 头；
logger 层统一挂 sanitize hook。

### 10–12. [低] 其他

- `prompt_template.py:77` Jinja2 autoescape=False：输出进入 LLM API
  请求，构成提示注入面而非 XSS；模板来源若引入用户可控内容需重新评估。
- 29 处 st.markdown(f...) 插值数据集/模型输出：Streamlit markdown
  默认不渲染原始 HTML，当前为 Markdown 格式注入（低危）；不要在这些
  渲染处加 unsafe_allow_html=True（现有 47 处该开关均为静态 CSS，
  ui/log_viewer.py 已正确 html.escape）。
- SHA1/MD5 用于 config/硬件指纹（非密码学用途，B324 可标
  usedforsecurity=False）。
- url_validator 自身缺口见 #4。

### 13. [低] 依赖漏洞（pip-audit，OSV 基线）

环境共 **228 个已知漏洞 / 33 包**。与 requirements.txt 直接相关的
运行时组件：

| 包 | 当前 | 漏洞数 | 修复版本 |
|----|------|--------|----------|
| pillow | 12.0.0 | 27 | 12.3.0 |
| transformers | 4.57.3 | 5 | 5.3.0（PYSEC-2025-217 无修复版） |
| tornado (streamlit) | 6.5.4 | 7 | 6.5.7 |
| starlette | 0.50.0 | 7 | 1.1.0 |
| pyjwt | 2.12.1 | 5 | 2.13.0 |
| requests | 2.32.5 | 1 | 2.33.0 |
| datasets | 5.0.0 | 1 | 5.0.1 |
| setuptools | 79.0.1 | 2 | 83.0.0 |
| pip | 25.2 | 7 | 26.2 |
| streamlit | 1.53.1 | 1 | 1.54.0 |

其余（vllm×49、aiohttp×33、gitpython、litellm、ray…）属
bench_workspace 等工具链环境，不影响交付镜像。
requirements.txt/pyproject.toml 全部为 >= 下限无锁定，构建不可复现。
建议引入 lock（uv/pip-tools/conda-lock）并在 CI 加 pip-audit 门禁。

## 正面确认（审计通过项）

- dataset_loader._get_file_path：路径遍历防护有效（单测覆盖 ../、
  绝对路径、反斜杠）
- preset_manager._sanitize_preset_name：文件名清洗有效
- request_logger：默认掩码认证头；文件名对 provider/model 做了字符过滤
- rate_limiter：令牌桶实现正确（单测通过）
- Dockerfile：.env 未进入镜像（实测 has_.env: False）；bundle 自包含
- pre-commit 集成 bandit/isort/black/mypy 作为门禁

## 建议处置顺序

1. **立即**: 沙箱改进程隔离 + 真超时（#1/#2）；
   LongBench eval 换 ast.literal_eval（#5，一行改动）
2. **短期**: 反代加认证/TLS、开 XSRF 或收窄监听（#3）；
   provider 入口接 URL 校验（#4）；生产去 docker.sock（#6）
3. **中期**: pickle 换格式（#7）；SQL 标识符校验（#8）；
   日志脱敏统一（#9）；依赖锁 + CI 审计门禁（#13）

## 修复状态附录（2026-08-22，fix/security-hardening 分支）

| # | 状态 | 修复方式 | 回归测试 |
|---|------|----------|----------|
| 1 | ✅ | 子进程隔离执行（`-I` + rlimit），PoC 困于子进程无害 | SandboxIsolation |
| 2 | ✅ | 真实墙钟强杀；内存炸弹被 RLIMIT_AS 拦截 | 同上 |
| 3 | 部分 | XSRF 开启、注释写明收敛建议；反代认证属部署侧 | — |
| 4 | ✅ | 工厂统一接 URL 校验；私网端点显式放行 | ProviderSSRF |
| 5 | ✅ | LongBench 改 literal_eval；HumanEval 进沙箱 | LongBenchSafeEval |
| 6 | 部分 | docker exec 开关默认关闭；compose 去 socket 属部署侧 | DockerExecGate |
| 7 | ✅ | HMAC 签名 + 名称清洗；篡改实测被拒 | CheckpointIntegrity |
| 8 | ✅ | 标识符白名单 + WHERE 黑名单；值仍参数化 | SQLIdentifier |
| 9 | ✅ | Gemini 头掩码；error 出口与 logger 统一清洗 | LogSanitization |
| 10 | 确认 | 提示注入面非 XSS；模板来源受控，维持现状并记录 | — |
| 11 | 缓解 | 默认不渲染 HTML，仅格式注入；无 unsafe_allow_html 叠加 | — |
| 12 | ✅ | usedforsecurity=False；URL 白名单；os.system 改 subprocess | — |
| 13 | 待办 | 修复版本已给出；依赖锁与 CI 门禁待基建项 | — |

**验证基线**: tests/test_security.py 21→42 项全过；全仓 1248 测试通过；
ruff 0 错误；mypy 错误数较 master 基线净减（115 vs 116）。
部署侧待办（#3 反代认证、#6 compose 去 socket、#13 依赖锁）需基础设施配合，
已在上方"建议处置顺序"中保留。
