# Git 推送计划

**仓库地址:** git@github.com:leukocy/llm-test.git
**日期:** 2026-01-31

---

## 当前状态分析

### 已暂存的文件（待推送）

```
已提交到本地 test 分支 (commit 4603768):
- .github/workflows/ (5个 CI/CD 工作流文件)
- .github/CODEOWNERS
- .github/PR 模板和 Issue 模板
- .pre-commit-config.yaml
- pyproject.toml (更新)
- config/auth.py, config/secrets.py, config/development_settings.py
- core/safe_executor.py, core/url_validator.py, core/rate_limiter.py
- tests/test_security.py
- utils/log_sanitizer.py
- docs/SECURITY.md
- docs/GIT_WORKFLOW.md (新增)
```

### 未提交的修改

```
约 50+ 个文件有修改，包括：
- 核心模块的安全修复 (benchmark_runner.py, enhanced_parser.py 等)
- Provider 修复 (openai.py, gemini.py)
- UI 组件修复
- 评估器修复
```

---

## 推送策略

### 方案 A: 分阶段推送（推荐）

**第1阶段：仅推送 CI/CD 配置**

```bash
# 当前状态：已提交 CI/CD 配置到本地 test 分支
# 目标：推送到远程 main 分支的 ci-setup 分支

# 1. 确保当前在 test 分支
git branch  # 应显示 * test

# 2. 创建 CI/CD 专用分支
git checkout -b ci-setup

# 3. 推送到远程
git push -u origin ci-setup

# 4. 在 GitHub 创建 PR: ci-setup → main
# 5. 等待 CI 验证通过
# 6. 合并到 main
```

**第2阶段：推送安全修复（单独分支）**

```bash
# 1. 从 main 创建新分支
git checkout main
git pull origin main
git checkout -b security-fixes

# 2. 添加安全修复文件
git add core/benchmark_runner.py
git add core/enhanced_parser.py
git add core/response_cache.py
git add evaluators/yaml_evaluator.py
git add core/failure_analyzer.py
git add core/consistency_tester.py
git add app.py
git add ui/thinking_components.py

# 3. 提交
git commit -m "fix: resolve code quality issues

- Fix bare except clauses (9 instances)
- Fix temporary file leaks (2 instances)
- Remove dead code (68 lines)
- Update error handling to use specific exceptions

Improves code quality grade from B+ to A-"

# 4. 推送
git push -u origin security-fixes

# 5. 创建 PR
```

**第3阶段：推送 Provider 修复**

```bash
# 1. 从 main 创建分支
git checkout main
git pull origin main
git checkout -b fix/providers

# 2. 添加 Provider 相关修复
git add core/providers/openai.py
git add core/providers/gemini.py
git add config/settings.py

# 3. 提交
git commit -m "fix(providers): improve provider security and reliability

- Add SSRF protection to URL validation
- Fix connection pool configuration
- Reduce max_workers from 5000 to 100
- Update internal server handling"

# 4. 推送
git push -u origin fix/providers
```

### 方案 B: 整批推送（快速但风险较高）

```bash
# 适用于：所有修改都经过充分测试

# 1. 从 main 创建功能分支
git checkout main
git pull origin main
git checkout -b feature/code-quality-improvements

# 2. 添加所有修改
git add -A

# 3. 提交（注意：这会是一个大提交）
git commit -m "feat: comprehensive code quality and security improvements

This includes:
- GitHub Actions CI/CD configuration
- Security improvements (SSRF protection, safe code execution)
- Code quality fixes (bare except, temp file leaks)
- Provider fixes and improvements
- Updated project configuration

BREAKING CHANGES: None

See individual commits for detailed changes"

# 4. 推送
git push -u origin feature/code-quality-improvements

# 5. 创建大 PR，可能需要拆分审查
```

---

## 推荐执行步骤

### Step 1: 验证 CI/CD 配置

```bash
cd D:\heyi\llm-test-streamlit\llm-test

# 确保当前提交包含 CI/CD 文件
git log --oneline -1

# 应该显示:
# 4603768 feat(ci): add GitHub Actions CI/CD configuration
```

### Step 2: 推送 CI/CD 配置

```bash
# 创建推送分支
git checkout -b ci-cd-setup

# 推送到远程
git push -u origin ci-cd-setup
```

### Step 3: 在 GitHub 操作

1. 访问：https://github.com/leukocy/llm-test
2. 应该看到新分支 `ci-cd-setup`
3. 创建 PR: `ci-cd-setup` → `main`
4. 等待 CI 检查完成（约 10-15 分钟）
5. 如果通过，合并 PR

### Step 4: 验证 CI 运行

CI 运行的检查：
- [ ] lint - 代码风格检查
- [ ] security-tests - 21 个安全测试
- [ ] unit-tests - 单元测试
- [ ] import-check - 导入验证
- [ ] dependency-check - 依赖审计

### Step 5: 合并后配置 GitHub 设置

**Settings → Branches → Add rule**

```
分支名称模式: main

✅ Require a pull request before merging
✅ Require approvals (1)
✅ Require status checks to pass before merging
   ✅ lint
   ✅ security-tests
   ✅ unit-tests
✅ Require branches to be up to date before merging
```

---

## 故障排查

### 如果推送失败

**问题：大文件错误**

```bash
# 查找大文件
git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  awk '/^blob/ {print substr($0,6)}' |
  sort -nk2 |
  tail -10

# 移除大文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch datasets/longbench/data.zip" \
  --prune-empty --tag-name-filter cat -- --all
```

**问题：合并冲突**

```bash
# 同步远程 main
git fetch origin main
git rebase origin/main

# 解决冲突后
git add .
git rebase --continue
```

---

## 推送后检查清单

- [ ] 代码已推送到远程仓库
- [ ] GitHub Actions CI 运行成功
- [ ] 分支保护规则已配置
- [ ] CODEOWNERS 已更新用户名
- [ ] Pre-commit hooks 已安装

---

**推荐方案:** 方案 A（分阶段推送）
**预计时间:** 30-45 分钟（包括 CI 运行时间）
**风险等级:** 低
