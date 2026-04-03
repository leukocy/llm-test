# 创建 Pull Request 指南

## 方式 1: 通过 GitHub 网页创建（推荐）

### 步骤 1: 打开仓库
访问：https://github.com/leukocy/llm-test

### 步骤 2: 创建 PR
你应该会看到一个黄色提示框：
```
ci-cd-setup had recent pushes 1 hour ago
Compare & pull request
```

点击 **"Compare & pull request"** 按钮

如果没有显示，手动创建：
1. 点击 **"Pull requests"** 标签
2. 点击 **"New pull request"**
3. 选择 base: `main` ← compare: `ci-cd-setup`

### 步骤 3: 填写 PR 信息

**标题:**
```
feat: comprehensive code quality and security improvements
```

**描述:**
```
## 变更摘要
全面的代码质量和安全性改进，包括：
- ✅ 完整的 CI/CD 配置
- ✅ 安全漏洞修复 (SSRF 防护、安全代码执行)
- ✅ 代码质量修复 (裸 except、临时文件泄漏、死代码)
- ✅ 完善的项目文档
- ✅ 代码质量评分: B+ → A-

## 主要改进

### CI/CD 配置 ⭐
- GitHub Actions 工作流 (ci, codeql, pr-check, release)
- Pre-commit hooks (black, isort, ruff, mypy, bandit)
- Issue 和 PR 模板
- CODEOWNERS 配置

### 安全改进 🔒
- SSRF 防护 (url_validator.py)
- 安全代码执行 (safe_executor.py)
- 速率限制 (rate_limiter.py)
- 认证和密钥管理

### 代码质量 🐛
- 修复 9 处裸 except
- 修复 2 处临时文件泄漏
- 移除 68 行死代码

### 文档 📚
- 代码审查报告
- 开发者视角分析
- 目标导向分析
- 技术深度分析
- Git 工作流规范

## 测试
- ✅ 21/21 安全测试通过
- ✅ 导入验证通过
- ✅ 代码规范检查通过

## 破坏性变更
无

Closes #1
Closes #2
```

### 步骤 4: 创建 PR
点击 **"Create pull request"** 按钮

---

## 方式 2: 使用 GitHub CLI

如果你安装了 GitHub CLI：

```bash
# 安装 GitHub CLI (如果没有)
# winget install GitHub.cli

# 登录
gh auth login

# 创建 PR
gh pr create \
  --title "feat: comprehensive code quality and security improvements" \
  --body "See full description in PR" \
  --base main \
  --head ci-cd-setup
```

---

## PR 创建后的检查

### 1. 等待 CI 检查
在 PR 页面底部会显示：
- ✅ lint - 代码风格检查
- ✅ security-tests - 安全测试 (21 个)
- ✅ unit-tests - 单元测试
- ✅ import-check - 导入验证
- ✅ dependency-check - 依赖审计

等待所有检查通过（约 10-15 分钟）

### 2. 审查代码
查看文件变更，确保：
- 所有修改符合预期
- 没有敏感信息泄露
- 配置正确

### 3. 合并 PR
检查全部通过后：
1. 点击 **"Merge pull request"**
2. 确认合并
3. 可选：删除分支 `ci-cd-setup`

---

## 合并后配置

### 配置分支保护

**Settings → Branches → Add rule**

```
分支名称模式: main

✅ Require a pull request before merging
   ✅ Require approvals: 1
   ✅ Dismiss stale reviews: 启用

✅ Require status checks to pass before merging
   ✅ Require branches to be up to date before merging
   选择必需的检查:
   ✅ lint
   ✅ security-tests
   ✅ unit-tests (Python 3.10)
   ✅ import-check

❌ Do not allow bypassing the above settings
```

### 安装 Pre-commit Hooks

```bash
cd D:\heyi\llm-test-streamlit\llm-test
pip install pre-commit
pre-commit install
```

### 更新 CODEOWNERS

编辑 `.github/CODEOWNERS`，将 `@maintainer` 等占位符替换为实际的 GitHub 用户名。

---

## 快速链接

- 仓库: https://github.com/leukocy/llm-test
- 创建 PR: https://github.com/leukocy/llm-test/compare/main...ci-cd-setup
- 分支保护: https://github.com/leukocy/llm-test/settings/branches
