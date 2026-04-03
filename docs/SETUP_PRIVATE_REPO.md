# 创建私有仓库并推送 CI/CD 配置指南

## 步骤 1: 在 GitHub 上创建私有仓库

### 1.1 登录 GitHub
访问：https://github.com

### 1.2 创建新仓库
1. 点击右上角 `+` → `New repository`
2. 填写仓库信息：
   - **Repository name**: `llm-test`（或你喜欢的名称）
   - **Description**: `LLM 性能基准测试平台 V2`
   - **选择 Private**（私有仓库）
   - **不要勾选** "Add a README file"
   - **不要勾选** "Add .gitignore"
   - **不要勾选** "Choose a license"
3. 点击 `Create repository`

### 1.3 获取仓库地址
创建后，GitHub 会显示你的仓库地址，格式如：
```
https://github.com/你的用户名/llm-test.git
```

---

## 步骤 2: 推送文件到仓库

### 方法 A: 使用脚本（推荐）

**Windows:**
```cmd
cd D:\heyi\llm-test-streamlit\llm-test
push_to_repo.bat https://github.com/你的用户名/llm-test.git
```

**Linux/Mac:**
```bash
cd D:\heyi\llm-test-streamlit\llm-test
chmod +x push_to_repo.sh
./push_to_repo.sh https://github.com/你的用户名/llm-test.git
```

### 方法 B: 手动命令

```bash
# 1. 进入项目目录
cd D:\heyi\llm-test-streamlit\llm-test

# 2. 添加远程仓库
git remote add origin https://github.com/你的用户名/llm-test.git

# 3. 提交已暂存的文件
git commit -m "feat(ci): add GitHub Actions CI/CD configuration

- Add CI workflow with automated testing
- Add CodeQL security analysis
- Add PR quality checks
- Add release workflow
- Add pre-commit hooks
- Add branch protection rules
- Add Issue and PR templates
- Update project configuration"

# 4. 推送到远程仓库
git push -u origin main
```

---

## 步骤 3: 验证推送成功

### 3.1 检查 GitHub 仓库
访问你的仓库地址，应该看到：
- `.github/workflows/` 文件夹
- `.pre-commit-config.yaml`
- `pyproject.toml`
- 等配置文件

### 3.2 检查 Actions
1. 在仓库页面点击 `Actions` 标签
2. 应该能看到 CI 工作流（如果已有提交触发）

---

## 步骤 4: 配置 GitHub 设置

### 4.1 配置分支保护规则

**Settings → Branches → Add rule**

**分支名称**: `main`

**启用以下选项：**
- ✅ `Require a pull request before merging`
- ✅ `Require approvals` (设置为 1)
- ✅ `Require status checks to pass before merging`
- ✅ 选择必需的检查：
  - `lint`
  - `security-tests`
  - `unit-tests`
- ✅ `Require branches to be up to date before merging`

### 4.2 更新 CODEOWNERS

编辑 `.github/CODEOWNERS`，将 `@maintainer` 等占位符替换为实际的 GitHub 用户名。

---

## 步骤 5: 本地安装 Pre-commit

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks
cd D:\heyi\llm-test-streamlit\llm-test
pre-commit install

# 可选：在所有文件上运行一次
pre-commit run --all-files
```

---

## 推送后立即检查

### 1. 创建测试 PR
```bash
git checkout -b test/ci-setup
echo "# Test CI" >> test_ci.md
git add test_ci.md
git commit -m "test: verify CI configuration"
git push origin test/ci-setup
```

### 2. 在 GitHub 上创建 PR
- 访问仓库页面
- 点击 `Compare & pull request`
- 创建 PR

### 3. 观察 CI 运行
- PR 页面应显示所有检查项
- 等待 CI 完成（约 5-15 分钟）
- 确认所有检查通过 ✅

---

## 常见问题

### Q: 推送时提示身份验证失败？
**A:** 使用 Personal Access Token：
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → 勾选 `repo` 权限
3. 使用 token 作为密码登录

### Q: 提示 "remote origin already exists"？
**A:** 更新远程仓库地址：
```bash
git remote set-url origin https://github.com/你的用户名/llm-test.git
```

### Q: 推送失败，提示分支受保护？
**A:** 先推送到其他分支，再创建 PR：
```bash
git push -u origin ci-setup
```

---

## 完成检查清单

- [ ] GitHub 私有仓库已创建
- [ ] 文件已推送到仓库
- [ ] `.github/workflows/` 文件夹存在
- [ ] 分支保护规则已配置
- [ ] CODEOWNERS 已更新用户名
- [ ] Pre-commit hooks 已安装
- [ ] 测试 PR 已创建并验证 CI 通过

---

## 需要帮助？

如果遇到问题，请提供：
1. 错误信息的截图或文本
2. 执行的命令
3. 仓库地址（如果是公开的）
