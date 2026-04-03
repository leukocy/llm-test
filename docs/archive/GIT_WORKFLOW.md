# Git 工作流与推送规范

**版本:** 1.0
**项目:** LLM 性能基准测试平台 V2
**生效日期:** 2026-01-31

---

## 一、分支策略

### 1.1 分支结构

```
main           ─┐
               ├── 合并后成为生产版本
develop        ─┤
               ├── 开发集成分支
feature/*      ─┤
               ├── 功能开发分支
fix/*          ─┤
               ├── Bug 修复分支
hotfix/*       ─┤
               ├── 紧急修复分支
release/*      ─┘
               ├── 发布准备分支
```

### 1.2 分支命名规范

| 分支类型 | 命名格式 | 示例 | 说明 |
|----------|----------|------|------|
| 主分支 | `main` | - | 生产环境代码 |
| 开发分支 | `develop` | - | 开发集成分支 |
| 功能分支 | `feature/<名称>` | `feature/benchmark-optimization` | 新功能开发 |
| 修复分支 | `fix/<问题描述>` | `fix/tokenization-leak` | Bug 修复 |
| 紧急修复 | `hotfix/<问题描述>` | `hotfix/security-patch` | 生产环境紧急修复 |
| 发布分支 | `release/<版本号>` | `release/v2.1.0` | 发布准备 |
| 文档分支 | `docs/<描述>` | `docs/api-reference` | 文档更新 |

### 1.3 分支保护规则

#### main 分支（保护级别：最高）

```
┌─────────────────────────────────────────────────────┐
│              main 分支保护规则                       │
├─────────────────────────────────────────────────────┤
│ ✅ 必须通过 PR 合并                                 │
│ ✅ 至少 1 个代码审查批准                            │
│ ✅ 所有 CI 检查必须通过                             │
│ ✅ 分支必须最新（与 main 同步）                     │
│ ✅ 新提交需重新审查                                 │
│ ✅ 需要 CODEOWNERS 批准                             │
│ ❌ 禁止直接推送                                     │
│ ❌ 禁止强制推送                                     │
└─────────────────────────────────────────────────────┘
```

#### develop 分支（保护级别：高）

```
┌─────────────────────────────────────────────────────┐
│             develop 分支保护规则                     │
├─────────────────────────────────────────────────────┤
│ ✅ 必须通过 PR 合并                                 │
│ ✅ 至少 1 个代码审查批准                            │
│ ✅ 基本检查必须通过（lint, unit-tests）             │
│ ❌ 禁止直接推送                                     │
└─────────────────────────────────────────────────────┘
```

---

## 二、提交规范

### 2.1 提交信息格式

采用 **Conventional Commits** 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Type 类型

| Type | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(benchmark): add concurrent test support` |
| `fix` | Bug 修复 | `fix: resolve token counting error in long text` |
| `docs` | 文档更新 | `docs: update API reference for new providers` |
| `style` | 代码格式 | `style: format code with black` |
| `refactor` | 重构 | `refactor(benchmark): simplify runner logic` |
| `perf` | 性能优化 | `perf(tokenizer): optimize tokenization algorithm` |
| `test` | 测试相关 | `test: add unit tests for url_validator` |
| `chore` | 构建/工具 | `chore: update dependencies to latest versions` |
| `ci` | CI 配置 | `ci: add GitHub Actions workflow` |
| `security` | 安全相关 | `security: patch SSRF vulnerability in url validator` |

#### Scope 范围

常用 scope：
- `benchmark` - 基准测试相关
- `provider` - LLM Provider 相关
- `evaluator` - 评估器相关
- `ui` - 用户界面相关
- `config` - 配置相关
- `security` - 安全相关
- `metrics` - 指标相关
- `docs` - 文档

#### 提交示例

```bash
# 简单提交
git commit -m "fix: resolve token count overflow"

# 带范围的提交
git commit -m "feat(provider): add Claude API support"

# 带详细说明的提交
git commit -m "refactor(benchmark): optimize concurrent test runner

- Implement thread-safe metrics collection
- Reduce memory usage by 30%
- Improve test execution speed by 2x

Closes #123"
```

### 2.2 提交最佳实践

**✅ 推荐做法**

```bash
# 每个提交做一件事
git commit -m "fix: resolve token count error"

# 提交信息清晰描述做了什么
git commit -m "feat(ui): add dark mode support"

# 关联 Issue
git commit -m "fix: resolve memory leak

Fixes #456"
```

**❌ 避免的做法**

```bash
# 不要使用模糊的提交信息
git commit -m "update files"

# 不要混合多个不相关的更改
git commit -m "fix bugs and add features"

# 不要使用中文提交信息（除非团队约定）
git commit -m "修复了一个bug"
```

---

## 三、推送规范

### 3.1 推送前检查清单

**代码质量检查**

- [ ] 代码通过 `ruff check` 检查
- [ ] 代码通过 `black --check` 格式检查
- [ ] 运行 `pytest tests/` 确保测试通过
- [ ] 运行 `pytest tests/test_security.py` 确保安全测试通过
- [ ] 更新了相关文档
- [ ] 添加了必要的测试

**功能检查**

- [ ] 功能完整可用
- [ ] 没有遗留的 TODO 或 FIXME
- [ ] 没有调试用的 print 语句
- [ ] 没有硬编码的密钥或敏感信息
- [ ] 配置文件正确更新

**提交检查**

- [ ] 提交信息符合规范
- [ ] 每个提交只做一件事
- [ ] 关联了相关的 Issue

### 3.2 推送流程

#### 功能开发流程

```bash
# 1. 更新本地 main 分支
git checkout main
git pull origin main

# 2. 创建功能分支
git checkout -b feature/your-feature-name

# 3. 进行开发...
# 编辑文件、添加功能等

# 4. 提交更改
git add .
git commit -m "feat(scope): description"

# 5. 同步远程 main（如有更新）
git fetch origin main
git rebase origin/main

# 6. 推送分支
git push -u origin feature/your-feature-name

# 7. 在 GitHub 创建 PR
```

#### Bug 修复流程

```bash
# 1. 从 main 创建修复分支
git checkout main
git pull origin main
git checkout -b fix/issue-description

# 2. 修复并测试
# ... 进行修复 ...

# 3. 提交修复
git add .
git commit -m "fix: resolve issue description

Fixes #123"

# 4. 推送并创建 PR
git push -u origin fix/issue-description
```

#### 紧急修复流程

```bash
# 1. 从 main 直接创建 hotfix 分支
git checkout main
git checkout -b hotfix/critical-security-fix

# 2. 快速修复
git add .
git commit -m "security: patch critical vulnerability"

# 3. 直接推送到 main（需要管理员权限）
git push origin hotfix/critical-security-fix

# 4. 创建 PR 到 main（快速审查后合并）
```

### 3.3 推送频率建议

| 分支类型 | 推送频率 | 说明 |
|----------|----------|------|
| `feature/*` | 每日至少 1 次 | 频繁推送便于备份和协作 |
| `fix/*` | 修复完成后立即推送 | 便于快速审查和合并 |
| `hotfix/*` | 完成后立即推送 | 紧急修复需尽快部署 |
| `develop` | 功能合并后推送 | 保持 develop 最新 |
| `main` | 仅通过 PR 合并 | 禁止直接推送 |

---

## 四、PR 规范

### 4.1 PR 创建

**PR 标题格式**

```
<type>: <简短描述>

示例:
feat: add concurrent benchmark support
fix: resolve token counting overflow
docs: update provider setup guide
```

**PR 描述模板**

```markdown
## 变更摘要
<!-- 简要描述此 PR 的目的 -->

## 变更类型
- [ ] 🐛 Bug 修复
- [ ] ✨ 新功能
- [ ] 💥 破坏性变更
- [ ] 📝 文档更新
- [ ] ⚡ 性能优化
- [ ] 🎨 代码重构
- [ ] ✅ 测试相关
- [ ] 🔒 安全相关

## 测试
- [ ] 添加了新测试
- [ ] 现有测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码符合项目规范
- [ ] 自我审查代码
- [ ] 添加了必要的注释
- [ ] 更新了相关文档

## 相关 Issue
Closes #(issue number)
Related to #(issue number)

## 截图/演示
<!-- 如果适用，添加截图或演示 -->

## 额外说明
<!-- 任何其他审查者需要了解的信息 -->
```

### 4.2 PR 审查标准

**代码质量标准**

- [ ] 代码符合项目风格指南
- [ ] 没有引入新的技术债务
- [ ] 适当的错误处理
- [ ] 必要的注释和文档
- [ ] 测试覆盖率不降低

**功能标准**

- [ ] 功能完整且可用
- [ ] 没有破坏现有功能
- [ ] 性能没有明显下降
- [ ] 安全性没有降低

### 4.3 PR 标签规范

| 标签 | 使用场景 | 自动应用 |
|------|----------|----------|
| `bug` | Bug 修复 | ✅ 自动 |
| `documentation` | 文档更新 | ✅ 自动 |
| `enhancement` | 功能增强 | ✅ 自动 |
| `security` | 安全相关 | ✅ 自动 |
| `ui` | UI 相关 | ✅ 自动 |
| `tests` | 测试相关 | ✅ 自动 |
| `dependencies` | 依赖更新 | ✅ 自动 |
| `performance` | 性能相关 | ✅ 自动 |
| `good first issue` | 适合新手 | 手动 |
| `help wanted` | 需要帮助 | 手动 |
| `work in progress` | 开发中 | 手动 |

---

## 五、文件大小限制

### 5.1 GitHub 限制

| 文件类型 | 大小限制 | 处理方式 |
|----------|----------|----------|
| 单文件 | 100 MB | 使用 Git LFS |
| 仓库总大小 | 建议 < 1 GB | 清理历史或归档 |
| 推送大小 | 建议 < 50 MB/次 | 分批推送 |

### 5.2 应放入 .gitignore 的文件

```gitignore
# 数据文件
*.csv
*.json
*.zip
*.tar.gz
datasets/*/data.zip

# 模型文件
*.pth
*.pkl
*.bin
*.safetensors

# 临时文件
*.tmp
*.log
*.cache
__pycache__/

# IDE 配置
.vscode/
.idea/
*.swp

# 环境变量
.env
.env.local

# 测试结果
test-results/
htmlcov/
.coverage
*.egg-info/
dist/
build/
```

### 5.3 大文件处理策略

**已提交的大文件**

```bash
# 1. 从 git 历史中移除
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large.file" \
  --prune-empty --tag-name-filter cat -- --all

# 2. 清理和垃圾回收
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. 强制推送（谨慎使用）
git push origin --force --all
```

**新增大文件**

```bash
# 1. 安装 Git LFS
git lfs install

# 2. 跟踪大文件类型
git lfs track "*.zip"
git lfs track "*.pkl"

# 3. 提交
git add .gitattributes
git add large_file.zip
git commit -m "feat: add large dataset"
git push
```

---

## 六、常用命令

### 6.1 日常开发

```bash
# 查看状态
git status

# 查看分支
git branch -a

# 切换分支
git checkout <branch>

# 创建并切换分支
git checkout -b <new-branch>

# 添加文件
git add <file>
git add .  # 添加所有更改

# 提交
git commit -m "type: description"

# 推送
git push
git push -u origin <branch>  # 首次推送
```

### 6.2 分支管理

```bash
# 同步远程分支
git fetch origin
git pull origin main

# 变基到最新 main
git rebase origin/main

# 合并分支
git merge feature/xxx

# 删除本地分支
git branch -d feature/xxx

# 删除远程分支
git push origin --delete feature/xxx
```

### 6.3 撤销操作

```bash
# 撤销工作区更改
git restore <file>

# 撤销暂存
git restore --staged <file>

# 撤销最后一次提交
git reset --soft HEAD~1  # 保留更改
git reset --hard HEAD~1  # 丢弃更改

# 撤销已推送的提交（危险）
git revert <commit-hash>
git push
```

### 6.4 查看历史

```bash
# 查看提交历史
git log --oneline --graph --all

# 查看文件历史
git log --follow <file>

# 查看分支图
git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit

# 查看更改统计
git diff --stat
```

---

## 七、故障排查

### 7.1 常见问题

**Q: 推送时提示 "remote rejected"**

```bash
# 原因：分支受保护或 CI 未通过
# 解决：
1. 等待 CI 通过
2. 创建 PR 而非直接推送
3. 联系管理员
```

**Q: 提示 "large files detected"**

```bash
# 原因：文件超过 100MB
# 解决：
1. 移除大文件：git rm --cached large.file
2. 添加到 .gitignore
3. 使用 Git LFS
```

**Q: 合并冲突**

```bash
# 解决步骤：
1. git fetch origin
2. git rebase origin/main
3. 解决冲突
4. git add <resolved-files>
5. git rebase --continue
```

**Q: CI 检查失败**

```bash
# 本地复现：
1. pre-commit run --all-files
2. pytest tests/ -v
3. 修复后重新提交
```

---

## 八、团队协作规范

### 8.1 协作原则

1. **小步快跑**：频繁提交、频繁推送
2. **代码审查**：所有代码必须经过审查
3. **自动化优先**：依赖 CI 而非手动测试
4. **文档同步**：代码和文档同步更新
5. **问题反馈**：及时响应审查意见

### 8.2 沟通规范

| 场景 | 沟通方式 | 响应时间 |
|------|----------|----------|
| PR 审查请求 | GitHub @ 提及 | 24 小时 |
| Bug 报告 | GitHub Issue | 根据优先级 |
| 紧急问题 | 即时通讯 + Issue | 立即 |
| 功能讨论 | GitHub Discussion | 48 小时 |

### 8.3 CODEOWNERS

```github
# 核心代码需要高级审查
core/benchmark_runner.py @senior-dev

# 安全代码需要安全团队审查
core/safe_executor.py @security-team

# 所有更改需要至少一个审查者
* @maintainer-team
```

---

**文档版本:** 1.0
**最后更新:** 2026-01-31
**维护者:** 开发团队
