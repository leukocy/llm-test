# 开发与 Git 管理守则

本文档旨在为本项目建立一套规范的开发流程和 Git 使用指南，以确保代码质量和历史记录的清晰。

## 1. 核心开发守则

### 分支管理规范
我们采用简化的 **Feature Branch** 工作流：
- **`main`**: 主分支，始终保持可运行、稳定的状态。不要直接在 `main` 上开发新功能。
- **功能分支 (`feature/...`)**: 开发新功能时使用。例如：`feature/add-chart-export`。
- **修复分支 (`fix/...`)**: 修复 Bug 时使用。例如：`fix/typo-in-report`。

### 提交 Commit 规范
提交信息（Commit Message）应当清晰明了，说明“做了什么”。
- **推荐格式**: `Type: Subject`
  - `Feat`: 新功能 (Feature)
  - `Fix`: 修补 Bug
  - `Docs`: 文档修改
  - `Style`: 格式调整（不影响代码运行）
  - `Refactor`: 重构（即不是新增功能，也不是修改bug的代码变动）
- **示例**: `Feat: 添加新的并发测试图表`

---

## 2. Git 常用操作速查

### 从零开始开发一个新功能
1. **确保在最新的主分支上**:
   ```bash
   git checkout main
   git pull origin main  # 如果有远程仓库的话，否则省略
   ```

2. **创建并切换到新分支**:
   ```bash
   git checkout -b feature/你的功能名称
   # 例如: git checkout -b feature/optimize-ui
   ```

3. **写代码...** (修改文件)

4. **查看状态**:
   ```bash
   git status
   ```

5. **添加修改**:
   ```bash
   git add .
   # 或者只添加特定文件: git add app.py
   ```

6. **提交修改**:
   ```bash
   git commit -m "Feat: 描述你完成了什么功能"
   ```

7. **合并回主分支** (功能完成后):
   ```bash
   git checkout main
   git merge feature/你的功能名称
   ```

### 常用命令大白话解释

| 命令 | 作用 | 也就是... |
| :--- | :--- | :--- |
| `git status` | 查看状态 | "我现在改了哪些文件？" |
| `git log` | 查看历史 | "之前都提交了什么版本？" (注意不是 logs) |
| `git add .` | 添加文件 | "把这些修改放入暂存区，准备提交" |
| `git commit -m "..."` | 提交 | "把暂存区的修改永久保存为一个版本" |
| `git checkout -b ...` | 创建分支 | "我要开始做一件新事情，先开个独立空间" |
| `git checkout main` | 切换分支 | "回到主线任务" |
| `git diff` | 查看差异 | "我到底改了这文件的哪一行？" |

---


## 3. 进阶 Git 操作 (提升效率必备)

当您对基础操作熟悉后，这些命令能帮您更灵活地管理代码。

### 暂存手头工作 (Stash)
场景：正在开发功能 A，突然要去修一个紧急 Bug，但功能 A 还没写完不能提交。
- **`git stash`**: 把当前所有未提交的修改“藏”起来，让工作区变干净。
- **`git stash pop`**: 忙完别的后，把刚才藏起来的修改“弹”回来，继续工作。
- **`git stash list`**: 看看藏了多少东西。

### 整理提交历史 (Rebase vs Merge)
场景：主分支 (`main`) 更新了，您的功能分支落后了。
- **`git merge main`**: 传统的合并。安全，但会产生额外的“Merge commit”，让历史线分叉。
- **`git rebase main`**: **变基**。把您的功能分支上的提交，一个个“挪”到最新的 `main` 后面。
  - **优点**: 提交历史是一条直线，非常干净。
  - **注意**: **不要在多人共享的分支上使用 rebase**，只在自己的本地功能分支上用。

#### 交互式整理 (Squash)
场景：开发一个功能提交了 10 次，想在合并前整理成 1 次完美的提交。
- **`git rebase -i HEAD~n`** (n 是要整理的提交数): 进入交互模式，可以把琐碎的 `pick` 改为 `squash` (合并)。

### 后悔药系列 (Reset)
场景：提交错了，或者改乱了。
- **`git reset --soft HEAD^`**: **最常用的后悔药**。撤销最近一次 `commit`，**但保留代码修改**。
  - 也就是回到了 `git add` 之后、`git commit` 之前的状态。有机会重新提交。
- **`git reset --hard HEAD^`**: **强力回退**。彻底回到上一个版本，**丢弃所有改动**。
  - **警告**: 慎用！除非你确定刚才写的代码完全不需要了。
- **`git reset HEAD <file>`**: 取消暂存。把 `git add` 进去的文件退回到工作区（不丢失修改）。

### 检出与漫游 (Checkout & Switch)
- **`git checkout -b <branch>`**: 创建并切换分支（老派写法）。
- **`git switch -c <branch>`**: 创建并切换分支（新版 Git 推荐，语义更清晰）。
- **`git checkout <commit-hash>`**: "穿越时空"。查看项目在过去某个时刻的样子（进入“分离头指针”状态，只能看不能改）。
- **`git checkout <file>`** (或 `git restore <file>`): 丢弃工作区的修改，恢复到上一次提交的状态。

---

## 4. 常见问题救急

