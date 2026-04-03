# 文档归档记录

> 归档日期: 2026-02-02
> 目的: 统一整理项目文档

---

## 📚 新文档体系

### 根目录文档 (保留)

| 文件 | 用途 | 状态 |
|------|------|------|
| `README.md` | 项目主文档 | ✅ 已更新 |
| `ARCHITECTURE.md` | 架构与文件结构 | ✅ 新建 |
| `DEVELOPMENT_GUIDE.md` | 开发指南 | ✅ 新建 |
| `API.md` | API 参考文档 | ✅ 新建 |
| `DEVELOPMENT.md` | Git 工作流 | ✅ 保留 |

### 新建文档详情

#### ARCHITECTURE.md
- 项目完整结构图
- 代码规模统计 (54,233 行)
- 模块分布和职责说明
- 最大文件 Top 10
- 数据流程图
- 核心接口定义

#### DEVELOPMENT_GUIDE.md
- 快速开始
- 开发环境设置
- 代码规范
- Git 工作流
- 添加新功能指南
- 测试指南
- 调试技巧

#### API.md
- 核心类 API
- Provider API
- Evaluator API
- LLM-as-Judge API
- 工具函数 API
- 配置 API

---

## 📦 归档文档 (移至 docs/archive/)

以下文档内容已整合到新文档体系中，移至归档目录：

| 原文档 | 内容已整合到 | 归档位置 |
|--------|-------------|---------|
| `app_development_doc.md` | ARCHITECTURE.md, DEVELOPMENT_GUIDE.md | docs/archive/ |
| `docs/FILE_ORGANIZATION.md` | ARCHITECTURE.md | docs/archive/ |
| `docs/PROJECT_STATUS.md` | README.md, ROADMAP.md | docs/archive/ |
| `docs/ROADMAP.md` | README.md (路线图部分) | docs/archive/ |
| `docs/GIT_WORKFLOW.md` | DEVELOPMENT.md | docs/archive/ |
| `docs/DOCUMENTATION_UPDATE_SUMMARY.md` | (此归档记录) | docs/archive/ |

---

## 🗑️ 可删除文档 (已过时或重复)

以下文档可考虑删除（内容已完全整合）：

- `app_development_doc.md` - 内容已完全整合到 ARCHITECTURE.md 和 DEVELOPMENT_GUIDE.md
- `docs/FILE_ORGANIZATION.md` - 内容已整合到 ARCHITECTURE.md
- `docs/PROJECT_STATUS.md` - 内容已整合到 README.md

---

## 📋 docs/ 目录组织

```
docs/
├── archive/                    # 归档文档
│   ├── app_development_doc.md
│   ├── FILE_ORGANIZATION.md
│   ├── PROJECT_STATUS.md
│   ├── ROADMAP.md
│   ├── GIT_WORKFLOW.md
│   └── DOCUMENTATION_UPDATE_SUMMARY.md
│
├── planning/                   # 规划文件
│   ├── phase3_plan.md
│   ├── task_plan.md
│   ├── findings.md
│   └── progress.md
│
├── analysis/                   # 分析文件
│   └── ...
│
└── guides/                     # 指南文件
    └── ...
```

---

## ✅ 文档迁移清单

- [x] 创建 ARCHITECTURE.md
- [x] 创建 DEVELOPMENT_GUIDE.md
- [x] 创建 API.md
- [x] 更新 README.md
- [x] 移动旧文档到 docs/archive/
- [ ] 删除重复文档 (需确认)
- [ ] 更新文档交叉引用

---

*最后更新: 2026-02-02*
