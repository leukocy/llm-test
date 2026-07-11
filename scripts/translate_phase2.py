"""
Phase 2: Aggressive Chinese-to-English translation using regex patterns.
Targets docstrings, comments, and common patterns across ALL files.
"""

import os
import re


def translate_line(line):
    """Translate a single line's Chinese content to English using pattern matching."""
    original = line

    # Skip lines that are just imports or blank
    stripped = line.strip()
    if not stripped or stripped.startswith("import ") or stripped.startswith("from "):
        return line

    # ===== DOCSTRING/COMMENT TRANSLATIONS =====
    # These are common patterns found across the codebase

    # Module-level docstring headers
    docstring_map = {
        "测试面板模块": "Test Panels Module",
        "统一测试控制面板模块": "Unified Test Control Panel Module",
        "测试执行模块": "Test Execution Module",
        "页面布局模块": "Page Layout Module",
        "新手引导系统模块": "Onboarding System Module",
        "图表模块": "Charts Module",
        "报告模块": "Reports Module",
        "导出模块": "Export Module",
        "日志查看器模块": "Log Viewer Module",
        "评测仪表板模块": "Evaluation Dashboard Module",
        "历史记录浏览器模块": "History Browser Module",
        "批量测试模块": "Batch Test Module",
        "高级面板模块": "Advanced Panels Module",
        "实时仪表板模块": "Realtime Dashboard Module",
        "仪表板组件模块": "Dashboard Components Module",
        "数据集管理器模块": "Dataset Manager Module",
        "对比页面模块": "Comparison Page Module",
        "思考组件模块": "Thinking Components Module",
        "静态图表生成器模块": "Static Chart Generator Module",
        "样式表格模块": "Styled Tables Module",
        "格式化器模块": "Formatters Module",
        "配置设置模块": "Configuration Settings Module",
        "测试配置加载器模块": "Test Config Loader Module",
        "基准运行器模块": "Benchmark Runner Module",
        "错误消息模块": "Error Messages Module",
        "指标模块": "Metrics Module",
        "测试运行器模块": "Test Runner Module",
        "测试配置模块": "Test Config Module",
        "任务配置模块": "Task Config Module",
        "系统信息模块": "System Info Module",
        "标准报告模块": "Standard Report Module",
        "智能回答解析器模块": "Smart Answer Parser Module",
        "提示模板模块": "Prompt Template Module",
        "响应解析器模块": "Response Parser Module",
        "增强解析器模块": "Enhanced Parser Module",
        "增强评估器模块": "Enhanced Evaluator Module",
        "评估报告模块": "Evaluation Report Module",
        "失败分析模块": "Failure Analysis Module",
        "模型比较器模块": "Model Comparator Module",
        "质量评估器模块": "Quality Evaluator Module",
        "推理评估器模块": "Reasoning Evaluator Module",
        "结果比较器模块": "Result Comparator Module",
        "一致性测试器模块": "Consistency Tester Module",
        "鲁棒性测试器模块": "Robustness Tester Module",
        "思考参数模块": "Thinking Params Module",
        "请求日志模块": "Request Logger Module",
        "响应缓存模块": "Response Cache Module",
        "重试处理器模块": "Retry Handler Module",
        "数据集管理器": "Dataset Manager",
        "Tokenizer 工具模块": "Tokenizer Utils Module",
        "数据库管理器模块": "Database Manager Module",
        "数据库连接模块": "Database Connection Module",
        "数据库 Schema 模块": "Database Schema Module",
        "数据库备份模块": "Database Backup Module",
        "数据库迁移模块": "Database Migrations Module",
        "数据导出服务模块": "Data Export Service Module",
        "数据导入服务模块": "Data Import Service Module",
        "帮助工具模块": "Helper Utilities Module",
        "日志模块": "Logger Module",
        "自定义配置模块": "Custom Config Module",
        "测试配置管理器模块": "Test Config Manager Module",
        "日志服务器模块": "Log Server Module",
        "数据集下载器模块": "Dataset Downloader Module",
        "基准测试运行器": "Benchmark Runner",
        "质量评估器": "Quality Evaluator",
        "推理评估器": "Reasoning Evaluator",
        "结果比较器": "Result Comparator",
        "一致性测试器": "Consistency Tester",
        "鲁棒性测试器": "Robustness Tester",
        "模型比较器": "Model Comparator",
        "失败分析器": "Failure Analyzer",
        "标准报告生成器": "Standard Report Generator",
        "评估报告生成器": "Evaluation Report Generator",
        "LLM 评判模块": "LLM Judge Module",
        "LLM 法官模块": "LLM Judge Module",
        "认证报告模块": "Certification Report Module",
        "数据集加载器模块": "Dataset Loader Module",
        "历史管理器模块": "History Manager Module",
    }

    for cn, en in docstring_map.items():
        line = line.replace(cn, en)

    # Common phrases in docstrings/comments
    phrase_map = {
        # Function/class descriptions
        "渲染所有测试面板": "Render all test panels",
        "渲染测试面板": "Render test panel",
        "渲染报告展示区域": "Render report display area",
        "渲染结果展示区域": "Render results display area",
        "渲染日志查看区域": "Render log viewer area",
        "渲染页面头部": "Render page header",
        "渲染空状态提示": "Render empty state prompt",
        "主页面布局渲染": "Main page layout render",
        "应用自定义 CSS 样式": "Apply custom CSS styles",
        "当前选择的测试类型": "Currently selected test type",
        "测试执行函数": "Test execution function",
        "如果测试被触发返回": "Returns True if test was triggered",
        "提供页面布局和导航功能": "Provides page layout and navigation",
        "提供各种测试类型的配置面板": "Provides configuration panels for various test types",
        "提供集中化的测试控制界面": "Provides centralized test control interface",
        "提供测试执行流程封装": "Provides test execution workflow",
        "提供首次使用时的帮助和引导": "Provides first-time use help and guidance",
        # Parameter descriptions
        "包括：": "including:",
        "参数：": "Parameters:",
        "返回：": "Returns:",
        "返回值：": "Return value:",
        "示例：": "Example:",
        "注意：": "Note:",
        "注意事项：": "Notes:",
        "异常：": "Raises:",
        "说明：": "Description:",
        # Functional descriptions
        "CSS 样式定义": "CSS style definitions",
        "页面头部": "Page header",
        "结果展示区域": "Results display area",
        "报告展示": "Report display",
        "统一的测试启动按钮": "Unified test start button",
        "测试进度显示": "Test progress display",
        "测试状态监控": "Test status monitoring",
        "测试控制操作": "Test control operations",
        "功能介绍": "Feature introduction",
        "分步教程": "Step-by-step tutorial",
        "快速开始指南": "Quick start guide",
        "常见问题解答": "FAQ",
        "引导状态管理": "Onboarding state management",
        "测试状态管理": "Test state management",
        "进度显示": "Progress display",
        "实时日志": "Real-time logging",
        "系统信息捕获": "System info capture",
        "测试结果存储": "Result storage",
        # Error messages
        "数据中缺少": "Missing in data",
        "列。": "column.",
        "没有有效的": "No valid",
        "测试数据": "test data",
        "长上下文数据": "long context data",
        "并发数据": "concurrency data",
        # Help text
        "严格校准输入 Prompt 的 Token 长度": "Strictly calibrate input prompt token length",
        "自动根据长度匹配指令后缀": "Auto-match instruction suffix based on length",
        "越短越好": "lower is better",
        "越高越好": "higher is better",
        "越大越好": "higher is better",
        "越小越好": "lower is better",
        # Status messages
        "正在启动并发测试...": "Starting concurrency test...",
        "正在启动 Prefill 测试...": "Starting Prefill test...",
        "正在启动长上下文测试...": "Starting long context test...",
        "正在启动分段测试...": "Starting segmented test...",
        "正在启动综合测试...": "Starting matrix test...",
        "正在启动自定义测试...": "Starting custom test...",
        "正在启动全部测试...": "Starting all tests...",
        "正在启动稳定性测试...": "Starting stability test...",
        # Common labels
        "配置预设": "Config Presets",
        "保存当前配置": "Save Current Config",
        "报告信息配置": "Report Info Config",
        "帮助/引导": "Help/Guide",
        "模型配置": "Model Configuration",
        "系统校准": "System Calibration",
        "随机种子": "Random Seed",
        "测试类型": "Test Type",
        "测试参数": "Test Parameters",
        "参数设置": "Parameter Settings",
        "高级设置": "Advanced Settings",
        "基本设置": "Basic Settings",
        "自定义配置": "Custom Configuration",
        "内置预设": "Built-in Presets",
        "用户预设": "User Presets",
        # Buttons/Actions
        "开始测试": "Start Test",
        "停止测试": "Stop Test",
        "暂停测试": "Pause Test",
        "恢复测试": "Resume Test",
        "下载报告": "Download Report",
        "下载结果": "Download Results",
        "导出数据": "Export Data",
        "重新运行": "Re-run",
        "清空结果": "Clear Results",
        "刷新": "Refresh",
        "确认": "Confirm",
        "取消": "Cancel",
        "关闭": "Close",
        "展开": "Expand",
        "收起": "Collapse",
        "应用": "Apply",
        "重置": "Reset",
        # Technical terms
        "缓存命中率": "Cache hit rate",
        "系统吞吐量": "System throughput",
        "输出吞吐量": "Output throughput",
        "输入吞吐量": "Input throughput",
        "总吞吐量": "Total throughput",
        "首字延迟": "Time to first token",
        "请求成功率": "Request success rate",
        "解码速度": "Decode speed",
        "生成速度": "Generation speed",
        "处理速度": "Processing speed",
        "总请求数": "Total requests",
        "成功请求数": "Successful requests",
        "失败请求数": "Failed requests",
        "错误率": "Error rate",
        "重试次数": "Retry count",
        "超时次数": "Timeout count",
        "平均响应时间": "Average response time",
        "中位数": "Median",
        "标准差": "Standard deviation",
        "变异系数": "Coefficient of variation",
        # Labels
        "预设名称": "Preset name",
        "预设描述": "Preset description",
        "配置名称": "Config name",
        "自定义名称": "Custom name",
        "文件名": "Filename",
        "文件路径": "File path",
        "创建时间": "Created at",
        "更新时间": "Updated at",
        "持续时间": "Duration",
        "开始时间": "Start time",
        "结束时间": "End time",
        "供应商": "Provider",
        "模型名称": "Model name",
        "模型 ID": "Model ID",
        "基础 URL": "Base URL",
        # Misc
        "向后兼容": "Backward compatibility",
        "创建默认实例": "Create default instance",
        "隐藏 Streamlit 默认元素": "Hide Streamlit default elements",
        "自定义样式": "Custom styles",
        "滚动条样式": "Scrollbar styles",
        "自动检测测试类型并生成相应报告": "Auto-detect test type and generate corresponding report",
        "结果数据请查看上方表格": "Please refer to the data table above for results",
        "统一格式化展示": "Unified formatted display",
        "显示格式化后的数据框": "Display formatted dataframe",
        "展开查看原始完整数据": "Expand to view full raw data",
        "下载按钮": "Download button",
        "始终下载完整原始数据": "Always download full raw data",
        "完整数据": "Full data",
        "显示报告": "Display report",
        "在左侧侧边栏配置测试参数，然后选择测试类型开始测试": "Configure test parameters in the left sidebar, then select a test type to begin",
        "功能特点": "Features",
        "新特性": "New Features",
        "模块化架构": "Modular architecture",
        "更清晰的代码组织": "Cleaner code organization",
        "更好的可维护性": "Better maintainability",
        "应用样式": "Apply styles",
        "渲染头部": "Render header",
        "渲染报告区域": "Render report section",
        "统计结果": "Statistical results",
        "移到前面": "Render first",
        "渲染结果区域": "Render results section",
        "原始数据": "Raw data",
        "移到后面": "Render after",
        "渲染日志区域": "Render log section",
        "打开日志查看器": "Open Log Viewer",
        # Database related
        "数据库": "Database",
        "数据库连接": "Database connection",
        "数据库管理": "Database management",
        "数据导出": "Data export",
        "数据导入": "Data import",
        "备份": "Backup",
        "恢复": "Restore",
        "迁移": "Migration",
        "表结构": "Table schema",
        "索引": "Index",
        "查询": "Query",
        "插入": "Insert",
        "更新": "Update",
        "删除": "Delete",
        "事务": "Transaction",
        # Evaluator related
        "评估器": "Evaluator",
        "评测": "Evaluation",
        "评分": "Score",
        "正确答案": "Correct answer",
        "模型回答": "Model answer",
        "解析结果": "Parse result",
        "评估结果": "Evaluation result",
        "测试用例": "Test case",
        "测试集": "Test set",
        "数据集": "Dataset",
        "样本数": "Sample count",
        "准确率": "Accuracy",
        "得分": "Score",
        "通过率": "Pass rate",
        "考试": "Exam",
        "题目": "Question",
        "选项": "Options",
        "答案": "Answer",
        # Insights
        "性能洞察": "Performance insights",
        "综合评级": "Overall grade",
        "评级": "Grade",
        "优秀": "Excellent",
        "良好": "Good",
        "一般": "Average",
        "较差": "Poor",
        "极差": "Very Poor",
        "不及格": "Failing",
        "及格": "Passing",
        "警告": "Warning",
        "提示": "Tip",
        "建议": "Suggestion",
        # Thinking/Reasoning
        "思考过程": "Thinking process",
        "推理过程": "Reasoning process",
        "推理链": "Reasoning chain",
        "思考链": "Chain of thought",
        "思考内容": "Thinking content",
        "思考参数": "Thinking parameters",
        "思考模式": "Thinking mode",
        "思考预算": "Thinking budget",
        "思考 Token": "Thinking tokens",
        # Certification
        "认证报告": "Certification report",
        "认证结果": "Certification result",
        "认证状态": "Certification status",
        # Download
        "下载中": "Downloading",
        "下载完成": "Download complete",
        "下载失败": "Download failed",
        "正在下载": "Downloading",
        # Misc
        "请稍候": "Please wait",
        "加载中": "Loading",
        "处理中": "Processing",
        "已保存": "Saved",
        "未找到": "Not found",
        "不支持": "Not supported",
        "暂不支持": "Not yet supported",
        "即将推出": "Coming soon",
        "无数据": "No data",
        "无结果": "No results",
        "分钟": "minutes",
        "小时": "hours",
        "秒钟": "seconds",
        "秒": "seconds",
        "个请求": " requests",
        "个测试": " tests",
        "个样本": " samples",
        "个模型": " models",
        "项": " items",
    }

    for cn, en in phrase_map.items():
        line = line.replace(cn, en)

    return line


def translate_file_aggressive(filepath):
    """Apply aggressive line-by-line translation to a file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except:
        return -1

    new_lines = [translate_line(line) for line in lines]

    content = "".join(new_lines)
    original = "".join(lines)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        remaining = len(re.findall(r"[\u4e00-\u9fff]+", content))
        return remaining
    return -1


# Process all Python files
total_translated = 0
total_remaining = 0

for root, _, files in os.walk("."):
    if ".git" in root or "__pycache__" in root or ".pytest_cache" in root:
        continue
    for f in files:
        if f.endswith(".py") and f not in (
            "find_chinese.py",
            "translate_reports.py",
            "translate_all.py",
            "translate_phase2.py",
        ):
            path = os.path.join(root, f)
            try:
                with open(path, encoding="utf-8") as fh:
                    if not re.search(r"[\u4e00-\u9fff]", fh.read()):
                        continue
            except:
                continue

            remaining = translate_file_aggressive(path)
            if remaining >= 0:
                total_translated += 1
                total_remaining += remaining
                status = f"({remaining} remaining)" if remaining > 0 else "(clean)"
                print(f"  {path} {status}")

print(f"\nPhase 2 translated {total_translated} files. {total_remaining} segments remaining.")
