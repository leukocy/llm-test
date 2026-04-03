# 结果统计增强功能集成示例

本文档展示如何将新的可视化增强功能集成到现有报告中。

## 新增模块

### 1. `ui/styled_tables.py`
- `create_styled_summary_table()` - 创建样式化表格
- `add_statistical_summary()` - 添加统计汇总行
- `create_comparison_table()` - 创建对比表格

### 2. `ui/insights.py`
- `generate_performance_insights()` - 自动生成性能洞察
- `get_performance_grade()` - 计算性能评级

### 3. `ui/charts.py` (增强)
- `apply_theme()` - 应用统一主题
- `plot_box_plot()` - 箱线图
- `plot_violin()` - 小提琴图
- `plot_scatter_with_trend()` - 带趋势线的散点图

### 4. `ui/export.py`
- `export_to_excel()` - Excel 导出
- `export_interactive_html()` - HTML 报告导出

---

## 集成到 ui/reports.py

### 步骤 1: 添加导入

在 `ui/reports.py` 文件开头添加：

```python
from ui.styled_tables import create_styled_summary_table, add_statistical_summary
from ui.insights import generate_performance_insights, get_performance_grade
from ui.charts import apply_theme, plot_box_plot
from ui.export import export_to_excel, create_excel_download_link
```

### 步骤 2: 增强并发测试报告

在 `generate_concurrency_report()` 函数中进行以下修改：

#### 2.1 替换简单表格为样式化表格

**原代码** (约第 67 行):
```python
st.dataframe(summary[display_columns].round(4), use_container_width=True)
```

**替换为**:
```python
# 创建样式化表格
styled_table = create_styled_summary_table(
    summary[display_columns].round(4),
    highlight_cols=['Best_TTFT', 'Max_System_Throughput', 'Max_Single_TPS'],
    highlight_best=True
)
st.dataframe(styled_table, use_container_width=True)
```

#### 2.2 添加性能洞察

在表格后面添加 (约第 68 行后):
```python
# 生成性能洞察
insights = generate_performance_insights(summary, 'concurrency', model_id)
if insights:
    with st.expander("📊 性能洞察与分析", expanded=True):
        grade, color, description = get_performance_grade(insights)
        st.markdown(f"### 综合评级: {grade} <span style='color:{color}'>({description})</span>", unsafe_allow_html=True)
        st.markdown("---")
        for insight in insights:
            st.markdown(insight)
```

#### 2.3 应用统一主题到图表

在每个图表创建后应用主题 (约第 74-86 行):
```python
fig1 = plot_plotly_bar(summary, 'concurrency_str', 'Best_TTFT', ...)
if fig1:
    fig1 = apply_theme(fig1)  # 添加这一行
    st.plotly_chart(fig1, use_container_width=True)
```

对 `fig2` 和 `fig3` 也做同样处理。

#### 2.4 添加导出功能

在报告末尾添加 (约第 88 行后):
```python
# 导出选项
with st.expander("📥 导出报告", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel 导出
        excel_data = {
            '汇总数据': summary[display_columns].round(4)
        }
        excel_link = create_excel_download_link(
            excel_data,
            filename=f'concurrency_report_{model_id}.xlsx'
        )
        if excel_link:
            st.markdown(excel_link, unsafe_allow_html=True)
    
    with col2:
        # Markdown 导出保持原样
        pass
```

---

### 步骤 3: 增强 Prefill 测试报告

在 `generate_prefill_report()` 中进行类似修改：

#### 3.1 样式化表格 (约第 132 行)

```python
styled_table = create_styled_summary_table(
    summary[display_columns].round(4),
    highlight_cols=['Best_TTFT', 'Max_Prefill_Speed'],
    highlight_best=True
)
st.dataframe(styled_table, use_container_width=True)
```

#### 3.2 性能洞察 (约第 133 行后)

```python
insights = generate_performance_insights(summary, 'prefill', model_id)
if insights:
    with st.expander("📊 性能洞察", expanded=True):
        for insight in insights:
            st.markdown(insight)
```

#### 3.3 添加箱线图 (约第 147 行后，可选)

```python
# 在原有图表后添加分布分析
with st.expander("📦 性能分布分析 (箱线图)", expanded=False):
    # 准备数据 - 将原始测试数据传入
    if 'ttft_distribution_data' in locals():  # 需要原始数据，不是汇总数据
        box_fig = plot_box_plot(
            ttft_distribution_data,
            x='input_tokens_target',
            y='ttft',
            title="TTFT 分布分析",
            xlabel="输入 Token 数",
            ylabel="TTFT (秒)"
        )
        if box_fig:
            st.plotly_chart(box_fig, use_container_width=True)
```

---

### 步骤 4: 增强长上下文测试报告

在 `generate_long_context_report()` 中：

#### 4.1 样式化表格 (约第 215 行)

```python
styled_table = create_styled_summary_table(
    summary[display_columns].round(4),
    highlight_cols=['Best_TTFT', 'Max_Prefill_Speed', 'Max_TPS'],
    highlight_best=True
)
st.dataframe(styled_table, use_container_width=True)
```

#### 4.2 性能洞察 (约第 216 行后)

```python
insights = generate_performance_insights(summary, 'long_context', model_id)
if insights:
    with st.expander("📊 性能洞察", expanded=True):
        for insight in insights:
            st.markdown(insight)
```

---

### 步骤 5: 增强矩阵测试报告

在 `generate_matrix_report()` 中：

#### 5.1 样式化表格 (约第 305 行)

```python
styled_table = create_styled_summary_table(
    summary,
    highlight_cols=['Best_TTFT', 'Max_System_Throughput', 'Max_Prefill_Speed'],
    highlight_best=True
)
st.dataframe(styled_table, use_container_width=True)
```

#### 5.2 性能洞察 (约第 306 行后)

```python
insights = generate_performance_insights(summary, 'matrix', model_id)
if insights:
    with st.expander("📊 综合性能洞察", expanded=True):
        grade, color, description = get_performance_grade(insights)
        st.markdown(f"**评级**: {grade} ({description})")
        st.markdown("---")
        for insight in insights:
            st.markdown(insight)
```

---

## 完整示例：增强版并发报告函数

```python
def generate_concurrency_report_enhanced(df_group, model_id):
    """Enhanced version with styling and insights."""
    from ui.styled_tables import create_styled_summary_table
    from ui.insights import generate_performance_insights, get_performance_grade
    from ui.charts import apply_theme
    from ui.export import create_excel_download_link
    
    st.subheader("并发性能测试图表")
    report_md = "## 并发性能测试\n\n"
    
    # ... (现有的数据处理代码保持不变) ...
    
    # === 新增：样式化表格 ===
    st.subheader(f"最佳性能统计 (每级 {rounds_per_level} 轮测试中)")
    
    styled_table = create_styled_summary_table(
        summary[display_columns].round(4),
        highlight_cols=['Best_TTFT', 'Max_System_Throughput', 'Max_Single_TPS'],
        highlight_best=True
    )
    st.dataframe(styled_table, use_container_width=True)
    
    # === 新增：性能洞察 ===
    insights = generate_performance_insights(summary, 'concurrency', model_id)
    if insights:
        with st.expander("📊 性能洞察与分析", expanded=True):
            grade, color, description = get_performance_grade(insights)
            st.markdown(f"**综合评级**: <span style='color:{color};font-size:20px;font-weight:bold'>{grade}</span> - {description}", unsafe_allow_html=True)
            st.markdown("---")
            for insight in insights:
                st.markdown(insight)
    
    # === 修改：应用统一主题到图表 ===
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = plot_plotly_bar(summary, 'concurrency_str', 'Best_TTFT', ...)
        if fig1:
            fig1 = apply_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = plot_plotly_bar(summary, 'concurrency_str', 'Max_System_Throughput', ...)
        if fig2:
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        fig3 = plot_plotly_bar(summary, 'concurrency_str', 'Max_Single_TPS', ...)
        if fig3:
            fig3 = apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
    
    # === 新增：导出功能 ===
    with st.expander("📥 导出报告", expanded=False):
        excel_data = {'并发测试结果': summary[display_columns].round(4)}
        excel_link = create_excel_download_link(
            excel_data,
            filename=f'concurrency_report_{model_id}_{time.strftime("%Y%m%d")}.xlsx'
        )
        if excel_link:
            st.markdown(excel_link, unsafe_allow_html=True)
    
    return report_md
```

---

## 注意事项

1. **渐进式集成**: 可以先集成样式化表格和洞察，测试通过后再添加导出功能
2. **错误处理**:  所有新功能都包含try-except，不会影响现有功能
3. **依赖安装**: 运行 `pip install -r requirements.txt` 安装新依赖
4. **性能影响**: 样式化表格对大数据集(>1000行)可能较慢，可选择性启用

---

## 测试建议

1. 先在一个测试类型上完整集成（推荐从并发测试开始）
2. 运行测试确保无错误
3. 验证表格样式、洞察准确性
4. 逐步推广到其他测试类型

---

## 后续扩展（可选）

1. 添加统计汇总行：使用 `add_statistical_summary(df, metric_cols)`
2. 对比分析：使用 `create_comparison_table(df1, df2)`
3. 更多图表类型：箱线图、小提琴图、散点图

---

需要帮助集成？参考本文档或查看新模块的docstring！
