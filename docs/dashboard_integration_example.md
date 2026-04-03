# 实时仪表盘集成示例
# 
# 在 app.py 中添加此代码以启用实时可视化

## 1. 在文件顶部添加导入
```python
from ui.realtime_dashboard import RealtimeDashboard
from ui.dashboard_components import render_dashboard_ui, create_dashboard_placeholders, update_dashboard_placeholders
```

## 2. 在侧边栏添加可视化选项
```python
st.sidebar.header("📊 可视化设置")
enable_realtime_viz = st.sidebar.checkbox("启用实时可视化", value=True, help="显示实时性能指标和图表")
viz_update_interval = st.sidebar.slider("更新间隔 (秒)", 0.1, 5.0, 0.5, 0.1)
```

## 3. 修改 run_test 函数以集成仪表盘
```python
def run_test(test_function, runner, *args):
    st.session_state.test_running = True
    st.session_state.stop_requested = False
    st.session_state.results_df = pd.DataFrame()
    st.session_state.log_content = []
    df = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # ===== 新增：创建实时仪表盘 =====
    dashboard = None
    dashboard_placeholder = None
    
    if enable_realtime_viz:
        dashboard = RealtimeDashboard(max_points=100)
        st.markdown("---")
        st.header("📊 实时性能监控")
       dashboard_placeholder = st.empty()
    # ==================================
    
    with st.expander("实时请求日志 (Live Request Log)", expanded=True):
        log_placeholder = st.empty()
        log_placeholder.markdown("ℹ️ *日志窗口已初始化... 等待测试开始*")
        
    placeholder = st.empty()

    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        test_type_str = test_type.replace(" ", "_").replace("-", "_")
        
        csv_filename = f"benchmark_results_{model_id}_{test_type_str}_{timestamp}.csv"
        st.session_state.current_csv_file = csv_filename
        
        log_filename = f"benchmark_log_{model_id}_{test_type_str}_{timestamp}.txt"
        st.session_state.current_log_file = log_filename
        
        # ===== 修改：传递 dashboard 参数 =====
        runner_instance = runner(
            placeholder, progress_bar, status_text, 
            api_base_url, model_id, tokenizer_option, 
            csv_filename, api_key, log_placeholder, 
            selected_provider,
            dashboard=dashboard  # 传递仪表盘实例
        )
        # ====================================
        
        # ... (预计算总请求数的代码保持不变) ...
        
        # ===== 新增：定义异步仪表盘更新任务 =====
        async def run_test_with_dashboard():
            """运行测试并定期更新仪表盘"""
            # 创建后台更新任务
            update_task = None
            
            async def update_dashboard_loop():
                while st.session_state.test_running:
                    if dashboard and dashboard_placeholder:
                        with dashboard_placeholder.container():
                            render_dashboard_ui(dashboard)
                    await asyncio.sleep(viz_update_interval)
            
            # 启动仪表盘更新任务
            if enable_realtime_viz:
                update_task = asyncio.create_task(update_dashboard_loop())
            
            # 运行测试
            try:
                result = await test_function(runner_instance, *args)
                return result
            finally:
                # 取消更新任务
                if update_task:
                    update_task.cancel()
                    try:
                        await update_task
                    except asyncio.CancelledError:
                        pass
        
        # 运行测试
        df = asyncio.run(run_test_with_dashboard())
        # =======================================
    
    except asyncio.CancelledError:
        st.warning("测试已由用户停止。")
        df = pd.DataFrame(runner_instance.results_list)
    except Exception as e:
        st.error(f"测试运行时发生错误: {e}")
    finally:
        st.session_state.results_df = df
        st.session_state.test_running = False
        st.session_state.stop_requested = False
        
        if 'runner_instance' in locals():
            st.session_state.log_content = runner_instance.log_content
        
        # ===== 新增：显示最终仪表盘状态 =====
        if dashboard and dashboard_placeholder:
            with dashboard_placeholder.container():
                st.success("✅ 测试完成")
                render_dashboard_ui(dashboard)
        # ====================================
        
        if 'runner_instance' in locals() and runner_instance.results_list:
            status_text.success(f"测试完成！结果已保存到 {st.session_state.current_csv_file}")
        elif st.session_state.stop_requested:
            status_text.warning(f"测试已停止。部分结果已保存到 {st.session_state.current_csv_file}")
        else:
            status_text.error("测试未生成任何结果。")
        
        st.rerun()
```

## 4. 简化版（最小集成）
如果只想要基础的实时指标，可以使用更简单的方式：

```python
# 创建仪表盘
dashboard = RealtimeDashboard()

# 创建占位符
metrics_placeholder = st.empty()

# 在测试循环中定期更新
while testing:
    with metrics_placeholder.container():
        metrics = dashboard.get_metrics()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("完成", metrics['completed'])
        with col2:
            st.metric("失败", metrics['failed'])
        with col3:
            st.metric("平均TTFT", f"{metrics['avg_ttft']:.3f}s")
```

## 5. 启用/禁用可视化
用户可以通过侧边栏的复选框控制是否启用实时可视化，避免在大规模测试时影响性能。
