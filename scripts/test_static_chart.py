"""
Test静态图表Generate器

运行此脚本查看Generate静态图表效果
"""

import sys

sys.path.insert(0, '.')

from datetime import datetime

from ui.static_chart_generator import StaticChartGenerator, generate_static_html_report


def test_static_chart():
    """Test静态图表Generate"""

    # 模拟TestResultData（类似 llm-performance-test.html in样例Data）
    test_results = [
        {'input_length': 128, 'prefill_speed': 98.16, 'output_speed': 19.48},
        {'input_length': 256, 'prefill_speed': 120.33, 'output_speed': 19.37},
        {'input_length': 512, 'prefill_speed': 139.43, 'output_speed': 19.22},
        {'input_length': 1024, 'prefill_speed': 139.01, 'output_speed': 19.19},
        {'input_length': 2048, 'prefill_speed': 135.91, 'output_speed': 18.85},
        {'input_length': 4096, 'prefill_speed': 127.63, 'output_speed': 17.92},
        {'input_length': 8192, 'prefill_speed': 112.07, 'output_speed': 15.77},
        {'input_length': 16384, 'prefill_speed': 88.17, 'output_speed': 13.23},
    ]

    # 系统信息
    system_info = {
        'processor': 'AMD EPYC 9355',
        'mainboard': 'no',
        'memory': 'DDR5 6400 12*48G',
        'gpu': 'NVIDIA RTX 5090',
        'system': 'Ubuntu 22.04',
        'engine_name': 'ftllm',
        'model_name': 'DeepSeek-V3.1-int4',
    }

    test_time = datetime.now()

    # Create chartGenerate器
    generator = StaticChartGenerator(dpi=150)

    # 1. Test单Line Chart
    print(" Generate预Pad速度图...")
    input_lengths = [r['input_length'] for r in test_results]
    prefill_speeds = [r['prefill_speed'] for r in test_results]

    fig1 = generator.draw_line_chart(
        input_lengths, prefill_speeds,
        title='预Pad速度',
        x_label='输入长度',
        y_label='预Pad速度 (token/s)',
        line_color='#4bc0c0'
    )
    generator.save_figure_to_file(fig1, 'test_prefill_chart.png')
    print("[OK] Saved: test_prefill_chart.png")

    # 2. Test输出速度图
    print(" Generate输出速度图...")
    output_speeds = [r['output_speed'] for r in test_results]

    fig2 = generator.draw_line_chart(
        input_lengths, output_speeds,
        title='输出速度',
        x_label='输入长度',
        y_label='输出速度 (token/s)',
        line_color='#ff6384'
    )
    generator.save_figure_to_file(fig2, 'test_output_chart.png')
    print("[OK] Saved: test_output_chart.png")

    # 3. Test完整报告图片
    print(" Generate完整性能报告图片...")
    fig3 = generator.create_performance_report_image(
        test_results, system_info, test_time
    )
    generator.save_figure_to_file(fig3, 'test_performance_report.png')
    print("[OK] Saved: test_performance_report.png")

    # 4. Generate HTML 报告
    print(" Generate HTML 报告...")
    html_content = generate_static_html_report(test_results, system_info, test_time)
    with open('test_performance_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("[OK] Saved: test_performance_report.html")

    print("\n 全部Test完成！请查看Generate文件：")
    print("   - test_prefill_chart.png")
    print("   - test_output_chart.png")
    print("   - test_performance_report.png")
    print("   - test_performance_report.html")


if __name__ == '__main__':
    test_static_chart()
