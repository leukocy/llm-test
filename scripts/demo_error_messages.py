"""
Error消息增强功能演示

展示新增ErrorTip功能如何Help用户更好地理解Error并获得Solution。
"""

import sys
import io

# Set UTF-8 编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.error_messages import get_enhanced_error

def demo_error_messages():
    """演示各种Error增强Tip"""

    print("=" * 80)
    print("Error消息增强功能演示")
    print("=" * 80)
    print()

    # 1. API 认证Error (401)
    print("【示例 1】API Authentication failed (401)")
    print("-" * 80)
    try:
        raise Exception("HTTP 401: Unauthorized - Invalid API key")
    except Exception as e:
        error_msg = get_enhanced_error(e, context="Model: gpt-4", language="zh")
        print(error_msg)
    print()

    # 2. Request timeout
    print("【示例 2】Request timeout")
    print("-" * 80)
    try:
        raise Exception("Request timeout after 30 seconds")
    except Exception as e:
        error_msg = get_enhanced_error(e, context="Model: deepseek-chat", language="zh")
        print(error_msg)
    print()

    # 3. Rate limited (429)
    print("【示例 3】请求频率超限 (429)")
    print("-" * 80)
    try:
        raise Exception("HTTP 429: Too many requests")
    except Exception as e:
        error_msg = get_enhanced_error(e, context="Model: claude-3", language="zh")
        print(error_msg)
    print()

    # 4. 服务器Error (500)
    print("【示例 4】服务器Error (500)")
    print("-" * 80)
    try:
        raise Exception("HTTP 500: Internal server error")
    except Exception as e:
        error_msg = get_enhanced_error(e, context="API URL: https://api.example.com", language="zh")
        print(error_msg)
    print()

    # 5. 网络ConnectError
    print("【示例 5】网络ConnectError")
    print("-" * 80)
    try:
        raise Exception("Connection refused - Network unreachable")
    except Exception as e:
        error_msg = get_enhanced_error(e, context="API URL: https://api.example.com", language="zh")
        print(error_msg)
    print()

    # 6. 空响应Error
    print("【示例 6】空响应Error")
    print("-" * 80)
    try:
        raise Exception("Empty response received from API")
    except Exception as e:
        error_msg = get_enhanced_error(e, context="Model: gemini-pro", language="zh")
        print(error_msg)
    print()

    print("=" * 80)
    print("演示完成")
    print("=" * 80)


if __name__ == "__main__":
    demo_error_messages()
