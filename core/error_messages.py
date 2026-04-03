"""
Error消息增强模块

提供详细、友好ErrorTip，支持in英文双语Display，并给出SolutionSuggestion。
"""

from typing import Optional, Dict, Any


class ErrorMessages:
    """Error消息增强类，提供in英文双语ErrorTipandSolution"""

    # API Error消息映射
    API_ERRORS = {
        # 网络ConnectError
        "connection_error": {
            "zh": "网络Connection failed",
            "en": "Network Connection Failed",
            "details_zh": "no法Connect到 API 服务器。Please check网络Connectand API URL。",
            "details_en": "Unable to connect to the API server. Please check your network connection and API URL.",
            "solutions_zh": [
                "Check API Base URL is否正确",
                "Confirm网络Connect正常",
                "Check防火墙Set",
                "尝试use VPN or代理",
                "Validate API 服务is否in线"
            ],
            "solutions_en": [
                "Verify the API Base URL is correct",
                "Ensure network connection is working",
                "Check firewall settings",
                "Try using VPN or proxy",
                "Verify API service is online"
            ]
        },

        # API Authentication failed (401)
        "authentication_error": {
            "zh": "API Authentication failed (401)",
            "en": "API Authentication Failed (401)",
            "details_zh": "API Key no效or已过期。Please check您 API 密钥。",
            "details_en": "API Key is invalid or expired. Please check your API key.",
            "solutions_zh": [
                "Confirm API Key is否正确输入",
                "Check API Key is否已过期",
                "重新Generate API Key",
                "Confirm API Key has访问此Model权限",
                "Check账户余额is否充足"
            ],
            "solutions_en": [
                "Verify API Key is entered correctly",
                "Check if API Key has expired",
                "Regenerate API Key",
                "Confirm API Key has permission to access this model",
                "Check if account balance is sufficient"
            ]
        },

        # Request timeout
        "timeout_error": {
            "zh": "Request timeout",
            "en": "Request Timeout",
            "details_zh": "API 请求in规定时间内未完成响应。",
            "details_en": "API request did not complete within the timeout period.",
            "solutions_zh": [
                "Check网络Connect稳定性",
                "减少请求 Token 数量",
                "增加超时时间Set",
                "稍后重试",
                "Check API 服务is否负载过高"
            ],
            "solutions_en": [
                "Check network connection stability",
                "Reduce the number of tokens in request",
                "Increase timeout setting",
                "Try again later",
                "Check if API service is under heavy load"
            ]
        },

        # Rate limited (429)
        "rate_limit_error": {
            "zh": "请求频率超限 (429)",
            "en": "Rate Limit Exceeded (429)",
            "details_zh": "请求过于频繁，超过 API Rate limited。",
            "details_en": "Too many requests, exceeding the API rate limit.",
            "solutions_zh": [
                "降低Concurrency请求数",
                "in请求之间AddLatency",
                "etc.待几minutes后重试",
                "升级 API 套餐以获得更高Rate limited",
                "Check账户Remaining配额"
            ],
            "solutions_en": [
                "Reduce concurrent requests",
                "Add delays between requests",
                "Wait a few minutes and retry",
                "Upgrade API plan for higher rate limit",
                "Check remaining account quota"
            ]
        },

        # 服务器Error (5xx)
        "server_error": {
            "zh": "服务器Error (5xx)",
            "en": "Server Error (5xx)",
            "details_zh": "API 服务器遇到Internal error。",
            "details_en": "API server encountered an internal error.",
            "solutions_zh": [
                "etc.待几minutes后重试",
                "Check API 服务Status页面",
                "联系 API 提供商支持",
                "尝试切换到otherModel",
                "查看 API 服务is否currently维护"
            ],
            "solutions_en": [
                "Wait a few minutes and retry",
                "Check API service status page",
                "Contact API provider support",
                "Try switching to another model",
                "Check if API service is under maintenance"
            ]
        },

        # 空响应Error
        "empty_response_error": {
            "zh": "收到空响应",
            "en": "Empty Response Received",
            "details_zh": "API Return空内容orno效响应。",
            "details_en": "API returned empty content or invalid response.",
            "solutions_zh": [
                "Check Prompt is否被Model拒绝",
                "尝试修改 Prompt 内容",
                "CheckModelis否支持此类型请求",
                "Confirm API 参数Configure正确",
                "尝试usenot同Model"
            ],
            "solutions_en": [
                "Check if prompt was rejected by model",
                "Try modifying the prompt content",
                "Check if model supports this type of request",
                "Verify API parameters are configured correctly",
                "Try using a different model"
            ]
        },

        # JSON Parse error
        "json_decode_error": {
            "zh": "响应Parse failed",
            "en": "Response Parsing Failed",
            "details_zh": "no法Parse API Return JSON Data。",
            "details_en": "Unable to parse JSON data returned by API.",
            "solutions_zh": [
                "Check API Return格式is否标准",
                "Confirmuseis兼容 API 端点",
                "联系 API 提供商Check服务Status",
                "尝试not同 API 端点"
            ],
            "solutions_en": [
                "Check if API returns standard format",
                "Confirm using compatible API endpoint",
                "Contact API provider to check service status",
                "Try a different API endpoint"
            ]
        },

        # Data集LoadError
        "dataset_load_error": {
            "zh": "DatasetLoad failed",
            "en": "Dataset Loading Failed",
            "details_zh": "no法LoadorParsetest data集。",
            "details_en": "Unable to load or parse test dataset.",
            "solutions_zh": [
                "CheckDatasetFile pathis否正确",
                "ConfirmDataset文件格式正确 (JSON/CSV)",
                "ValidateDataset文件未损坏",
                "Check文件读取权限",
                "尝试重新under载Dataset"
            ],
            "solutions_en": [
                "Check if dataset file path is correct",
                "Confirm dataset file format is correct (JSON/CSV)",
                "Verify dataset file is not corrupted",
                "Check file read permissions",
                "Try re-downloading the dataset"
            ]
        },

        # Tokenizer LoadError
        "tokenizer_load_error": {
            "zh": "Tokenizer Load failed",
            "en": "Tokenizer Loading Failed",
            "details_zh": "no法Load指定 Tokenizer Model。",
            "details_en": "Unable to load the specified Tokenizer model.",
            "solutions_zh": [
                "Check HuggingFace Model ID is否正确",
                "Confirm网络can访问 HuggingFace",
                "尝试useother Tokenizer",
                "Check本地磁盘空间",
                "use 'API (usage field)' 作is替代方案"
            ],
            "solutions_en": [
                "Check if HuggingFace model ID is correct",
                "Confirm network can access HuggingFace",
                "Try using a different Tokenizer",
                "Check local disk space",
                "Use 'API (usage field)' as an alternative"
            ]
        },

        # ResultSaveError
        "result_save_error": {
            "zh": "ResultSave failed",
            "en": "Result Saving Failed",
            "details_zh": "no法willTest ResultsSave到文件。",
            "details_en": "Unable to save test results to file.",
            "solutions_zh": [
                "Check输出目录is否存in",
                "Confirmhas写入文件权限",
                "Check磁盘空间is否充足",
                "ValidateFile path格式正确",
                "尝试usenot同输出路径"
            ],
            "solutions_en": [
                "Check if output directory exists",
                "Confirm have write permission",
                "Check if disk space is sufficient",
                "Verify file path format is correct",
                "Try using a different output path"
            ]
        }
    }

    @classmethod
    def get_error_message(
        cls,
        error_type: str,
        original_error: Optional[str] = None,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        Get增强Error消息

        Args:
            error_type: Error type (如 "authentication_error", "timeout_error")
            original_error: 原始Error消息
            language: 语言偏好 ("zh" or "en")

        Returns:
            包含Error详细信息字典
            {
                "title": str,          # Error标题
                "details": str,        # 详细描述
                "solutions": list,     # Solution列表
                "original": str        # 原始Error
            }
        """
        error_info = cls.API_ERRORS.get(error_type, cls._get_default_error())

        title = error_info.get(f"title_{language}", error_info.get("title_zh", error_type))
        details = error_info.get(f"details_{language}", error_info.get("details_zh", ""))
        solutions = error_info.get(f"solutions_{language}", error_info.get("solutions_zh", []))

        result = {
            "title": title,
            "details": details,
            "solutions": solutions,
            "original": original_error or ""
        }

        return result

    @classmethod
    def _get_default_error(cls) -> Dict[str, Any]:
        """ReturndefaultError message"""
        return {
            "title_zh": "未知Error",
            "title_en": "Unknown Error",
            "details_zh": "发生未知Error。",
            "details_en": "An unknown error occurred.",
            "solutions_zh": ["Please checkLog以Get更多信息", "尝试重启Apply程序"],
            "solutions_en": ["Check logs for more information", "Try restarting the application"]
        }

    @classmethod
    def format_for_display(cls, error_data: Dict[str, Any], language: str = "zh") -> str:
        """
        FormatError消息用于Display

        Args:
            error_data: ErrorData字典
            language: 语言偏好

        Returns:
            FormatError消息字符串
        """
        separator = "\n"
        bullet = "• " if language == "en" else "• "

        msg_parts = [
            f"❌ {error_data['title']}",
            f"\n{error_data['details']}",
        ]

        if error_data['solutions']:
            solutions_title = "Solution:" if language == "zh" else "Solutions:"
            msg_parts.append(f"\n\n{solutions_title}")
            for solution in error_data['solutions']:
                msg_parts.append(f"{bullet}{solution}")

        if error_data['original']:
            original_title = "原始Error:" if language == "zh" else "Original Error:"
            msg_parts.append(f"\n\n{original_title}\n`{error_data['original']}`")

        return separator.join(msg_parts)

    @classmethod
    def detect_and_enhance_error(
        cls,
        error: Exception,
        context: Optional[str] = None,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        自动检测Error type并Return增强Error message

        Args:
            error: 异常对象
            context: 额外onunder文信息
            language: 语言偏好

        Returns:
            增强Error message字典
        """
        error_str = str(error).lower()
        error_type = "unknown"

        # 检测Error type
        if any(code in error_str for code in ["401", "unauthorized", "authentication"]):
            error_type = "authentication_error"
        elif any(code in error_str for code in ["429", "rate limit", "too many requests"]):
            error_type = "rate_limit_error"
        elif any(code in error_str for code in ["timeout", "timed out"]):
            error_type = "timeout_error"
        elif any(code in error_str for code in ["500", "502", "503", "504", "server error"]):
            error_type = "server_error"
        elif any(code in error_str for code in ["connection", "connect", "network"]):
            error_type = "connection_error"
        elif "empty" in error_str:
            error_type = "empty_response_error"
        elif "json" in error_str:
            error_type = "json_decode_error"

        # Get增强Error message
        error_data = cls.get_error_message(error_type, str(error), language)

        # Addonunder文信息
        if context:
            context_prefix = "onunder文: " if language == "zh" else "Context: "
            error_data['context'] = f"{context_prefix}{context}"

        return error_data


def get_enhanced_error(
    error: Exception,
    context: Optional[str] = None,
    language: str = "zh"
) -> str:
    """
    GetFormat增强Error消息

    Args:
        error: 异常对象
        context: onunder文信息
        language: 语言偏好 ("zh" or "en")

    Returns:
        FormatError消息字符串
    """
    error_data = ErrorMessages.detect_and_enhance_error(error, context, language)
    return ErrorMessages.format_for_display(error_data, language)


def get_error_info(
    error: Exception,
    context: Optional[str] = None,
    language: str = "zh"
) -> Dict[str, Any]:
    """
    Get结构化Error message

    Args:
        error: 异常对象
        context: onunder文信息
        language: 语言偏好

    Returns:
        结构化Error message字典
    """
    return ErrorMessages.detect_and_enhance_error(error, context, language)
