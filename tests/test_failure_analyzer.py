"""
FailureAnalyzer 模块单元Test

Test失败案例分析系统核心功能：
- 失败类别分类（CalculateError、概念Error、幻觉etc.）
- 单失败案例分析
- 批量失败分析报告
- SuggestionGenerate
"""

import pytest

from core.failure_analyzer import (
    FailureAnalyzer,
    FailureCategory,
    FailureCase,
    FailureAnalysisReport,
    analyze_failures
)


class TestFailureCategory:
    """Test FailureCategory 枚举"""

    def test_all_categories_defined(self):
        """Test所has失败类别已定义"""
        categories = [
            FailureCategory.CALCULATION_ERROR,
            FailureCategory.CONCEPT_MISUNDERSTANDING,
            FailureCategory.REASONING_GAP,
            FailureCategory.HALLUCINATION,
            FailureCategory.FORMAT_MISMATCH,
            FailureCategory.KNOWLEDGE_GAP,
            FailureCategory.ATTENTION_ERROR,
            FailureCategory.MAGNITUDE_ERROR,
            FailureCategory.UNIT_CONVERSION,
            FailureCategory.LOGIC_ERROR,
            FailureCategory.NO_RESPONSE,
            FailureCategory.TIMEOUT,
            FailureCategory.API_ERROR,
            FailureCategory.UNKNOWN,
        ]
        assert len(categories) == 14

    def test_category_values(self):
        """Test类别值正确"""
        assert FailureCategory.CALCULATION_ERROR.value == "calculation_error"
        assert FailureCategory.HALLUCINATION.value == "hallucination"
        assert FailureCategory.NO_RESPONSE.value == "no_response"


class TestFailureCase:
    """Test FailureCase Data类"""

    def test_create_failure_case_minimal(self):
        """TestCreate最小 FailureCase"""
        case = FailureCase(
            sample_id="001",
            question="What is 2+2?",
            correct_answer="4",
            predicted_answer="5",
            model_response="2+2=5"
        )
        assert case.sample_id == "001"
        assert case.category == FailureCategory.UNKNOWN
        assert case.confidence == 0.0

    def test_create_failure_case_full(self):
        """TestCreate完整 FailureCase"""
        case = FailureCase(
            sample_id="002",
            question="Calculate 15 * 3",
            correct_answer="45",
            predicted_answer="30",
            model_response="15 * 3 = 30",
            reasoning_content="Step 1: 15 * 3 = 30",
            category=FailureCategory.CALCULATION_ERROR,
            confidence=0.9,
            analysis="CalculateError",
            root_cause="乘法运算Error",
            suggestions=["CheckCalculate", "useCalculate器"],
            question_type="math",
            difficulty="easy"
        )
        assert case.category == FailureCategory.CALCULATION_ERROR
        assert case.confidence == 0.9
        assert len(case.suggestions) == 2


class TestFailureAnalyzer:
    """Test FailureAnalyzer 类"""

    @pytest.fixture
    def analyzer(self):
        """Create分析器实例"""
        return FailureAnalyzer()

    # ==================== 系统ErrorTest ====================

    def test_analyze_timeout_error(self, analyzer):
        """Test超时Error分析"""
        case = analyzer.analyze_single(
            sample_id="001",
            question="What is 2+2?",
            correct_answer="4",
            predicted_answer="",
            model_response="",
            error="Request timeout after 30s"
        )
        assert case.category == FailureCategory.TIMEOUT
        assert case.confidence == 1.0
        assert "增加超时时间" in case.suggestions

    def test_analyze_api_error(self, analyzer):
        """Test API Error分析"""
        case = analyzer.analyze_single(
            sample_id="002",
            question="Test question",
            correct_answer="42",
            predicted_answer="",
            model_response="",
            error="HTTP 429: Rate limit exceeded"
        )
        assert case.category == FailureCategory.API_ERROR
        assert "重试" in case.suggestions[0] or "重试" in "".join(case.suggestions)

    # ==================== no响应Test ====================

    def test_analyze_no_response(self, analyzer):
        """Testno响应分析"""
        case = analyzer.analyze_single(
            sample_id="003",
            question="Test",
            correct_answer="A",
            predicted_answer="",
            model_response="",
            reasoning_content=""
        )
        assert case.category == FailureCategory.NO_RESPONSE
        assert case.confidence == 1.0

    def test_analyze_whitespace_only_response(self, analyzer):
        """Test纯空白响应"""
        case = analyzer.analyze_single(
            sample_id="004",
            question="Test",
            correct_answer="B",
            predicted_answer="",
            model_response="   \n\t  ",
            reasoning_content=""
        )
        assert case.category == FailureCategory.NO_RESPONSE

    # ==================== 数量级ErrorTest ====================

    def test_analyze_magnitude_error_10x(self, analyzer):
        """Test 10 倍数量级Error"""
        case = analyzer.analyze_single(
            sample_id="005",
            question="Calculate 100 / 10",
            correct_answer="10",
            predicted_answer="100",
            model_response="100 / 10 = 100"
        )
        assert case.category == FailureCategory.MAGNITUDE_ERROR
        assert case.confidence == 0.9
        assert "数量级" in case.analysis

    def test_analyze_magnitude_error_01x(self, analyzer):
        """Test 0.1 倍数量级Error"""
        case = analyzer.analyze_single(
            sample_id="006",
            question="What is 1000?",
            correct_answer="1000",
            predicted_answer="100",
            model_response="The answer is 100"
        )
        assert case.category == FailureCategory.MAGNITUDE_ERROR

    # ==================== CalculateErrorTest ====================

    def test_analyze_calculation_error(self, analyzer):
        """TestCalculateError"""
        case = analyzer.analyze_single(
            sample_id="007",
            question="What is 15 + 27?",
            correct_answer="42",
            predicted_answer="41",
            model_response="15 + 27 = 41"
        )
        # CalculateError模式匹配，置信度0.85
        assert case.category == FailureCategory.CALCULATION_ERROR
        assert case.confidence >= 0.8

    def test_analyze_near_correct_value(self, analyzer):
        """Test接近正确值（舍入误差）"""
        case = analyzer.analyze_single(
            sample_id="008",
            question="Calculate 10 / 3",
            correct_answer="3.333",
            predicted_answer="3.33",
            model_response="10 / 3 ≈ 3.33"
        )
        assert case.category == FailureCategory.CALCULATION_ERROR
        assert "接近" in case.analysis

    # ==================== 格式not匹配Test ====================

    def test_analyze_format_mismatch(self, analyzer):
        """Test格式not匹配but内容etc.价"""
        case = analyzer.analyze_single(
            sample_id="009",
            question="Choose the correct option",
            correct_answer="A",
            predicted_answer="(A)",
            model_response="The answer is (A)"
        )
        assert case.category == FailureCategory.FORMAT_MISMATCH
        assert case.confidence == 0.95
        assert "格式" in case.analysis

    def test_analyze_format_mismatch_with_spacing(self, analyzer):
        """Test带空格格式not匹配"""
        case = analyzer.analyze_single(
            sample_id="010",
            question="What is 42?",
            correct_answer="42",
            predicted_answer="4 2",
            model_response="Answer: 4 2"
        )
        assert case.category == FailureCategory.FORMAT_MISMATCH

    # ==================== 知识缺失Test ====================

    def test_analyze_knowledge_gap_english(self, analyzer):
        """Test英文知识缺失"""
        case = analyzer.analyze_single(
            sample_id="011",
            question="What is the capital of XYZ?",
            correct_answer="Unknown",
            predicted_answer="I don't know",
            model_response="I'm not sure about this, I don't know the answer"
        )
        assert case.category == FailureCategory.KNOWLEDGE_GAP
        assert case.confidence == 0.8

    def test_analyze_knowledge_gap_chinese(self, analyzer):
        """Testin文知识缺失"""
        case = analyzer.analyze_single(
            sample_id="012",
            question="某冷知识问题",
            correct_answer="Answer",
            predicted_answer="not确定",
            model_response="我not知道这问题Answer"
        )
        assert case.category == FailureCategory.KNOWLEDGE_GAP

    # ==================== 注意力ErrorTest ====================

    def test_analyze_attention_error(self, analyzer):
        """Test注意力Error（遗漏QuestionData）"""
        case = analyzer.analyze_single(
            sample_id="013",
            question="If x=10, y=20, z=30, what is x+y+z?",
            correct_answer="60",
            predicted_answer="30",
            model_response="x + y = 30"
        )
        assert case.category == FailureCategory.ATTENTION_ERROR
        assert "遗漏" in case.analysis

    # ==================== 推理跳步Test ====================

    def test_analyze_reasoning_gap(self, analyzer):
        """Test推理跳步"""
        case = analyzer.analyze_single(
            sample_id="014",
            question="A long complex math problem that requires multiple steps of reasoning and calculation to solve the question about some abstract concept...",
            correct_answer="100",
            predicted_answer="50",
            model_response="The answer is 50",
            reasoning_content="Answer: 50"
        )
        # Reasoning process过于简略，step_count <= 1，且Answernot接近正确值
        assert case.category == FailureCategory.REASONING_GAP
        assert case.confidence >= 0.6

    # ==================== 幻觉Test ====================

    def test_analyze_hallucination(self, analyzer):
        """Test幻觉（引入额外信息）"""
        case = analyzer.analyze_single(
            sample_id="015",
            question="What is 5 plus 3?",
            correct_answer="8",
            predicted_answer="20",
            model_response="Let's assume x=10 and y=5. Then we add 1, 2, 3, 4, 5.",
            reasoning_content="Suppose we also include 15, 25, 35 in our calculation"
        )
        # 幻觉检测：Questionin只has 5, 3，but响应引入 10, 1, 2, 3, 4, 5, 15, 25, 35
        assert case.category == FailureCategory.HALLUCINATION
        assert case.confidence >= 0.5

    # ==================== default分类Test ====================

    def test_analyze_unknown_error(self, analyzer):
        """Test未知Errordefault分类"""
        case = analyzer.analyze_single(
            sample_id="016",
            question="Some question",
            correct_answer="Answer",
            predicted_answer="Wrong",
            model_response="Some response that doesn't match any pattern"
        )
        assert case.category == FailureCategory.CONCEPT_MISUNDERSTANDING
        assert case.confidence == 0.5

    # ==================== 问题类型检测Test ====================

    def test_detect_math_question(self, analyzer):
        """Test数学问题类型检测"""
        qtype = analyzer._detect_question_type("Calculate the value of x")
        assert qtype == "math"

    def test_detect_choice_question(self, analyzer):
        """Test选择题类型检测"""
        qtype = analyzer._detect_question_type("Which option is correct?")
        assert qtype == "choice"

    def test_detect_reasoning_question(self, analyzer):
        """Test推理问题类型检测"""
        qtype = analyzer._detect_question_type("Why is the sky blue?")
        assert qtype == "reasoning"

    def test_detect_boolean_question(self, analyzer):
        """Test布尔问题类型检测"""
        qtype = analyzer._detect_question_type("Is this statement true?")
        assert qtype == "boolean"

    def test_detect_general_question(self, analyzer):
        """Test通用问题类型检测"""
        qtype = analyzer._detect_question_type("What do you think?")
        assert qtype == "general"

    # ==================== CalculateValidateTest ====================

    def test_is_calculation_wrong_addition(self, analyzer):
        """TestError加法"""
        assert analyzer._is_calculation_wrong("2 + 2 = 5") is True

    def test_is_calculation_correct_addition(self, analyzer):
        """Test正确加法"""
        assert analyzer._is_calculation_wrong("2 + 2 = 4") is False

    def test_is_calculation_wrong_subtraction(self, analyzer):
        """TestError减法"""
        assert analyzer._is_calculation_wrong("10 - 3 = 6") is True

    def test_is_calculation_wrong_multiplication(self, analyzer):
        """TestError乘法"""
        assert analyzer._is_calculation_wrong("5 * 6 = 35") is True

    def test_is_calculation_wrong_division(self, analyzer):
        """TestError除法"""
        assert analyzer._is_calculation_wrong("20 / 4 = 3") is True

    def test_is_calculation_no_match(self, analyzer):
        """Testnot匹配表达式"""
        assert analyzer._is_calculation_wrong("not a calculation") is False

    # ==================== SuggestionGenerateTest ====================

    def test_suggestions_for_calculation_error(self, analyzer):
        """TestCalculateErrorSuggestion"""
        suggestions = analyzer._generate_suggestions(
            FailureCategory.CALCULATION_ERROR,
            "CalculateError"
        )
        assert len(suggestions) > 0
        assert any("few-shot" in s.lower() or "示例" in s for s in suggestions)

    def test_suggestions_for_hallucination(self, analyzer):
        """Test幻觉Suggestion"""
        suggestions = analyzer._generate_suggestions(
            FailureCategory.HALLUCINATION,
            "引入额外信息"
        )
        assert len(suggestions) > 0

    def test_suggestions_for_unknown_category(self, analyzer):
        """Test未知类别defaultSuggestion"""
        suggestions = analyzer._generate_suggestions(
            FailureCategory.UNKNOWN,
            "未知Error"
        )
        assert len(suggestions) == 1
        assert "人工分析" in suggestions[0]

    # ==================== 批量分析Test ====================

    def test_analyze_batch_empty(self, analyzer):
        """Test空批量分析"""
        report = analyzer.analyze_batch([])
        assert report.total_samples == 0
        assert report.failed_samples == 0
        assert report.failure_rate == 0

    def test_analyze_batch_single(self, analyzer):
        """Test单 samples批量分析"""
        samples = [{
            'sample_id': '001',
            'question': 'What is 2+2?',
            'correct_answer': '4',
            'predicted_answer': '5',
            'model_response': '2+2=5'
        }]
        report = analyzer.analyze_batch(samples, total_samples=10)
        assert report.total_samples == 10
        assert report.failed_samples == 1
        assert report.failure_rate == 10.0
        assert len(report.cases) == 1

    def test_analyze_batch_multiple(self, analyzer):
        """Test多样本批量分析"""
        samples = [
            {
                'sample_id': '001',
                'question': '2+2=?',
                'correct_answer': '4',
                'predicted_answer': '5',
                'model_response': '2+2=5'
            },
            {
                'sample_id': '002',
                'question': '10*10=?',
                'correct_answer': '100',
                'predicted_answer': '10',
                'model_response': '10*10=10'
            },
            {
                'sample_id': '003',
                'question': 'Timeout test',
                'correct_answer': '42',
                'predicted_answer': '',
                'model_response': '',
                'error': 'timeout'
            }
        ]
        report = analyzer.analyze_batch(samples)
        assert report.failed_samples == 3
        assert len(report.category_distribution) > 0
        assert len(report.cases) == 3

    def test_analyze_batch_category_distribution(self, analyzer):
        """Test批量分析类别分布"""
        samples = [
            {
                'sample_id': f'{i:03d}',
                'question': 'Question',
                'correct_answer': '4',
                'predicted_answer': '5',
                'model_response': '2 + 2 = 5'
            }
            for i in range(5)
        ]
        report = analyzer.analyze_batch(samples)
        # 实际Calculate表达式匹配到 calculation_error
        assert 'calculation_error' in report.category_distribution
        assert report.category_distribution['calculation_error'] == 5
        assert report.category_percentage['calculation_error'] == 100.0

    def test_analyze_batch_top_issues(self, analyzer):
        """Test批量分析顶级问题"""
        samples = [
            {
                'sample_id': f'{i:03d}',
                'question': 'Question',
                'correct_answer': '4',
                'predicted_answer': '5',
                'model_response': '2+2=5'
            }
            for i in range(10)
        ]
        report = analyzer.analyze_batch(samples)
        assert len(report.top_issues) <= 3
        assert any('calculation' in issue.lower() for issue in report.top_issues)

    def test_analyze_batch_improvement_suggestions(self, analyzer):
        """Test批量分析改进Suggestion"""
        samples = [
            {
                'sample_id': '001',
                'question': 'Question',
                'correct_answer': '4',
                'predicted_answer': '5',
                'model_response': '2+2=5'
            }
        ]
        report = analyzer.analyze_batch(samples)
        assert len(report.improvement_suggestions) > 0
        assert len(report.improvement_suggestions) <= 5


class TestFailureAnalysisReport:
    """Test FailureAnalysisReport Data类"""

    def test_create_report_minimal(self):
        """TestCreate最小报告"""
        report = FailureAnalysisReport(
            total_samples=100,
            failed_samples=10,
            failure_rate=10.0
        )
        assert report.total_samples == 100
        assert report.failed_samples == 10
        assert report.failure_rate == 10.0
        assert report.category_distribution == {}
        assert report.cases == []

    def test_create_report_full(self):
        """TestCreate完整报告"""
        report = FailureAnalysisReport(
            total_samples=100,
            failed_samples=10,
            failure_rate=10.0,
            category_distribution={'calculation_error': 5, 'format_mismatch': 5},
            category_percentage={'calculation_error': 50.0, 'format_mismatch': 50.0},
            cases=[],
            top_issues=['calculation_error: 5 例 (50.0%)'],
            improvement_suggestions=['Suggestion1', 'Suggestion2']
        )
        assert len(report.category_distribution) == 2
        assert len(report.improvement_suggestions) == 2


class TestConvenienceFunctions:
    """Test便捷函数"""

    def test_analyze_failures_function(self):
        """Test analyze_failures 便捷函数"""
        samples = [{
            'sample_id': '001',
            'question': '2+2=?',
            'correct_answer': '4',
            'predicted_answer': '5',
            'model_response': '2+2=5'
        }]
        report = analyze_failures(samples, total=50)
        assert report.total_samples == 50
        assert report.failed_samples == 1
        assert report.failure_rate == 2.0
