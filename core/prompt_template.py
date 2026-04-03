"""
Prompt 模板系统 (Prompt Template System)

提供统一、可Configure Prompt 模板Render功能。

功能：
1. Jinja2 模板Render
2. 支持 Completion and Chat 两种格式
3. Few-shot 示例注入
4. 系统消息Configure
5. 从 YAML ConfigureLoad模板

借鉴 lm-evaluation-harness  doc_to_text 设计。

use方式：
    from core.prompt_template import PromptTemplate, ChatTemplate

    # 方式 1: 直接定义模板
    template = PromptTemplate(
        doc_to_text="Question: {{question}}\\nAnswer:",
        doc_to_target="{{answer}}"
    )
    prompt = template.render(sample)

    # 方式 2: Chat 格式
    chat_template = ChatTemplate(
        system_message="You are a helpful assistant.",
        user_template="{{question}}",
        assistant_template="{{answer}}"
    )
    messages = chat_template.render_messages(sample, few_shot_examples)

    # 方式 3: 从 YAML Load
    template = PromptTemplate.from_yaml("task_configs/mmlu.yaml")
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import yaml

# Jinja2 Import
try:
    from jinja2 import BaseLoader, Environment, StrictUndefined, TemplateSyntaxError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None


# ============================================
# 模板格式枚举
# ============================================

class PromptFormat:
    """Prompt 格式类型"""
    COMPLETION = "completion"  # 传统补全格式 (GPT-3 style)
    CHAT = "chat"              # Chat 格式 (ChatGPT style)


# ============================================
# Jinja2 环境Configure
# ============================================

def create_jinja_env() -> 'Environment':
    """CreateConfigure好 Jinja2 环境"""
    if not JINJA2_AVAILABLE:
        raise ImportError("jinja2 is required for prompt templates. Install with: pip install jinja2")

    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True
    )

    # AddCustomFilter器
    env.filters['choice_letter'] = lambda idx: chr(ord('A') + int(idx))
    env.filters['strip'] = lambda s: s.strip() if isinstance(s, str) else s
    env.filters['escape_newlines'] = lambda s: s.replace('\n', '\\n') if isinstance(s, str) else s

    return env


def render_template(template_str: str, context: dict[str, Any]) -> str:
    """
    Render Jinja2 模板

    Args:
        template_str: 模板字符串
        context: onunder文变量字典

    Returns:
        Render后字符串
    """
    if not template_str:
        return ""

    # 简单变量替换 (作is fallback)
    if not JINJA2_AVAILABLE:
        result = template_str
        for key, value in context.items():
            result = result.replace("{{" + key + "}}", str(value))
        return result

    try:
        env = create_jinja_env()
        template = env.from_string(template_str)
        return template.render(**context)
    except Exception:
        # 回退到简单替换
        result = template_str
        for key, value in context.items():
            result = result.replace("{{" + key + "}}", str(value))
        return result


# ============================================
# Prompt 模板类
# ============================================

@dataclass
class PromptTemplate:
    """
    Prompt 模板 (Completion 格式)

    用于传统补全式 Prompt Build。

    Attributes:
        doc_to_text: will样本Convertis prompt 模板
        doc_to_target: will样本Convertis目标Answer模板
        description: 任务描述 (optional)
        fewshot_delimiter: Few-shot 示例之间分隔符
    """
    doc_to_text: str
    doc_to_target: str = ""
    description: str = ""
    fewshot_delimiter: str = "\n\n"
    answer_delimiter: str = " "  # doc_to_text and目标Answer之间分隔符

    # optionalPreprocess/Postprocess函数
    pre_process: Callable[[dict], dict] | None = None
    post_process: Callable[[str], str] | None = None

    def render(
        self,
        sample: dict[str, Any],
        include_target: bool = False
    ) -> str:
        """
        Render单 samples

        Args:
            sample: Sample count据
            include_target: is否包含目标Answer

        Returns:
            Render后 prompt
        """
        # Preprocess
        if self.pre_process:
            sample = self.pre_process(sample)

        # Render doc_to_text
        prompt = render_template(self.doc_to_text, sample)

        # optionalAdd目标Answer
        if include_target and self.doc_to_target:
            target = render_template(self.doc_to_target, sample)
            prompt = prompt + self.answer_delimiter + target

        # Postprocess
        if self.post_process:
            prompt = self.post_process(prompt)

        return prompt

    def render_full(
        self,
        sample: dict[str, Any],
        few_shot_examples: list[dict[str, Any]] | None = None,
        include_description: bool = True
    ) -> str:
        """
        Render完整 Prompt (包含描述and few-shot)

        Args:
            sample: 待评估样本
            few_shot_examples: Few-shot 示例列表
            include_description: is否包含任务描述

        Returns:
            完整 prompt
        """
        parts = []

        # 任务描述
        if include_description and self.description:
            desc = render_template(self.description, sample)
            parts.append(desc)

        # Few-shot 示例 (包含Answer)
        if few_shot_examples:
            for example in few_shot_examples:
                example_prompt = self.render(example, include_target=True)
                parts.append(example_prompt)

        # 待评估样本 (not包含Answer)
        sample_prompt = self.render(sample, include_target=False)
        parts.append(sample_prompt)

        return self.fewshot_delimiter.join(parts)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PromptTemplate':
        """从 YAML Configure文件Load模板"""
        with open(yaml_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return cls(
            doc_to_text=config.get('doc_to_text', '{{question}}'),
            doc_to_target=config.get('doc_to_target', '{{answer}}'),
            description=config.get('description', ''),
            fewshot_delimiter=config.get('fewshot_delimiter', '\n\n')
        )


# ============================================
# Chat 模板类
# ============================================

@dataclass
class ChatMessage:
    """单条聊天消息"""
    role: str  # system, user, assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatTemplate:
    """
    Chat 格式模板

    用于 ChatGPT 风格对话式 Prompt Build。

    Attributes:
        system_message: 系统消息模板
        user_template: 用户消息模板 (问题)
        assistant_template: 助手消息模板 (Answer)
        fewshot_as_multiturn: Few-shot is否作is多轮对话
    """
    system_message: str = ""
    user_template: str = "{{question}}"
    assistant_template: str = "{{answer}}"
    fewshot_as_multiturn: bool = True

    def render_user(self, sample: dict[str, Any]) -> str:
        """Render用户消息"""
        return render_template(self.user_template, sample)

    def render_assistant(self, sample: dict[str, Any]) -> str:
        """Render助手消息"""
        return render_template(self.assistant_template, sample)

    def render_system(self, sample: dict[str, Any] | None = None) -> str:
        """Render系统消息"""
        if not self.system_message:
            return ""
        return render_template(self.system_message, sample or {})

    def render_messages(
        self,
        sample: dict[str, Any],
        few_shot_examples: list[dict[str, Any]] | None = None
    ) -> list[dict[str, str]]:
        """
        Renderis消息列表

        Args:
            sample: 待评估样本
            few_shot_examples: Few-shot 示例

        Returns:
            消息列表 [{"role": "...", "content": "..."}]
        """
        messages = []

        # 系统消息
        if self.system_message:
            system_content = self.render_system(sample)
            messages.append({"role": "system", "content": system_content})

        # Few-shot 示例
        if few_shot_examples and self.fewshot_as_multiturn:
            for example in few_shot_examples:
                # 用户问题
                user_content = self.render_user(example)
                messages.append({"role": "user", "content": user_content})

                # 助手回答
                assistant_content = self.render_assistant(example)
                messages.append({"role": "assistant", "content": assistant_content})

        # 待评估样本 (只has用户消息)
        user_content = self.render_user(sample)
        messages.append({"role": "user", "content": user_content})

        return messages

    def render_as_completion(
        self,
        sample: dict[str, Any],
        few_shot_examples: list[dict[str, Any]] | None = None,
        delimiter: str = "\n\n"
    ) -> str:
        """
        will Chat 格式Convertis Completion 格式

        用于Not supported Chat API Model。
        """
        messages = self.render_messages(sample, few_shot_examples)

        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        # 最后一条is用户消息，needAdd Assistant: Tip
        parts.append("Assistant:")

        return delimiter.join(parts)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ChatTemplate':
        """从 YAML Configure文件Load模板"""
        with open(yaml_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return cls(
            system_message=config.get('system_message', ''),
            user_template=config.get('user_template', config.get('doc_to_text', '{{question}}')),
            assistant_template=config.get('assistant_template', config.get('doc_to_target', '{{answer}}')),
            fewshot_as_multiturn=config.get('fewshot_as_multiturn', True)
        )


# ============================================
# 预定义模板库
# ============================================

class TemplateLibrary:
    """预定义常用模板"""

    @staticmethod
    def mmlu() -> PromptTemplate:
        """MMLU 多选题模板"""
        return PromptTemplate(
            description="The following are multiple choice questions (with answers) about {{subject|default('general knowledge')}}.\n\n",
            doc_to_text="""Question: {{question}}
A. {{choices[0]}}
B. {{choices[1]}}
C. {{choices[2]}}
D. {{choices[3]}}
Answer:""",
            doc_to_target=" {{answer|choice_letter if answer is number else answer}}"
        )

    @staticmethod
    def gsm8k() -> PromptTemplate:
        """GSM8K 数学推理模板"""
        return PromptTemplate(
            description="Solve the following grade school math problems step by step. Show your work and put your final answer after ####.\n\n",
            doc_to_text="Question: {{question}}\nSolution:",
            doc_to_target=" {{answer}}"
        )

    @staticmethod
    def humaneval() -> PromptTemplate:
        """HumanEval 代码Generate模板"""
        return PromptTemplate(
            description="",
            doc_to_text="{{prompt}}",
            doc_to_target="{{canonical_solution}}"
        )

    @staticmethod
    def truthfulqa() -> PromptTemplate:
        """TruthfulQA 真实性问答模板"""
        return PromptTemplate(
            description="Answer the following questions truthfully.\n\n",
            doc_to_text="Q: {{question}}\nA:",
            doc_to_target=" {{best_answer}}"
        )

    @staticmethod
    def arc() -> PromptTemplate:
        """ARC 科学推理模板"""
        return PromptTemplate(
            description="",
            doc_to_text="""Question: {{question}}
{% for label, text in zip(choices.label, choices.text) %}{{label}}. {{text}}
{% endfor %}Answer:""",
            doc_to_target=" {{answerKey}}"
        )

    @staticmethod
    def chat_qa() -> ChatTemplate:
        """通用问答 Chat 模板"""
        return ChatTemplate(
            system_message="You are a helpful, accurate, and concise assistant. Answer the user's questions directly.",
            user_template="{{question}}",
            assistant_template="{{answer}}"
        )

    @staticmethod
    def chat_cot() -> ChatTemplate:
        """Chain-of-Thought Chat 模板"""
        return ChatTemplate(
            system_message="You are a helpful assistant that thinks step by step before providing the final answer.",
            user_template="{{question}}\n\nLet's think step by step.",
            assistant_template="{{reasoning}}\n\nThe answer is: {{answer}}"
        )

    @staticmethod
    def chat_code() -> ChatTemplate:
        """代码Generate Chat 模板"""
        return ChatTemplate(
            system_message="You are an expert programmer. Write clean, efficient, and well-documented code.",
            user_template="{{prompt}}",
            assistant_template="{{solution}}"
        )


# ============================================
# 模板Factory
# ============================================

class TemplateFactory:
    """模板Factory - 统一模板Get接口"""

    # 预定义模板映射
    TEMPLATES = {
        "mmlu": TemplateLibrary.mmlu,
        "gsm8k": TemplateLibrary.gsm8k,
        "humaneval": TemplateLibrary.humaneval,
        "truthfulqa": TemplateLibrary.truthfulqa,
        "arc": TemplateLibrary.arc,
    }

    CHAT_TEMPLATES = {
        "chat_qa": TemplateLibrary.chat_qa,
        "chat_cot": TemplateLibrary.chat_cot,
        "chat_code": TemplateLibrary.chat_code,
    }

    @classmethod
    def get(
        cls,
        name: str,
        format: str = PromptFormat.COMPLETION
    ) -> Union[PromptTemplate, ChatTemplate]:
        """
        Get模板

        Args:
            name: 模板名称
            format: 格式类型 (completion/chat)

        Returns:
            模板实例
        """
        if format == PromptFormat.CHAT:
            if name in cls.CHAT_TEMPLATES:
                return cls.CHAT_TEMPLATES[name]()
            # 尝试Convertis Chat 格式
            if name in cls.TEMPLATES:
                completion_template = cls.TEMPLATES[name]()
                return ChatTemplate(
                    user_template=completion_template.doc_to_text,
                    assistant_template=completion_template.doc_to_target
                )
        else:
            if name in cls.TEMPLATES:
                return cls.TEMPLATES[name]()

        raise ValueError(f"Unknown template: {name}")

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        format: str = PromptFormat.COMPLETION
    ) -> Union[PromptTemplate, ChatTemplate]:
        """从 YAML 文件Load模板"""
        if format == PromptFormat.CHAT:
            return ChatTemplate.from_yaml(yaml_path)
        return PromptTemplate.from_yaml(yaml_path)

    @classmethod
    def list_templates(cls) -> dict[str, list[str]]:
        """列出所has可用模板"""
        return {
            "completion": list(cls.TEMPLATES.keys()),
            "chat": list(cls.CHAT_TEMPLATES.keys())
        }


# ============================================
# 便捷函数
# ============================================

def get_template(name: str, format: str = "completion") -> Union[PromptTemplate, ChatTemplate]:
    """Get模板便捷函数"""
    return TemplateFactory.get(name, format)


def render_prompt(
    template_str: str,
    sample: dict[str, Any],
    few_shot_examples: list[dict[str, Any]] | None = None
) -> str:
    """
    快速Render prompt

    Args:
        template_str: Jinja2 模板字符串
        sample: Sample count据
        few_shot_examples: Few-shot 示例

    Returns:
        Render后 prompt
    """
    template = PromptTemplate(doc_to_text=template_str)
    return template.render_full(sample, few_shot_examples)
