export const templates = [
  { label: "OpenAI", provider: "OpenAI", url: "https://api.openai.com/v1" },
  {
    label: "Anthropic 原生",
    provider: "Anthropic",
    url: "https://api.anthropic.com/v1",
  },
  { label: "Together", provider: "OpenAI", url: "https://api.together.ai/v1" },
  {
    label: "火山引擎",
    provider: "OpenAI",
    url: "https://ark.cn-beijing.volces.com/api/v3",
  },
  { label: "DeepSeek", provider: "OpenAI", url: "https://api.deepseek.com/v1" },
  { label: "MiMo", provider: "OpenAI", url: "https://api.xiaomimimo.com/v1" },
  {
    label: "智谱",
    provider: "OpenAI",
    url: "https://open.bigmodel.cn/api/paas/v4",
  },
  {
    label: "硅基流动",
    provider: "OpenAI",
    url: "https://api.siliconflow.cn/v1",
  },
  { label: "Moonshot", provider: "OpenAI", url: "https://api.moonshot.cn/v1" },
  { label: "MiniMax", provider: "OpenAI", url: "https://api.minimax.chat/v1" },
  {
    label: "OpenRouter",
    provider: "OpenAI",
    url: "https://openrouter.ai/api/v1",
  },
  {
    label: "阿里云百炼",
    provider: "OpenAI",
    url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
  {
    label: "Gemini 原生",
    provider: "Gemini",
    url: "https://generativelanguage.googleapis.com",
  },
] as const;

export const originalModels = [
  "DeepSeek-V3.1",
  "DeepSeek-V3.2",
  "mimo-v2-flash",
  "Qwen3.5-397B-A17B-FP8",
  "Qwen3-Coder-480B-A35B",
  "Qwen3-Next-80B-A3B",
  "Kimi-K2.5",
  "Qwen3-235B-A22B",
  "MiniMax-M2.5",
  "MiniMax-M2.1",
  "GLM-5",
  "gpt-oss-120b",
  "deepseek-v3-1-terminus",
  "deepseek-v3-2-251201",
  "XiaomiMiMo/MiMo-V2-Flash",
  "DeepSeek-V4-Flash",
  "DeepSeek-V4-Pro",
];
