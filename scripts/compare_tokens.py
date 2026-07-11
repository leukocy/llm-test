import time

import requests
from transformers import AutoTokenizer

# Configure
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = "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"
MODEL = "qwen/qwen3-next-80b-a3b-instruct"
TOKENIZER_PATH = "./tokenizers/Qwen3-Next-80B-A3B-Instruct"
TARGET_INPUT_TOKENS = 1024
TARGET_OUTPUT_TOKENS = 256


def generate_calibrated_prompt(tokenizer, target_tokens):
    """Generate精准 prompt，使其 token 数etc.于 target_tokens"""
    base_word = "machine "
    base_tokens = tokenizer.encode(base_word, add_special_tokens=False)

    current_tokens = []
    text = ""

    while len(current_tokens) < target_tokens:
        diff = target_tokens - len(current_tokens)
        if diff >= len(base_tokens):
            text += base_word
        else:
            text += "." * diff
            break
        current_tokens = tokenizer.encode(text, add_special_tokens=False)

    return text, len(current_tokens)


print(f"Loading tokenizer from {TOKENIZER_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

print("-" * 50)
print(f"Generating prompt with ~{TARGET_INPUT_TOKENS} tokens...")
prompt_text, local_prompt_count = generate_calibrated_prompt(tokenizer, TARGET_INPUT_TOKENS)
print(f"Local Tokenizer Count (Input): {local_prompt_count}")

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt_text}],
    "max_tokens": TARGET_OUTPUT_TOKENS,
    "stream": False,
}

print(f"Sending request to API ({API_URL})...")
start_time = time.time()
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    result = response.json()
    end_time = time.time()
except Exception as e:
    print(f"API Request failed: {e}")
    if hasattr(e, "response") and e.response:
        print(f"Response Body: {e.response.text}")
    exit(1)

content = result["choices"][0]["message"]["content"]
usage = result["usage"]

local_output_count = len(tokenizer.encode(content, add_special_tokens=False))

# --- Report ---
output_str = ""
output_str += "-" * 50 + "\n"
output_str += "【NVIDIA API Comparison Results】\n"
output_str += f"Model: {MODEL}\n"
output_str += f"Time:  {end_time - start_time:.2f}s\n"
output_str += "-" * 50 + "\n\n"

output_str += (
    f"{'Type':<15} | {'Local Tokenizer':<15} | {'API Usage':<15} | {'Gap':<15} | {'Gap %':<10}\n"
)
output_str += "-" * 80 + "\n"

api_prompt = usage["prompt_tokens"]
diff_input = api_prompt - local_prompt_count
diff_input_pct = (diff_input / local_prompt_count) * 100
output_str += f"{'Input (Prefill)':<15} | {local_prompt_count:<15} | {api_prompt:<15} | {diff_input:<15} | {diff_input_pct:.1f}%\n"

api_completion = usage["completion_tokens"]
diff_output = api_completion - local_output_count
diff_output_pct = (diff_output / local_output_count) * 100 if local_output_count > 0 else 0
output_str += f"{'Output (Decode)':<15} | {local_output_count:<15} | {api_completion:<15} | {diff_output:<15} | {diff_output_pct:.1f}%\n"

output_str += "-" * 80 + "\n"
print(output_str)

with open("nvidia_comparison.txt", "w", encoding="utf-8") as f:
    f.write(output_str)
