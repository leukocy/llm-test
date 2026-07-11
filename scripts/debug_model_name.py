import os

import requests

base_url = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
model_name = "DeepSeek-V3.1"

payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 10,
}

print(f"Testing model: {model_name}")
try:
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
