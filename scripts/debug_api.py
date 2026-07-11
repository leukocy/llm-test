import os

import requests

base_url = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
model = "DeepSeek-V3.1"

print(f"Checking base URL: {base_url}")

# 1. Check /models endpoint
try:
    resp = requests.get(f"{base_url}/models", timeout=5)
    print(f"GET /models status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Models: {resp.json()}")
    else:
        print(f"Error text: {resp.text[:200]}")
except Exception as e:
    print(f"GET /models failed: {e}")

# 2. Check /chat/completions connectivity (404/405/200?)
try:
    # Just a HEAD or dummy request
    resp = requests.post(f"{base_url}/chat/completions", json={}, timeout=5)
    print(f"POST /chat/completions (empty json) status: {resp.status_code}")
except Exception as e:
    print(f"POST /chat/completions failed: {e}")
