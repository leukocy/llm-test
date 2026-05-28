
import asyncio
import time
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.providers.openai import OpenAIProvider


async def test_provider(name, api_base_url, api_key, model_id=None):
    print(f"\n{'='*20} Testing Provider: {name} {'='*20}")
    print(f"Base URL: {api_base_url}")
    
    # Try to fetch available models if model_id is not provided
    if not model_id:
        import aiohttp
        print(f"Fetching available models...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            model_id = models[0]["id"]
                            print(f"Using first available model: {model_id}")
                        else:
                            print("No models found, using default DeepSeek-V3.1")
                            model_id = "DeepSeek-V3.1"
                    else:
                        print(f"Failed to fetch models: {resp.status}")
                        model_id = "DeepSeek-V3.1"
        except Exception as e:
            print(f"Error fetching models: {e}")
            model_id = "DeepSeek-V3.1"
    else:
        print(f"Using specified model: {model_id}")

    provider = OpenAIProvider(api_base_url, api_key, model_id)
    
    print(f"Provider initialized.")
    
    client = None
    session_id = 12345 + hash(name) % 1000
    prompt = "Write a very short poem about coding."
    max_tokens = 50
    
    start_time = 0
    end_time = 0

    try:
        print("Sending request...")
        result = await provider.get_completion(client, session_id, prompt, max_tokens)
        
        if result.get("error"):
            print(f"❌ Request failed: {result.get('error')}")
            if result.get("error_info"):
                 import json
                 print(f"Error Info:\n{json.dumps(result['error_info'], ensure_ascii=False, indent=2)}")
            return

        print("✅ Request successful!")
        
        # Verify timestamps
        created_at = result.get("created_at")
        start_time = result.get("start_time")
        first_token_time = result.get("first_token_time")
        end_time = result.get("end_time")
        
        # Validation
        print("Validations:")
        
        now = time.time()
        if created_at and abs(now - created_at) < 60:
            print(f"✅ created_at valid (delta={now - created_at:.2f}s)")
        else:
            print(f"❌ created_at invalid! (delta={now - created_at:.2f}s, val={created_at})")
            
        if start_time < end_time:
             print(f"✅ Monotonicity: start < end")
        else:
             print(f"❌ Monotonicity violation!")
             
        if first_token_time:
            ttft = first_token_time - start_time
            if start_time <= first_token_time <= end_time:
                print(f"✅ TTFT valid: {ttft*1000:.2f}ms")
            else:
                print(f"❌ TTFT out of range!")
        else:
            print("❌ No TTFT recorded!")

        duration = end_time - start_time
        print(f"Duration: {duration:.4f}s")
        
        # Granularity
        token_timestamps = result.get("token_timestamps", [])
        completion_tokens = result['usage_info'].get('completion_tokens', len(token_timestamps))
        
        print(f"Tokens: {completion_tokens}, Chunks: {len(token_timestamps)}")
        
        if len(token_timestamps) > 1:
            tokens_per_chunk = completion_tokens / len(token_timestamps)
            print(f"Granularity: {tokens_per_chunk:.2f} tokens/chunk")
            if tokens_per_chunk <= 1.5:
                print("✅ Good granularity (Streaming works)")
            else:
                print("⚠️  Poor granularity (Buffered/Packet aggregation)")
        else:
            print("⚠️  Not enough chunks for granularity analysis")

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("Starting Multi-Provider Verification...")
    
    providers = [
        # Local (configure via environment variables)
        ("Local", os.getenv("API_BASE_URL", "http://localhost:8000/v1"), os.getenv("API_KEY", ""), None),
        # Cloud (NVIDIA NIM)
        ("NVIDIA NIM", "https://integrate.api.nvidia.com/v1", os.getenv("NVIDIA_API_KEY", "nvapi-placeholder"), None)
    ]
    
    try:
        for name, url, key, model in providers:
            await test_provider(name, url, key, model)
    except Exception as e:
        print(f"\n❌ Exception occurred in main loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
