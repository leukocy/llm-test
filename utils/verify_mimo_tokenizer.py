import os

from config.settings import HF_MODEL_MAPPING
from core.tokenizer_utils import get_cached_tokenizer


def test_mimo_loading():
    model_id = "XiaomiMiMo/MiMo-V2-Flash"
    model_id_lower = model_id.lower()

    inferred_path = None
    for key, path in HF_MODEL_MAPPING.items():
        if key.lower() in model_id_lower:
            inferred_path = path
            break

    print(f"Model ID: {model_id}")
    print(f"Inferred Path: {inferred_path}")

    if inferred_path and os.path.exists(inferred_path):
        print(f"Path exists: {inferred_path}")
        try:
            tokenizer = get_cached_tokenizer(inferred_path)
            if tokenizer:
                print("✅ Successfully loaded MiMo tokenizer!")
                test_text = "Hello, MiMo-V2-Flash!"
                tokens = tokenizer.encode(test_text, add_special_tokens=False)
                print(f"Encoded '{test_text}' into {len(tokens)} tokens.")
            else:
                print("❌ Failed to load tokenizer (returned None)")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
    else:
        print(f"❌ Inferred path does not exist or not found: {inferred_path}")

if __name__ == "__main__":
    test_mimo_loading()
