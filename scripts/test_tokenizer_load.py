from transformers import AutoTokenizer
import os

tokenizer_path = "./tokenizers/MiniMax-M2"

print(f"Testing tokenizer load from: {tokenizer_path}")
print(f"Path exists: {os.path.exists(tokenizer_path)}")

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    print("Successfully loaded tokenizer!")
    print(f"Tokenizer class: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    text = "Hello world, this is a test."
    tokens = tokenizer.encode(text)
    print(f"Test encode '{text}': {tokens} (len={len(tokens)})")
    
except Exception as e:
    print(f"FAILED to load tokenizer: {e}")
    import traceback
    traceback.print_exc()
