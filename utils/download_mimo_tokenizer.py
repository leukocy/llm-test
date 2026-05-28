import os

from huggingface_hub import hf_hub_download

repo_id = "XiaomiMiMo/MiMo-V2-Flash"
local_dir = "./tokenizers/MiMo-V2-Flash"

files_to_download = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "config.json"
]

os.makedirs(local_dir, exist_ok=True)

for file in files_to_download:
    print(f"Downloading {file}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=local_dir
        )
        print(f"Successfully downloaded {file}")
    except Exception as e:
        print(f"Failed to download {file}: {e}")

print("Download complete.")
