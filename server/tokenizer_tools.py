"""Read-only, registered tokenizer tools for run setup."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import HF_MODEL_MAPPING, TOKENIZER_SOURCES


def tokenizer_catalog(model_id: str) -> dict[str, Any]:
    root = Path("tokenizers")
    items = [
        {"name": name, "available": (root / name).is_dir() and any((root / name).iterdir())}
        for name in sorted(TOKENIZER_SOURCES)
    ]
    matched = next(
        (Path(path).name for key, path in HF_MODEL_MAPPING.items() if key in model_id.lower()),
        None,
    )
    return {"items": items, "matched_name": matched}


@lru_cache(maxsize=8)
def _load_local(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path("tokenizers") / name),
        local_files_only=True,
        trust_remote_code=False,
    )


def count_text(text: str, mode: str, name: str | None) -> dict[str, Any]:
    if mode == "characters":
        return {"count": len(text), "unit": "characters", "method": "Unicode code points"}
    if mode == "tiktoken":
        import tiktoken

        try:
            count = len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception as exc:
            raise ValueError("Reference tokenizer is unavailable") from exc
        return {
            "count": count,
            "unit": "tokens",
            "method": "cl100k_base reference; may differ from target model",
        }
    if mode != "local" or not name or name not in TOKENIZER_SOURCES:
        raise ValueError("Select a registered local tokenizer")
    path = Path("tokenizers") / name
    if not path.is_dir() or not any(path.iterdir()):
        raise ValueError("Tokenizer is not installed locally")
    try:
        tokenizer = _load_local(name)
        count = len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:
        raise ValueError("Tokenizer cannot be loaded without remote code") from exc
    return {"count": count, "unit": "tokens", "method": f"local/{name}; no special tokens"}
