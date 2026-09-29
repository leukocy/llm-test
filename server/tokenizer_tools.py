"""Registered local tokenizer inventory, reference encoding and text counters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import HF_MODEL_MAPPING, TOKENIZER_SOURCES
from config.tokenizer_paths import installation_manifest, registered_tokenizer_path
from core.tokenizer_utils import get_cached_tokenizer


def tokenizer_catalog(model_id: str) -> dict[str, Any]:
    items = []
    for name in sorted(TOKENIZER_SOURCES):
        path = registered_tokenizer_path(name)
        manifest = installation_manifest(path) if path else None
        source = TOKENIZER_SOURCES[name]
        items.append(
            {
                "name": name,
                "available": path is not None,
                "repo_id": source if isinstance(source, str) else source["hf"],
                "source": "installed" if manifest else "local" if path else None,
                "revision": manifest.get("revision") if manifest else None,
            }
        )
    matched = next(
        (Path(path).name for key, path in HF_MODEL_MAPPING.items() if key in model_id.lower()),
        None,
    )
    return {"items": items, "matched_name": matched}


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
    path = registered_tokenizer_path(name)
    if path is None:
        raise ValueError("Tokenizer is not installed locally")
    try:
        tokenizer = get_cached_tokenizer(str(path))
        if tokenizer is None:
            raise ValueError("Local load failed")
        count = len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:
        raise ValueError("Tokenizer cannot be loaded without remote code") from exc
    return {"count": count, "unit": "tokens", "method": f"local/{name}; no special tokens"}
