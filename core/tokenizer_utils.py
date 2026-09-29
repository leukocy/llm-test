"""Offline tokenizer loading shared by counting, prompt calibration and workers."""

from __future__ import annotations

import functools
from pathlib import Path

from config.settings import TOKENIZER_SOURCES
from config.tokenizer_paths import installation_manifest, registered_tokenizer_path
from utils.get_logger import get_logger

logger = get_logger(__name__)
_AutoTokenizer = None


def _get_auto_tokenizer():
    global _AutoTokenizer
    if _AutoTokenizer is None:
        from transformers import AutoTokenizer

        _AutoTokenizer = AutoTokenizer
    return _AutoTokenizer


def _from_pretrained_with_compat(AutoTokenizer, model_path, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        compat_kwargs = dict(kwargs)
        compat_kwargs.setdefault("extra_special_tokens", {})
        return AutoTokenizer.from_pretrained(model_path, **compat_kwargs)


def local_tokenizer_path(model_path: str) -> Path | None:
    """Resolve registered overlays first; accept existing administrator-owned local paths."""
    path = Path(model_path)
    name = path.name
    for registered_name, registered_source in TOKENIZER_SOURCES.items():
        repos = (
            [registered_source]
            if isinstance(registered_source, str)
            else list(registered_source.values())
        )
        if model_path in repos:
            name = registered_name
            break
    source = TOKENIZER_SOURCES.get(name)
    repositories = [source] if isinstance(source, str) else list(source.values()) if source else []
    if model_path == name or path.as_posix() == f"tokenizers/{name}" or model_path in repositories:
        registered = registered_tokenizer_path(name)
        if registered is not None:
            return registered
    if path.is_dir():
        return path
    return None


@functools.lru_cache(maxsize=8)
def _load_local(path: str):
    return _from_pretrained_with_compat(
        _get_auto_tokenizer(), path, local_files_only=True, trust_remote_code=False
    )


def get_cached_tokenizer(model_path: str):
    """Never download during measurement, execute repository code, or cache failed loads."""
    path = local_tokenizer_path(model_path)
    if path is None:
        return None
    try:
        return _load_local(str(path.resolve()))
    except Exception as exc:
        logger.warning("Local tokenizer could not load: %s (%s)", path.name, type(exc).__name__)
        return None


def tokenizer_provenance(model_path: str | None) -> dict | None:
    if not model_path:
        return None
    path = local_tokenizer_path(model_path)
    return installation_manifest(path) if path else None


def list_registered_tokenizers() -> list[dict]:
    items = []
    for name, source in sorted(TOKENIZER_SOURCES.items()):
        path = registered_tokenizer_path(name)
        items.append(
            {
                "name": name,
                "local_path": str(path) if path else None,
                "available": path is not None,
                "hf_repo_id": source if isinstance(source, str) else source["hf"],
                "modelscope_repo_id": source if isinstance(source, str) else source["ms"],
                "installation": installation_manifest(path) if path else None,
            }
        )
    return items
