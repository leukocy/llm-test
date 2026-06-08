import os

import streamlit as st

from utils.get_logger import get_logger

# Module logger
logger = get_logger(__name__)

# LatencyImport transformers（重型库）
_AutoTokenizer = None

def _get_auto_tokenizer():
    """LatencyGet AutoTokenizer 类"""
    global _AutoTokenizer
    if _AutoTokenizer is None:
        from transformers import AutoTokenizer
        _AutoTokenizer = AutoTokenizer
    return _AutoTokenizer


def _is_extra_special_tokens_list_error(error: Exception) -> bool:
    return (
        isinstance(error, AttributeError)
        and "'list' object has no attribute 'keys'" in str(error)
    )


def _from_pretrained_with_compat(AutoTokenizer, model_path, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except Exception as e:
        if not _is_extra_special_tokens_list_error(e):
            raise

        compat_kwargs = dict(kwargs)
        compat_kwargs.setdefault("extra_special_tokens", {})
        logger.info(
            "Retrying tokenizer load with extra_special_tokens compatibility: "
            f"{model_path}"
        )
        return AutoTokenizer.from_pretrained(model_path, **compat_kwargs)


def _resolve_hf_repo_id(local_dir_name: str) -> str | None:
    """
    Resolve a local tokenizer directory name to its HuggingFace Hub repo ID.

    Args:
        local_dir_name: The local directory name (e.g. "DeepSeek-V3.1-Terminus")

    Returns:
        HuggingFace Hub repo ID (e.g. "deepseek-ai/DeepSeek-V3") or None
    """
    from config.settings import TOKENIZER_SOURCES

    source = TOKENIZER_SOURCES.get(local_dir_name)
    if source is None:
        return None
    return source if isinstance(source, str) else source["hf"]


def _resolve_modelscope_repo_id(local_dir_name: str) -> str | None:
    """
    Resolve a local tokenizer directory name to its ModelScope repo ID.

    Args:
        local_dir_name: The local directory name (e.g. "DeepSeek-V3.1-Terminus")

    Returns:
        ModelScope repo ID (e.g. "deepseek-ai/DeepSeek-V3") or None
    """
    from config.settings import TOKENIZER_SOURCES

    source = TOKENIZER_SOURCES.get(local_dir_name)
    if source is None:
        return None
    return source if isinstance(source, str) else source["ms"]


def _download_tokenizer_from_modelscope(repo_id: str, local_dir: str) -> bool:
    """
    Download tokenizer files from ModelScope to a local directory.

    Args:
        repo_id: ModelScope repo ID (e.g. "deepseek-ai/DeepSeek-V3")
        local_dir: Local directory to save tokenizer files

    Returns:
        True if download succeeded
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        logger.debug("modelscope library not installed, skipping ModelScope download")
        return False

    try:
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Downloading tokenizer from ModelScope: {repo_id} -> {local_dir}")

        snapshot_download(
            repo_id,
            local_dir=local_dir,
            allow_patterns=[
                "tokenizer*",
                "vocab*",
                "merges.*",
                "special_tokens_map*",
                "added_tokens*",
                "tokenization*",
                "preprocessor_config.*",
                "*.tiktoken",
                "chat_template*",
                "config.json",
            ],
        )
        logger.info(f"Tokenizer downloaded from ModelScope and saved to {local_dir}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download tokenizer from ModelScope {repo_id}: {e}")
        return False


def _resolve_modelscope_id(local_dir_name: str) -> str | None:
    """Resolve a local tokenizer directory name to its ModelScope repo ID."""
    from config.settings import TOKENIZER_MODELSCOPE_MAPPING
    return TOKENIZER_MODELSCOPE_MAPPING.get(local_dir_name)


def _download_tokenizer_from_hf(hf_repo_id: str, local_dir: str) -> bool:
    """
    Download tokenizer files from HuggingFace Hub to a local directory.

    Args:
        hf_repo_id: HuggingFace Hub repo ID (e.g. "deepseek-ai/DeepSeek-V3")
        local_dir: Local directory to save tokenizer files

    Returns:
        True if download succeeded
    """
    try:
        AutoTokenizer = _get_auto_tokenizer()
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Downloading tokenizer from HuggingFace: {hf_repo_id} -> {local_dir}")

        # Load from hub and save locally
        tokenizer = _from_pretrained_with_compat(
            AutoTokenizer,
            hf_repo_id,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(local_dir)
        logger.info(f"Tokenizer downloaded and saved to {local_dir}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download tokenizer from {hf_repo_id}: {e}")
        return False


def _get_local_dir_name(model_path: str) -> str:
    """Determine the local directory name from a model path."""
    if model_path.startswith("./tokenizers/"):
        return os.path.basename(model_path)
    elif "/" in model_path and not os.path.isabs(model_path):
        return model_path.split("/")[-1]
    else:
        return os.path.basename(model_path)


def ensure_tokenizer_available(model_path: str) -> str | None:
    """
    Ensure a tokenizer is available locally, downloading from ModelScope (优先)
    or HuggingFace Hub if needed.

    Args:
        model_path: The tokenizer path (local path, ./tokenizers/X, or repo ID)

    Returns:
        The local path where the tokenizer is available, or None if unavailable
    """
    tokenizers_dir = os.path.join(".", "tokenizers")

    # Build candidate local paths (same logic as get_cached_tokenizer)
    candidates = [model_path]
    if not os.path.isabs(model_path) and not model_path.startswith("./"):
        candidates.append(os.path.join(tokenizers_dir, model_path))
        if "/" in model_path:
            candidates.append(os.path.join(tokenizers_dir, model_path.split("/")[-1]))

    # Check if any local path already exists
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Determine local dir name
    local_dir_name = _get_local_dir_name(model_path)
    for c in candidates:
        name = os.path.basename(c)
        if name:
            local_dir_name = name
            break

    local_dir = os.path.join(tokenizers_dir, local_dir_name)

    # 1. Try ModelScope first (优先国内源)
    ms_repo_id = _resolve_modelscope_repo_id(local_dir_name)
    if not ms_repo_id and "/" in model_path and not model_path.startswith(".") and not os.path.isabs(model_path):
        ms_repo_id = model_path
    if ms_repo_id and _download_tokenizer_from_modelscope(ms_repo_id, local_dir):
        return local_dir

    # 2. Fall back to HuggingFace Hub
    hf_repo_id = _resolve_hf_repo_id(local_dir_name)
    if not hf_repo_id and "/" in model_path and not model_path.startswith(".") and not os.path.isabs(model_path):
        hf_repo_id = model_path
    if hf_repo_id and _download_tokenizer_from_hf(hf_repo_id, local_dir):
        return local_dir

    return None


@st.cache_resource
def get_cached_tokenizer(model_path):
    """
    Load a HuggingFace tokenizer with caching.
    Supports smart local directory search and auto-download from HuggingFace.
    """
    AutoTokenizer = _get_auto_tokenizer()
    candidates = []

    # 1. As-is path
    candidates.append(model_path)

    # 2. Local 'tokenizers' directory prefix
    if not os.path.isabs(model_path) and not model_path.startswith("./"):
        candidates.append(os.path.join(".", "tokenizers", model_path))
        # Handle "repo/model" -> "./tokenizers/model"
        if "/" in model_path:
            candidates.append(os.path.join(".", "tokenizers", model_path.split("/")[-1]))

    # 3. Fuzzy match in tokenizers directory (if directory exists)
    tokenizers_dir = os.path.join(".", "tokenizers")
    if os.path.exists(tokenizers_dir):
        try:
            target_name = model_path.split("/")[-1].lower() if "/" in model_path else model_path.lower()
            for item in os.listdir(tokenizers_dir):
                if target_name in item.lower() or item.lower() in target_name:
                    candidates.append(os.path.join(tokenizers_dir, item))
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    # Attempt load
    last_error = None

    # Try local loads first
    for candidate in unique_candidates:
        if os.path.exists(candidate): # Only try if path exists
            try:
                logger.debug(f"Trying local tokenizer: {candidate}")
                return _from_pretrained_with_compat(
                    AutoTokenizer,
                    candidate,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                last_error = e
                logger.debug(f"Failed local load for {candidate}: {e}")
                pass

    # No local copy found — try auto-download
    local_path = ensure_tokenizer_available(model_path)
    if local_path and os.path.exists(local_path):
        try:
            logger.info(f"Using auto-downloaded tokenizer: {local_path}")
            return _from_pretrained_with_compat(
                AutoTokenizer,
                local_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            last_error = e

    # Finally, try online load on the original path as last resort
    try:
        if "/" in model_path and not os.path.exists(model_path):
             return _from_pretrained_with_compat(
                 AutoTokenizer,
                 model_path,
                 trust_remote_code=True,
             )
    except Exception as e:
        last_error = e

    if last_error:
        logger.warning(f"Failed to load tokenizer '{model_path}'. Last error: {last_error}")

    return None


def list_registered_tokenizers() -> list[dict]:
    """
    List all registered tokenizers with their availability status.

    Returns:
        List of dicts with keys: name, local_path, available, hf_repo_id,
        modelscope_repo_id, size_mb
    """
    from config.settings import TOKENIZER_SOURCES

    tokenizers_dir = os.path.join(".", "tokenizers")
    results = []

    for local_name in sorted(TOKENIZER_SOURCES):
        source = TOKENIZER_SOURCES.get(local_name)
        if isinstance(source, str):
            ms_repo_id = source
            hf_repo_id = source
        else:
            ms_repo_id = source["ms"]
            hf_repo_id = source["hf"]

        local_path = os.path.join(tokenizers_dir, local_name)
        available = os.path.exists(local_path)

        size_mb = 0.0
        if available:
            try:
                size_mb = sum(
                    os.path.getsize(os.path.join(local_path, f))
                    for f in os.listdir(local_path)
                    if os.path.isfile(os.path.join(local_path, f))
                ) / (1024 * 1024)
            except Exception:
                pass

        results.append({
            "name": local_name,
            "local_path": local_path,
            "available": available,
            "modelscope_repo_id": ms_repo_id,
            "hf_repo_id": hf_repo_id,
            "size_mb": size_mb,
        })

    return results
