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


@st.cache_resource
def get_cached_tokenizer(model_path):
    """
    Load a HuggingFace tokenizer with caching.
    Supports smart local directory search for offline use.
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
                return AutoTokenizer.from_pretrained(candidate, trust_remote_code=True, local_files_only=True)
            except Exception as e:
                last_error = e
                logger.debug(f"Failed local load for {candidate}: {e}")
                pass

    # Finally, try unsafe/online load on the original path as last resort
    # This covers cases where it is a hub ID not present locally
    try:
        if "/" in model_path and not os.path.exists(model_path):
             return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        last_error = e

    if last_error:
        logger.warning(f"Failed to load tokenizer '{model_path}'. Last error: {last_error}")

    return None
