#!/usr/bin/env python3
"""
Materialize all tokenizers referenced by HF_MODEL_MAPPING into real files.

Why: The image must be self-contained for offline deployment. Tokenizer dirs are a
mix of real dirs (clean) and symlinks into /DATA/Model/<full-model-dir> (huge —
contains weights). This script copies ONLY tokenizer files (following symlinks),
producing a clean, bakeable ./tokenizers tree with no weights and no symlinks.

Source precedence for each tokenizer dir name:
  1. Existing ./tokenizers/<name> (real dir → copy tokenizer files)
  2. Symlink ./tokenizers/<name> → resolve target, copy tokenizer files
  3. DATA_MODEL_SOURCES map → /DATA/Model/<dir>
  4. /DATA/Model/<name> directly
  5. ModelScope download (build machine must have network) via TOKENIZER_SOURCES

Usage:
    python scripts/prepare_tokenizers.py          # materialize into ./tokenizers
    python scripts/prepare_tokenizers.py --check  # report status only, no writes
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

# Ensure repo root is importable when run from anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import HF_MODEL_MAPPING, TOKENIZER_SOURCES  # noqa: E402

TOKENIZERS_DIR = REPO_ROOT / "tokenizers"
# Materialized output: a clean, user-writable bundle copied into the image.
# HF_MODEL_MAPPING uses ./tokenizers/<name> at runtime; the Dockerfile maps this
# bundle onto /app/tokenizers/, so paths resolve correctly without modifying the
# (root-owned) dev tokenizers/ tree.
BUNDLE_DIR = REPO_ROOT / "tokenizers_bundle"
DATA_MODEL_ROOT = Path("/DATA/Model")

# Tokenizer files are typically in preprocessor_config.json, chat_template.jinja, vocab.json,
# merges.txt, special_tokens_map.json, tokenizer.json, etc.
#
# When the model directory contains tokenizer files but uses SentencePiece, the actual vocab
# file is named *.model (e.g., tokenizer.model). These are kept too.
TOKENIZER_FILENAME_PATTERNS = [
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
    "generation_config.json",
    "*.model",
]

# Hardcoded mapping for symlinked tokenizer dirs → /DATA/Model source.
# Encoded here so a fresh clone (no symlinks) can still resolve local model dirs
# without re-downloading.
DATA_MODEL_SOURCES = {
    "Qwen3.5-397B-A17B-FP8": "Qwen3.5-9B",
    "DeepSeek-V4-Pro": "DeepSeek-V4-Flash",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    "MiMo-V2-Flash": "MiMo-V2.5",
    "MiniMax-M2": "MiniMax-M2.7",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen3.6-35B-A3B-FP8",
}

# Tokenizer dirs that share a tokenizer with another (already-local) dir.
# Used when a symlink target in /DATA/Model is missing but an identical
# tokenizer exists under a different name in ./tokenizers.
LOCAL_TOKENIZER_ALIASES = {
    "MiMo-V2-Flash": "MiMo-V2.5",
}


def is_tokenizer_file(filename: str) -> bool:
    """Match filename against tokenizer whitelist patterns."""
    name = filename.lower()
    return any(
        fnmatch.fnmatch(name, pat.lower()) for pat in TOKENIZER_FILENAME_PATTERNS
    )


def collect_tokenizer_files(src_dir: Path) -> list[Path]:
    """Collect only tokenizer files from a source directory (non-recursive top level)."""
    files = []
    try:
        for entry in src_dir.iterdir():
            if entry.is_file() and is_tokenizer_file(entry.name):
                files.append(entry)
    except (PermissionError, OSError):
        pass
    return sorted(files, key=lambda p: p.name)


def resolve_source(name: str) -> Path | None:
    """Find the real source directory holding tokenizer files for `name`."""
    candidates: list[Path] = []

    local = TOKENIZERS_DIR / name
    # Symlink → resolve its target
    if local.is_symlink():
        resolved = Path(os.readlink(local))
        if not resolved.is_absolute():
            resolved = TOKENIZERS_DIR / resolved
        candidates.append(resolved)
    elif local.is_dir():
        candidates.append(local)

    # Explicit /DATA/Model map
    dm_name = DATA_MODEL_SOURCES.get(name)
    if dm_name:
        candidates.append(DATA_MODEL_ROOT / dm_name)

    # Local tokenizer alias (shared tokenizer under a different name)
    alias = LOCAL_TOKENIZER_ALIASES.get(name)
    if alias:
        candidates.append(TOKENIZERS_DIR / alias)

    # /DATA/Model/<name>
    candidates.append(DATA_MODEL_ROOT / name)

    for cand in candidates:
        if cand.is_dir() and collect_tokenizer_files(cand):
            return cand
    return None


def materialize(name: str, src: Path, dst: Path, dry_run: bool) -> tuple[bool, str]:
    """Copy tokenizer files from src into dst (clean, real files only)."""
    files = collect_tokenizer_files(src)
    if not files:
        return False, f"no tokenizer files in {src}"

    total_kb = sum(f.stat().st_size for f in files) // 1024
    if dry_run:
        return True, f"would copy {len(files)} files (~{total_kb} KB) from {src}"

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        shutil.copy2(f, dst / f.name)
    return True, f"copied {len(files)} files (~{total_kb} KB) from {src}"


def download_from_modelscope(name: str, dst: Path) -> tuple[bool, str]:
    """Fall back: download tokenizer from ModelScope (needs network)."""
    source = TOKENIZER_SOURCES.get(name)
    if source is None:
        return False, "no TOKENIZER_SOURCES entry"
    repo_id = source if isinstance(source, str) else source.get("ms")
    if not repo_id:
        return False, "no modelscope repo id"

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        return False, "modelscope not installed"

    os.makedirs(dst, exist_ok=True)
    try:
        snapshot_download(
            repo_id,
            local_dir=str(dst),
            allow_patterns=TOKENIZER_FILENAME_PATTERNS,
        )
        return True, f"downloaded from ModelScope {repo_id}"
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(dst, ignore_errors=True)
        return False, f"download failed: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize tokenizers for offline image build"
    )
    parser.add_argument(
        "--check", action="store_true", help="report status without writing"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BUNDLE_DIR,
        help=f"output directory (default: {BUNDLE_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    # Unique tokenizer dir names referenced by HF_MODEL_MAPPING
    names: list[str] = []
    seen: set[str] = set()
    for target in HF_MODEL_MAPPING.values():
        name = Path(target).name
        if name not in seen:
            seen.add(name)
            names.append(name)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    print(f"Processing {len(names)} unique tokenizer dirs → {out_dir}")
    print("-" * 70)
    for name in names:
        dst = out_dir / name
        src = resolve_source(name)
        if src is not None:
            success, msg = materialize(name, src, dst, args.check)
        else:
            # No local source → download (build machine must have network)
            if args.check:
                success, msg = False, "no local source (would try ModelScope download)"
            else:
                success, msg = download_from_modelscope(name, dst)
        status = "OK " if success else "FAIL"
        print(f"  [{status}] {name:40s} {msg}")
        if success:
            ok += 1
        else:
            fail += 1

    print("-" * 70)
    print(f"Done: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
