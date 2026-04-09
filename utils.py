"""Utility functions for the TikTok Analyzer Streamlit app."""

import io
import json
import shutil
import sys
import contextlib
from pathlib import Path


@contextlib.contextmanager
def capture_stdout():
    """Context manager that captures stdout to a StringIO buffer."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def check_deps() -> list[str]:
    """Return list of missing dependency names. Empty list means all good."""
    missing = []
    for tool in ("gallery-dl", "yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            missing.append(tool)
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("openai-whisper")
    return missing


def load_results(data_dir: Path, handles: list[str]) -> list[dict]:
    """Load all post.json results for the given handles."""
    results = []
    for handle in handles:
        handle_dir = data_dir / handle
        if not handle_dir.exists():
            continue
        for p in sorted(handle_dir.rglob("post.json")):
            with open(p) as f:
                results.append(json.load(f))
    return results


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
