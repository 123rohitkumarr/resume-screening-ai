import logging
import re
import sys
import os
import json
import uuid
from typing import Any


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configure a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def clean_text(text: str) -> str:
    """Strip extra whitespace and non-printable characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def normalize_skill(skill: str) -> str:
    """Lowercase, strip, and collapse separators for skill comparison."""
    s = skill.lower().strip()
    s = re.sub(r"[-_./]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def safe_json_loads(text: str, default=None) -> Any:
    """Parse JSON with a fallback value."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []


def generate_session_id() -> str:
    return uuid.uuid4().hex[:12]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def flatten_list(nested: list) -> list:
    """Flatten one level of nesting."""
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
