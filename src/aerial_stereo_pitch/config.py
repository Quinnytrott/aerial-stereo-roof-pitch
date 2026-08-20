"""Configuration loading and fail-closed public-path validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an input cannot be interpreted safely."""


def load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not load JSON config {candidate}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("Top-level config must be a JSON object")
    return value


def require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ConfigError(f"{context} is missing required keys: {', '.join(missing)}")


def safe_relative_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{context} must stay within the configured private root")
    return path


def write_stable_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = stable_json_text(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".json-stage-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stable_json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
