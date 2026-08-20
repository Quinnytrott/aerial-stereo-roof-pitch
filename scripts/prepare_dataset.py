#!/usr/bin/env python3
"""Validate a user-acquired Ontario private-data layout; never download or copy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_stereo_pitch.config import ConfigError, load_json, require_keys, safe_relative_path  # noqa: E402


def validate_layout(config_path: Path) -> list[str]:
    config = load_json(config_path)
    require_keys(config, {"rights_acknowledged", "private_data_root", "required_paths"}, "dataset config")
    if config["rights_acknowledged"] is not True:
        raise ConfigError(
            "rights_acknowledged must be true after the user verifies licence, third-party rights, and access"
        )
    root_value = safe_relative_path(config["private_data_root"], "private_data_root")
    data_root = (config_path.parent / root_value).resolve()
    required = config["required_paths"]
    if not isinstance(required, list) or not required:
        raise ConfigError("required_paths must be a non-empty list")
    missing = []
    for index, value in enumerate(required):
        relative = safe_relative_path(value, f"required_paths[{index}]")
        candidate = (data_root / relative).resolve()
        if data_root not in candidate.parents:
            raise ConfigError("required path escapes private_data_root")
        if not candidate.is_file():
            missing.append(str(relative))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        missing = validate_layout(args.config)
    except ConfigError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if missing:
        print("REFUSED: user-acquired files are missing from the configured private root:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        return 2
    print("Dataset layout is present. No files were downloaded, copied, or published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
