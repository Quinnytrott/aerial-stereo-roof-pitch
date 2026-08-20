#!/usr/bin/env python3
"""Run the deterministic analytic experiment without requiring installation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_stereo_pitch.config import ConfigError, load_json, stable_json_text  # noqa: E402
from aerial_stereo_pitch.diagnostics import write_pitch_profile_svg, write_summary  # noqa: E402
from aerial_stereo_pitch.synthetic import run_experiment  # noqa: E402

_MISSING = object()
_MAX_SEMANTIC_DIFFERENCES = 3


def _json_path_key(path: str, key: str) -> str:
    return f"{path}.{key}" if key.isidentifier() else f"{path}[{json.dumps(key)}]"


def _anonymous_key_path(path: str, side: str, key: str, index: int) -> str:
    return f"{path}[<{side}-only-key#{index},len={len(key)}>]"


def _scalar_equal(expected: object, actual: object) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, float):
        if expected != actual:
            return False
        if expected == 0.0:
            return math.copysign(1.0, expected) == math.copysign(1.0, actual)
        return True
    return expected == actual


def _semantic_differences(
    expected: object, actual: object, path: str = "$"
) -> list[tuple[str, object, object]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[tuple[str, object, object]] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys & actual_keys):
            child_path = _json_path_key(path, key)
            differences.extend(_semantic_differences(expected[key], actual[key], child_path))
        for index, key in enumerate(sorted(expected_keys - actual_keys), start=1):
            child_path = _anonymous_key_path(path, "expected", key, index)
            differences.append((child_path, expected[key], _MISSING))
        for index, key in enumerate(sorted(actual_keys - expected_keys), start=1):
            child_path = _anonymous_key_path(path, "actual", key, index)
            differences.append((child_path, _MISSING, actual[key]))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            child_path = f"{path}[{index}]"
            if index >= len(expected):
                differences.append((child_path, _MISSING, actual[index]))
            elif index >= len(actual):
                differences.append((child_path, expected[index], _MISSING))
            else:
                differences.extend(
                    _semantic_differences(expected[index], actual[index], child_path)
                )
        return differences
    return [] if _scalar_equal(expected, actual) else [(path, expected, actual)]


def _redacted_scalar(value: object, *, hide_scalar: bool = False) -> str:
    if value is _MISSING:
        return "missing"
    if isinstance(value, str):
        return f"string(len={len(value)})"
    if isinstance(value, dict):
        return f"object(keys={len(value)})"
    if isinstance(value, list):
        return f"array(len={len(value)})"
    if hide_scalar:
        if value is None:
            return "type=null"
        if isinstance(value, bool):
            return "type=boolean"
        if isinstance(value, (int, float)):
            return "type=number"
        return f"type={type(value).__name__}"
    if value is None:
        return "null"
    return repr(value)


def _mismatch_detail(expected_bytes: bytes, actual: object) -> str:
    try:
        expected = json.loads(expected_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return f"expected file is not valid JSON ({len(expected_bytes)} bytes)"
    differences = _semantic_differences(expected, actual)
    if not differences:
        return "semantic values match; byte formatting differs"
    rendered = []
    for path, expected_value, actual_value in differences[:_MAX_SEMANTIC_DIFFERENCES]:
        hide_scalar = expected_value is _MISSING or actual_value is _MISSING
        rendered.append(
            f"{path}: expected {_redacted_scalar(expected_value, hide_scalar=hide_scalar)}, "
            f"actual {_redacted_scalar(actual_value, hide_scalar=hide_scalar)}"
        )
    omitted = len(differences) - len(rendered)
    if omitted:
        rendered.append(f"... {omitted} more difference{'s' if omitted != 1 else ''}")
    return "; ".join(rendered)


def _pitch(value: object) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _print_summary(summary: dict[str, object], summary_path: Path) -> None:
    print("Synthetic known-truth validation (pitch is rise per 12; RMS Z is metres)")
    print(
        f"{'Scene':34} {'Truth':>9} {'Pair A':>9} {'Pair B':>9} {'Balanced':>9} "
        f"{'Error':>9} {'Accepted A/B':>14} {'Shared':>7} {'RMS Z':>9}  Verdict"
    )
    for scene in summary["scenes"]:
        status = str(scene["status"])
        if status == "scored":
            pairs = scene["pair_rise_per_12"]
            diagnostics = scene["pair_diagnostics"]
            accepted = (
                f"{diagnostics['pair-a']['accepted']}/{diagnostics['pair-a']['input']}|"
                f"{diagnostics['pair-b']['accepted']}/{diagnostics['pair-b']['input']}"
            )
            values = (
                _pitch(scene["truth_rise_per_12"]),
                _pitch(pairs["pair-a"]),
                _pitch(pairs["pair-b"]),
                _pitch(scene["recovered_rise_per_12"]),
                _pitch(scene["absolute_error_rise_per_12"]),
                accepted,
                str(scene["shared_inliers"]),
                f"{float(scene['shared_rms_z']):.6f}",
            )
            reason = "scored"
        else:
            values = ("-", "-", "-", "-", "-", "-", "-", "-")
            reason = f"{status}: {scene.get('reason', 'no reason recorded')}"
        print(
            f"{str(scene['id']):34.34} {values[0]:>9} {values[1]:>9} {values[2]:>9} "
            f"{values[3]:>9} {values[4]:>9} {values[5]:>14} {values[6]:>7} "
            f"{values[7]:>9}  {reason}"
        )
    counts = summary["counts"]
    print(
        f"Result: scored={counts['scored']} refused={counts['refused']} "
        f"failed={counts['failed']} missing={counts['missing']}"
    )
    print(f"Artifacts: {summary_path.parent}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples" / "example_config.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "example",
        help="new output directory (default: outputs/example relative to the current directory)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="fail unless summary bytes equal this file",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        print("REFUSED: --output-dir must be a new path", file=sys.stderr)
        return 2
    stage: Path | None = None
    try:
        summary = run_experiment(load_json(args.config))
        scored = [scene for scene in summary["scenes"] if scene["status"] == "scored"]
        summary_bytes = stable_json_text(summary).encode("utf-8")
        if args.check is not None:
            try:
                expected_bytes = args.check.read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"could not read expected file ({type(exc).__name__})"
                ) from None
            if summary_bytes != expected_bytes:
                detail = _mismatch_detail(expected_bytes, summary)
                raise ValueError(f"summary differs from expected file: {detail}")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".example-stage-", dir=args.output_dir.parent))
        summary_path = stage / "summary.json"
        write_summary(summary_path, summary)
        write_pitch_profile_svg(
            stage / "pitch-profile.svg",
            [
                (scene["id"], scene["truth_rise_per_12"], scene["recovered_rise_per_12"])
                for scene in scored
            ],
        )
        os.replace(stage, args.output_dir)
        stage = None
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    summary_path = args.output_dir / "summary.json"
    _print_summary(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
