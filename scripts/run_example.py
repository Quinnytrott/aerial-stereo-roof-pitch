#!/usr/bin/env python3
"""Run the deterministic analytic experiment without requiring installation."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_stereo_pitch.config import load_json, stable_json_text  # noqa: E402
from aerial_stereo_pitch.diagnostics import write_pitch_profile_svg, write_summary  # noqa: E402
from aerial_stereo_pitch.synthetic import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples" / "example_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", type=Path, help="fail unless summary bytes equal this file")
    args = parser.parse_args()

    summary = run_experiment(load_json(args.config))
    scored = [scene for scene in summary["scenes"] if scene["status"] == "scored"]
    summary_bytes = stable_json_text(summary).encode("utf-8")
    if args.check and summary_bytes != args.check.read_bytes():
        print(f"summary differs from {args.check}", file=sys.stderr)
        return 2
    if args.output_dir.exists():
        print("REFUSED: --output-dir must be a new path", file=sys.stderr)
        return 2
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".example-stage-", dir=args.output_dir.parent))
    try:
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
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    summary_path = args.output_dir / "summary.json"
    counts = summary["counts"]
    print(
        f"wrote {summary_path}: scored={counts['scored']} refused={counts['refused']} "
        f"failed={counts['failed']} missing={counts['missing']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
