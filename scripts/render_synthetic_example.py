#!/usr/bin/env python3
"""Render the tracked synthetic roof-plane reconstruction as a technical PNG."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerial_stereo_pitch.config import ConfigError, load_json  # noqa: E402
from aerial_stereo_pitch.pitch import pitch_from_plane  # noqa: E402
from aerial_stereo_pitch.synthetic import run_scene_with_trace  # noqa: E402

DEFAULT_SCENE = "seeded-noise-and-outliers-6-in-12"
PIXEL_SIZE = (1960, 1120)


def _scene_from_config(config: dict[str, object], scene_id: str) -> dict[str, object]:
    scenes = config.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("synthetic config requires a scenes list")
    for scene in scenes:
        if isinstance(scene, dict) and scene.get("id") == scene_id:
            return scene
    raise ValueError(f"scene {scene_id!r} was not found in the config")


def render(config_path: Path, scene_id: str, output: Path) -> None:
    if output.exists():
        raise ValueError("--output must be a new path")
    config = load_json(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("synthetic config schema_version must be 1")
    scene = _scene_from_config(config, scene_id)
    result, trace = run_scene_with_trace(scene, int(config.get("seed", 0)))
    if result.status != "scored" or trace is None:
        raise ValueError(f"selected scene did not score: {result.detail.get('reason', result.status)}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Matplotlib is required; install the 'plots' or 'test' extra") from exc
    plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.monospace": ["DejaVu Sans Mono"],
            "axes.unicode_minus": True,
        }
    )

    pair_colors = {"pair-a": "#0072B2", "pair-b": "#D55E00"}
    shared = trace.shared_fit
    truth_slope = trace.truth_rise_per_12 / 12.0
    x_values = np.linspace(-4.3, 4.3, 29)
    y_values = np.linspace(-4.3, 4.3, 29)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    truth_z = truth_slope * grid_x
    fitted_z = shared.model.predict(np.column_stack((grid_x.ravel(), grid_y.ravel()))).reshape(
        grid_x.shape
    )

    figure = plt.figure(figsize=(14, 8), dpi=140, facecolor="white")
    layout = figure.add_gridspec(2, 2, width_ratios=(1.55, 1.0), height_ratios=(1.1, 0.9))
    axes_3d = figure.add_subplot(layout[:, 0], projection="3d")
    axes_residual = figure.add_subplot(layout[0, 1])
    axes_facts = figure.add_subplot(layout[1, 1])

    axes_3d.plot_surface(
        grid_x,
        grid_y,
        truth_z,
        color="#8EC9E8",
        alpha=0.28,
        linewidth=0,
        antialiased=True,
        label="analytic truth plane",
    )
    axes_3d.plot_wireframe(
        grid_x,
        grid_y,
        fitted_z,
        color="#2F3E46",
        linewidth=0.7,
        rstride=4,
        cstride=4,
        alpha=0.72,
        label="balanced fitted plane",
    )

    residual_cursor = 0
    offscale_residuals: list[float] = []
    for pair_id, cloud in sorted(trace.pair_clouds.items()):
        residuals = shared.model.residuals(cloud)
        shared_mask = np.abs(residuals) <= trace.residual_threshold
        color = pair_colors[pair_id]
        axes_3d.scatter(
            cloud[shared_mask, 0],
            cloud[shared_mask, 1],
            cloud[shared_mask, 2],
            s=21,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            depthshade=False,
            label=f"{pair_id.title()} shared inliers",
        )
        if np.any(~shared_mask):
            axes_3d.scatter(
                cloud[~shared_mask, 0],
                cloud[~shared_mask, 1],
                cloud[~shared_mask, 2],
                s=30,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                depthshade=False,
                label=f"{pair_id.title()} rejected by shared fit",
            )
        indices = np.arange(len(cloud)) + residual_cursor
        display_limit = 0.052
        visible = np.abs(residuals) <= display_limit
        axes_residual.scatter(
            indices[visible], residuals[visible], s=15, color=color, alpha=0.8, label=pair_id.title()
        )
        if np.any(~visible):
            clipped = np.sign(residuals[~visible]) * display_limit
            axes_residual.scatter(
                indices[~visible], clipped, s=38, color=color, marker="v", edgecolors="black",
                linewidths=0.4,
            )
            offscale_residuals.extend(float(value) for value in residuals[~visible])
        residual_cursor += len(cloud)

    axes_3d.set_xlabel("World X (m)", labelpad=8)
    axes_3d.set_ylabel("World Y (m)", labelpad=8)
    axes_3d.set_zlabel("World Z (m)", labelpad=8)
    axes_3d.set_xlim(-4.5, 4.5)
    axes_3d.set_ylim(-4.5, 4.5)
    axes_3d.set_zlim(-2.4, 2.4)
    axes_3d.set_box_aspect((9.0, 9.0, 4.8))
    axes_3d.view_init(elev=24, azim=-55)
    axes_3d.legend(loc="upper left", fontsize=8, frameon=True)
    axes_3d.set_title("World-space roof-plane reconstruction", loc="left", pad=12)

    threshold = trace.residual_threshold
    axes_residual.axhspan(-threshold, threshold, color="#A7C7A1", alpha=0.2)
    axes_residual.axhline(threshold, color="#4D7C52", linestyle="--", linewidth=1.1)
    axes_residual.axhline(-threshold, color="#4D7C52", linestyle="--", linewidth=1.1)
    axes_residual.axhline(0.0, color="#777777", linewidth=0.7)
    axes_residual.set_title("Balanced-plane vertical residuals", loc="left")
    axes_residual.set_xlabel("Accepted triangulated observation")
    axes_residual.set_ylabel("Vertical residual Z (m)")
    axes_residual.set_ylim(-0.057, 0.057)
    axes_residual.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes_residual.legend(loc="upper right", fontsize=8)
    axes_residual.text(
        0.01,
        0.94,
        f"Shared inlier threshold: ±{threshold:.02f} m",
        transform=axes_residual.transAxes,
        va="top",
        fontsize=8,
        color="#365B3A",
    )
    if offscale_residuals:
        values = ", ".join(f"{value:.2f} m" for value in offscale_residuals)
        axes_residual.text(
            0.99,
            0.04,
            f"Triangle clipped for readability; actual residual: {values}",
            transform=axes_residual.transAxes,
            ha="right",
            fontsize=7.4,
            color="#7A3E00",
        )

    pair_pitch = {
        pair_id: pitch_from_plane(fit.model).rise_per_12
        for pair_id, fit in sorted(trace.pair_fits.items())
    }
    balanced_pitch = pitch_from_plane(shared.model).rise_per_12
    pair_counts = {
        pair_id: f"{values['accepted']}/{values['input']}"
        for pair_id, values in sorted(trace.pair_diagnostics.items())
    }
    facts = (
        "Synthetic known-truth reconstruction\n\n"
        f"True pitch                 {trace.truth_rise_per_12:.6f}/12\n"
        f"Pair A pitch               {pair_pitch['pair-a']:.6f}/12\n"
        f"Pair B pitch               {pair_pitch['pair-b']:.6f}/12\n"
        f"Balanced pitch             {balanced_pitch:.9f}/12\n"
        f"Absolute pitch error       {abs(balanced_pitch - trace.truth_rise_per_12):.9f}/12\n\n"
        f"Accepted Pair A / Pair B   {pair_counts['pair-a']} / {pair_counts['pair-b']}\n"
        f"Shared inliers             {shared.inlier_count}/{shared.input_count}\n"
        f"Balanced vertical RMS Z    {shared.rms_z:.9f} m\n"
        f"Residual threshold         ±{threshold:.2f} m\n\n"
        "Calibrated analytic cameras → correspondence\n"
        "→ triangulation → robust pair fits → balanced plane"
    )
    axes_facts.axis("off")
    axes_facts.text(
        0.0,
        1.0,
        facts,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.3,
        linespacing=1.32,
        color="#1F2933",
    )

    figure.suptitle(
        "Synthetic stereo roof-plane reconstruction",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.945,
        "Known cameras + known plane + seeded image noise/outliers; not real imagery or real-world accuracy evidence",
        ha="left",
        fontsize=10,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.045, right=0.98, top=0.895, bottom=0.07, wspace=0.18, hspace=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=".synthetic-render-", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            dpi=140,
            metadata={
                "Title": "Synthetic stereo roof-plane reconstruction",
                "Description": "Analytic known-truth calibrated stereo validation; no real imagery.",
                "Software": "aerial-stereo-roof-pitch",
            },
        )
        plt.close(figure)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples" / "example_config.json")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "synthetic-roof-plane-reconstruction.png",
        help="new PNG path (default: outputs/synthetic-roof-plane-reconstruction.png)",
    )
    args = parser.parse_args()
    try:
        render(args.config, args.scene_id, args.output)
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} from public synthetic scene {args.scene_id!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
