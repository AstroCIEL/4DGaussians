#!/usr/bin/env python3
"""
Analyze DyNeRF latency breakdown per scene.

Inputs (per scene directory under output/dynerf):
- analysis_per_kernel.csv:
    focus on rows where Metric Name == "gpu__time_duration.sum"
    kernel categories:
      - renderCUDA      -> rasterization stage
      - preprocessCUDA  -> preprocessing stage
      - everything else -> sorting stage
    note: Metric Unit can be ms/us/ns/s; we normalize to seconds.

- test/ours_14000/statistics.txt (by default):
    contains:
      - Deformation Time ave: <seconds>
      - Rasterization Time ave: <seconds>

We compute the 3-stage ratio from the CSV, then apply it to Rasterization Time ave
to get actual seconds for preprocessing/sorting/rasterization. Deformation time is
the 4th stage. Finally we plot a per-scene stacked bar chart.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


GPU_TIME_METRIC = "gpu__time_duration.sum"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _list_scene_dirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"dynerf base_dir not found: {base_dir}")
    out: List[str] = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            out.append(p)
    return out


def _unit_to_seconds_multiplier(unit: str) -> float:
    u = unit.strip().lower()
    if u == "s":
        return 1.0
    if u == "ms":
        return 1e-3
    if u == "us":
        return 1e-6
    if u == "ns":
        return 1e-9
    raise ValueError(f"Unsupported Metric Unit for gpu time: {unit!r}")


def _kernel_category(kernel_name: str) -> str:
    # NOTE: NCU kernel names include full signatures; substring match is enough.
    if "renderCUDA" in kernel_name:
        return "render"
    if "preprocessCUDA" in kernel_name:
        return "preprocess"
    return "sort"


@dataclass(frozen=True)
class KernelStageRatios:
    preprocess: float
    render: float
    sort: float

    @staticmethod
    def from_stage_seconds(preprocess_s: float, render_s: float, sort_s: float) -> "KernelStageRatios":
        total = preprocess_s + render_s + sort_s
        if total <= 0:
            return KernelStageRatios(preprocess=0.0, render=0.0, sort=0.0)
        return KernelStageRatios(
            preprocess=preprocess_s / total,
            render=render_s / total,
            sort=sort_s / total,
        )


def parse_analysis_per_kernel_csv(csv_path: str) -> Tuple[KernelStageRatios, Dict[str, float]]:
    """
    Returns:
      - ratios across (preprocess, render, sort) based on gpu__time_duration.sum
      - raw summed seconds per stage: {"preprocess": s, "render": s, "sort": s}
    """
    stage_sums_s: Dict[str, float] = {"preprocess": 0.0, "render": 0.0, "sort": 0.0}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Kernel Name", "Metric Name", "Metric Unit", "Average"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing columns: {sorted(missing)} (got: {reader.fieldnames})")

        for row in reader:
            if (row.get("Metric Name") or "").strip() != GPU_TIME_METRIC:
                continue

            kernel = (row.get("Kernel Name") or "").strip()
            unit = (row.get("Metric Unit") or "").strip()
            avg_str = (row.get("Average") or "").strip()
            if not kernel or not unit or not avg_str:
                continue

            try:
                avg = float(avg_str)
            except ValueError:
                continue

            mult = _unit_to_seconds_multiplier(unit)
            dur_s = avg * mult
            stage_sums_s[_kernel_category(kernel)] += dur_s

    ratios = KernelStageRatios.from_stage_seconds(
        preprocess_s=stage_sums_s["preprocess"],
        render_s=stage_sums_s["render"],
        sort_s=stage_sums_s["sort"],
    )
    return ratios, stage_sums_s


_DEFORM_RE = re.compile(r"^\s*Deformation\s+Time\s+ave:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
_RASTER_RE = re.compile(r"^\s*Rasterization\s+Time\s+ave:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")


def parse_statistics_txt(stat_path: str) -> Tuple[float, float]:
    """
    Returns (deformation_s, rasterization_s), both in seconds.
    """
    deformation_s: Optional[float] = None
    rasterization_s: Optional[float] = None
    with open(stat_path, "r") as f:
        for line in f:
            m = _DEFORM_RE.match(line)
            if m:
                deformation_s = float(m.group(1))
                continue
            m = _RASTER_RE.match(line)
            if m:
                rasterization_s = float(m.group(1))
                continue

    if deformation_s is None:
        raise ValueError(f"{stat_path}: cannot find 'Deformation Time ave'")
    if rasterization_s is None:
        raise ValueError(f"{stat_path}: cannot find 'Rasterization Time ave'")
    return deformation_s, rasterization_s


@dataclass(frozen=True)
class SceneBreakdown:
    scene: str
    deformation_s: float
    preprocess_s: float
    sort_s: float
    rasterize_s: float
    ratios: KernelStageRatios

    @property
    def total_s(self) -> float:
        return self.deformation_s + self.preprocess_s + self.sort_s + self.rasterize_s


def compute_scene_breakdown(scene_dir: str, statistics_relpath: str) -> SceneBreakdown:
    scene = os.path.basename(scene_dir.rstrip("/"))
    csv_path = os.path.join(scene_dir, "analysis_per_kernel.csv")
    stat_path = os.path.join(scene_dir, statistics_relpath)

    ratios, _raw_stage_sums = parse_analysis_per_kernel_csv(csv_path)
    deformation_s, rasterization_s = parse_statistics_txt(stat_path)

    preprocess_s = ratios.preprocess * rasterization_s
    sort_s = ratios.sort * rasterization_s
    rasterize_s = ratios.render * rasterization_s

    return SceneBreakdown(
        scene=scene,
        deformation_s=deformation_s,
        preprocess_s=preprocess_s,
        sort_s=sort_s,
        rasterize_s=rasterize_s,
        ratios=ratios,
    )


def write_csv_summary(out_csv: str, rows: List[SceneBreakdown]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scene",
                "deformation_s",
                "preprocess_s",
                "sort_s",
                "rasterize_s",
                "total_s",
                "ratio_preprocess",
                "ratio_sort",
                "ratio_render",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.scene,
                    r.deformation_s,
                    r.preprocess_s,
                    r.sort_s,
                    r.rasterize_s,
                    r.total_s,
                    r.ratios.preprocess,
                    r.ratios.sort,
                    r.ratios.render,
                ]
            )


def plot_breakdown(out_png: str, rows: List[SceneBreakdown], unit: str = "ms") -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (e.g. `pip install matplotlib`)."
        ) from e

    if unit == "ms":
        scale = 1e3
        y_label = "Time (ms)"
    elif unit == "s":
        scale = 1.0
        y_label = "Time (s)"
    else:
        raise ValueError(f"Unsupported unit: {unit!r} (use 'ms' or 's')")

    scenes = [r.scene for r in rows]
    deform = [r.deformation_s * scale for r in rows]
    preprocess = [r.preprocess_s * scale for r in rows]
    sort = [r.sort_s * scale for r in rows]
    raster = [r.rasterize_s * scale for r in rows]

    x = list(range(len(rows)))
    fig_w = max(10.0, 0.9 * len(rows))
    fig_h = 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    b1 = ax.bar(x, deform, label="Deformation", color="#4C78A8")
    b2 = ax.bar(x, preprocess, bottom=deform, label="Preprocess (preprocessCUDA)", color="#F58518")
    bottom2 = [a + b for a, b in zip(deform, preprocess)]
    b3 = ax.bar(x, sort, bottom=bottom2, label="Sorting (other kernels)", color="#54A24B")
    bottom3 = [a + b for a, b in zip(bottom2, sort)]
    b4 = ax.bar(x, raster, bottom=bottom3, label="Rasterize (renderCUDA)", color="#E45756")

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=30, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title("Latency breakdown per scene")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    # annotate totals (small, but handy)
    totals = [r.total_s * scale for r in rows]
    for i, t in enumerate(totals):
        ax.text(i, t, f"{t:.1f}", ha="center", va="bottom", fontsize=9)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _default_out_png(base_dir: str) -> str:
    return os.path.join(base_dir, "latency_breakdown.png")


def _default_out_csv(base_dir: str) -> str:
    return os.path.join(base_dir, "latency_breakdown.csv")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze DyNeRF latency breakdown per scene.")
    parser.add_argument(
        "--dynerf_dir",
        default=os.path.join(_repo_root(), "output", "dynerf"),
        help="Directory containing scene folders (default: <repo>/output/dynerf)",
    )
    parser.add_argument(
        "--statistics_relpath",
        default=os.path.join("test", "ours_14000", "statistics.txt"),
        help="Path to statistics.txt relative to each scene dir",
    )
    parser.add_argument("--out_png", default=None, help="Output png path (default: <dynerf_dir>/latency_breakdown.png)")
    parser.add_argument("--out_csv", default=None, help="Output csv path (default: <dynerf_dir>/latency_breakdown.csv)")
    parser.add_argument("--unit", choices=["ms", "s"], default="ms", help="Plot y-axis unit")
    parser.add_argument("--only_scene", default=None, help="Only analyze this scene name (optional)")
    args = parser.parse_args(argv)

    dynerf_dir = os.path.abspath(args.dynerf_dir)
    out_png = os.path.abspath(args.out_png) if args.out_png else _default_out_png(dynerf_dir)
    out_csv = os.path.abspath(args.out_csv) if args.out_csv else _default_out_csv(dynerf_dir)

    scene_dirs = _list_scene_dirs(dynerf_dir)
    if args.only_scene:
        scene_dirs = [d for d in scene_dirs if os.path.basename(d) == args.only_scene]

    rows: List[SceneBreakdown] = []
    skipped: List[Tuple[str, str]] = []
    for scene_dir in scene_dirs:
        scene = os.path.basename(scene_dir.rstrip("/"))
        csv_path = os.path.join(scene_dir, "analysis_per_kernel.csv")
        stat_path = os.path.join(scene_dir, args.statistics_relpath)

        if not os.path.isfile(stat_path):
            skipped.append((scene, f"missing statistics: {stat_path}"))
            continue
        if not os.path.isfile(csv_path):
            skipped.append((scene, f"missing analysis_per_kernel.csv: {csv_path}"))
            continue

        try:
            rows.append(compute_scene_breakdown(scene_dir, args.statistics_relpath))
        except Exception as e:
            skipped.append((scene, f"failed to parse: {e}"))

    if not rows:
        print("No scenes parsed successfully.", file=sys.stderr)
        for scene, reason in skipped:
            print(f"  - {scene}: {reason}", file=sys.stderr)
        return 2

    # stable order by scene name
    rows = sorted(rows, key=lambda r: r.scene)

    write_csv_summary(out_csv, rows)
    plot_breakdown(out_png, rows, unit=args.unit)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    if skipped:
        print("\nSkipped scenes:")
        for scene, reason in skipped:
            print(f"  - {scene}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
