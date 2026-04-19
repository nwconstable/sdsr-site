"""Render detailed debug views for one cached v2 task.

This module is a read-only inspection tool for cached v2 wetland-window tasks.
It loads an existing task artifact, renders boundary-focused geometry views,
and produces categorical attribute maps for selected wetland fields without
modifying the immutable task cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt

from v2.task_cache import TASK_CACHE_ROOT, CachedTask, load_cached_task

DEFAULT_RENDER_ROOT = Path(__file__).resolve().parents[2] / "results" / "v2" / "task_debug"
ATTRIBUTE_SPECS = (
    ("COW_CLASS1", "cow_class1", "cow_class1.png"),
    ("SPCC_DESC", "spcc_desc", "spcc_desc.png"),
)


def _resolve_attribute_name(window_gdf: gpd.GeoDataFrame, target_name: str) -> str | None:
    target_lower = target_name.lower()
    for column_name in window_gdf.columns:
        if column_name.lower() == target_lower:
            return column_name
    return None


def _resolve_cached_task(
    task_id: str | None,
    task_dir: Path | None,
    cache_root: Path,
) -> CachedTask:
    if (task_id is None) == (task_dir is None):
        raise ValueError("Provide exactly one of --task-id or --task-dir.")

    if task_id is not None:
        return load_cached_task(task_id, cache_root=cache_root)

    resolved_task_dir = task_dir.resolve()
    inferred_task_id = resolved_task_dir.name
    inferred_cache_root = resolved_task_dir.parent
    return load_cached_task(inferred_task_id, cache_root=inferred_cache_root)


def _task_render_dir(render_root: Path, task: CachedTask) -> Path:
    return render_root / task.task_id


def render_boundary_overview(task: CachedTask, output_path: Path) -> Path:
    """Render a detailed boundary-focused geometry view for one cached task."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    task.window_gdf.boundary.plot(ax=ax, color="#0b3c5d", linewidth=0.5)
    ax.set_title(f"{task.task_id} wetland boundaries")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def render_attribute_map(
    task: CachedTask,
    requested_label: str,
    fallback_label: str,
    output_path: Path,
) -> tuple[Path | None, dict[str, Any]]:
    """Render one categorical attribute map, or report a documented fallback."""

    column_name = _resolve_attribute_name(task.window_gdf, requested_label)
    if column_name is None:
        return None, {
            "requested_attribute": requested_label,
            "resolved_column": None,
            "status": "missing",
            "reason": f"Column matching {requested_label!r} not found in cached task.",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    value_counts = task.window_gdf[column_name].fillna("<missing>").astype(str).value_counts()
    top_values = value_counts.head(8).index.tolist()
    plot_gdf = task.window_gdf.copy()
    series = plot_gdf[column_name].fillna("<missing>").astype(str)
    plot_gdf["_plot_value"] = series.where(series.isin(top_values), other="<other>")

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    plot_gdf.plot(
        ax=ax,
        column="_plot_value",
        categorical=True,
        legend=True,
        linewidth=0.2,
        edgecolor="#1f1f1f",
        legend_kwds={"loc": "upper left", "fontsize": 7, "title": fallback_label},
    )
    plot_gdf.boundary.plot(ax=ax, color="#1f1f1f", linewidth=0.15)
    ax.set_title(f"{task.task_id} {fallback_label} distribution")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, {
        "requested_attribute": requested_label,
        "resolved_column": column_name,
        "status": "rendered",
        "category_count": int(series.nunique(dropna=False)),
        "top_categories": top_values,
    }


def render_task_debug_views(
    task: CachedTask,
    render_root: Path = DEFAULT_RENDER_ROOT,
) -> dict[str, Any]:
    """Render all detailed debug views for one cached task."""

    task_output_dir = _task_render_dir(render_root, task)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    boundary_path = render_boundary_overview(task, task_output_dir / "boundaries.png")
    render_summary: dict[str, Any] = {
        "task_id": task.task_id,
        "task_dir": str(task.task_dir),
        "render_dir": str(task_output_dir),
        "boundary_image": str(boundary_path),
        "attributes": {},
    }

    for requested_label, fallback_label, filename in ATTRIBUTE_SPECS:
        image_path, attribute_summary = render_attribute_map(
            task,
            requested_label=requested_label,
            fallback_label=fallback_label,
            output_path=task_output_dir / filename,
        )
        if image_path is not None:
            attribute_summary["image_path"] = str(image_path)
        render_summary["attributes"][requested_label] = attribute_summary

    summary_path = task_output_dir / "render_manifest.json"
    summary_path.write_text(json.dumps(render_summary, indent=2, sort_keys=True), encoding="utf-8")
    render_summary["render_manifest"] = str(summary_path)
    return render_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render detailed cached-task debug views for one v2 wetland-window task."
        )
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Cached task identifier such as task-abcdef1234567890.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=None,
        help="Explicit cached task directory containing metadata.json and window.geojson.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=TASK_CACHE_ROOT,
        help="Cache root used when resolving --task-id.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RENDER_ROOT,
        help="Directory under which task-specific debug render outputs are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for detailed cached-task rendering."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cached_task = _resolve_cached_task(
        task_id=args.task_id,
        task_dir=args.task_dir,
        cache_root=args.cache_root,
    )
    render_summary = render_task_debug_views(cached_task, render_root=args.output_root)
    print(json.dumps(render_summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ATTRIBUTE_SPECS",
    "DEFAULT_RENDER_ROOT",
    "main",
    "render_attribute_map",
    "render_boundary_overview",
    "render_task_debug_views",
]


if __name__ == "__main__":
    raise SystemExit(main())