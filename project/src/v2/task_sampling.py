"""Sample deterministic wetland-window tasks for the v2 workflow.

This module defines the real-data window sampler used by the v2 task library.
It deliberately stops short of graph construction or label generation: the
output is a clipped GeoDataFrame plus immutable metadata describing how that
sampled map window was produced. The default sampling mode is a physically
small, meter-scale, wetland-centered window so tasks resemble local mapping
zones rather than broad statewide crops.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import geopandas as gpd
from shapely.geometry import box

from load_data import GPKG_PATH, LAYER_NAME, load_gdf

SAMPLER_VERSION = "v2-axis-window-2"


@dataclass(frozen=True)
class TaskSamplingSpec:
    """Deterministic specification for one sampled wetland-window task.

    By default the window size is expressed in projected meters and is centered
    around a sampled wetland feature. Fraction-based statewide sampling remains
    available for coarse debugging, but it is no longer the default because it
    produces windows far larger than the intended v2 local-mapping regime.
    """

    seed: int
    window_width_m: float | None = 500.0
    window_height_m: float | None = 500.0
    window_width_fraction: float | None = None
    window_height_fraction: float | None = None
    anchor_mode: str = "wetland-feature"
    min_feature_count: int = 1
    min_total_wetland_area: float = 0.0
    max_attempts: int = 50
    sampler_version: str = SAMPLER_VERSION

    def __post_init__(self) -> None:
        metric_mode = (
            self.window_width_m is not None and self.window_height_m is not None
        )
        fraction_mode = (
            self.window_width_fraction is not None
            and self.window_height_fraction is not None
        )
        if metric_mode == fraction_mode:
            raise ValueError(
                "Specify exactly one window-size mode: either projected meter "
                "dimensions or statewide fractional dimensions."
            )
        if metric_mode:
            if self.window_width_m <= 0.0:
                raise ValueError(
                    f"window_width_m must be positive, got {self.window_width_m}"
                )
            if self.window_height_m <= 0.0:
                raise ValueError(
                    f"window_height_m must be positive, got {self.window_height_m}"
                )
        if fraction_mode:
            if not 0.0 < self.window_width_fraction <= 1.0:
                raise ValueError(
                    "window_width_fraction must be in (0, 1], "
                    f"got {self.window_width_fraction}"
                )
            if not 0.0 < self.window_height_fraction <= 1.0:
                raise ValueError(
                    "window_height_fraction must be in (0, 1], "
                    f"got {self.window_height_fraction}"
                )
        if self.anchor_mode not in {"wetland-feature", "statewide-uniform"}:
            raise ValueError(
                "anchor_mode must be 'wetland-feature' or 'statewide-uniform', "
                f"got {self.anchor_mode}"
            )
        if self.min_feature_count < 1:
            raise ValueError(
                f"min_feature_count must be at least 1, got {self.min_feature_count}"
            )
        if self.min_total_wetland_area < 0.0:
            raise ValueError(
                "min_total_wetland_area must be non-negative, "
                f"got {self.min_total_wetland_area}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_window_size(
    total_bounds: tuple[float, float, float, float],
    spec: TaskSamplingSpec,
) -> tuple[float, float, str]:
    minx, miny, maxx, maxy = total_bounds
    total_width = maxx - minx
    total_height = maxy - miny

    if spec.window_width_m is not None and spec.window_height_m is not None:
        window_width = float(spec.window_width_m)
        window_height = float(spec.window_height_m)
        size_mode = "meters"
    else:
        window_width = total_width * float(spec.window_width_fraction)
        window_height = total_height * float(spec.window_height_fraction)
        size_mode = "fractions"

    if window_width > total_width or window_height > total_height:
        raise ValueError(
            "Requested window size exceeds the statewide dataset bounds: "
            f"window=({window_width}, {window_height}), bounds={total_bounds}"
        )

    return window_width, window_height, size_mode


def _center_window_on_point(
    center_x: float,
    center_y: float,
    window_width: float,
    window_height: float,
    total_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = total_bounds
    start_x = min(max(center_x - window_width / 2.0, minx), maxx - window_width)
    start_y = min(max(center_y - window_height / 2.0, miny), maxy - window_height)
    return (
        float(start_x),
        float(start_y),
        float(start_x + window_width),
        float(start_y + window_height),
    )


def _sample_window_bounds(
    total_bounds: tuple[float, float, float, float],
    spec: TaskSamplingSpec,
    rng: random.Random,
    source_gdf: gpd.GeoDataFrame,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    minx, miny, maxx, maxy = total_bounds
    total_width = maxx - minx
    total_height = maxy - miny
    if total_width <= 0.0 or total_height <= 0.0:
        raise ValueError(
            "Source GeoDataFrame bounds must span positive width and height, "
            f"got {total_bounds}"
        )

    window_width, window_height, size_mode = _resolve_window_size(total_bounds, spec)

    if spec.anchor_mode == "statewide-uniform":
        x_span = max(total_width - window_width, 0.0)
        y_span = max(total_height - window_height, 0.0)
        start_x = minx if x_span == 0.0 else minx + rng.random() * x_span
        start_y = miny if y_span == 0.0 else miny + rng.random() * y_span
        return (
            (
                float(start_x),
                float(start_y),
                float(start_x + window_width),
                float(start_y + window_height),
            ),
            {
                "mode": spec.anchor_mode,
                "window_size_mode": size_mode,
                "feature_source_index": None,
                "feature_point": None,
            },
        )

    feature_position = rng.randrange(len(source_gdf))
    feature_source_index = source_gdf.index[feature_position]
    feature_point = source_gdf.geometry.iloc[feature_position].representative_point()
    bounds = _center_window_on_point(
        float(feature_point.x),
        float(feature_point.y),
        window_width,
        window_height,
        total_bounds,
    )
    return (
        bounds,
        {
            "mode": spec.anchor_mode,
            "window_size_mode": size_mode,
            "feature_source_index": str(feature_source_index),
            "feature_point": {
                "x": float(feature_point.x),
                "y": float(feature_point.y),
            },
        },
    )


def _clean_clipped_gdf(clipped: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = clipped.loc[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    cleaned["_source_index"] = cleaned.index.astype(str)
    return cleaned.reset_index(drop=True)


def summarize_window(
    clipped: gpd.GeoDataFrame,
    window_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    minx, miny, maxx, maxy = window_bounds
    window_area = float((maxx - minx) * (maxy - miny))
    total_wetland_area = float(clipped.geometry.area.sum()) if not clipped.empty else 0.0
    return {
        "feature_count": int(len(clipped)),
        "window_area": window_area,
        "total_wetland_area": total_wetland_area,
        "wetland_area_fraction": (
            float(total_wetland_area / window_area) if window_area > 0.0 else 0.0
        ),
    }


def window_meets_minimum_content(
    summary: dict[str, Any], spec: TaskSamplingSpec
) -> bool:
    return (
        int(summary["feature_count"]) >= spec.min_feature_count
        and float(summary["total_wetland_area"]) > spec.min_total_wetland_area
    )


def sample_wetland_window(
    spec: TaskSamplingSpec,
    gdf: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Return a reproducibly sampled wetland window and its metadata.

    The minimum-content rule is:
    - at least `min_feature_count` clipped wetland features remain after window
      clipping, and
    - total clipped wetland area is strictly greater than
      `min_total_wetland_area`.
    """

    source_gdf = load_gdf() if gdf is None else gdf
    total_bounds = tuple(float(value) for value in source_gdf.total_bounds)
    rng = random.Random(spec.seed)

    for attempt in range(spec.max_attempts):
        bounds, anchor_metadata = _sample_window_bounds(total_bounds, spec, rng, source_gdf)
        window_geom = box(*bounds)
        clipped = source_gdf.clip(window_geom)
        clipped = _clean_clipped_gdf(clipped)
        summary = summarize_window(clipped, bounds)
        if not window_meets_minimum_content(summary, spec):
            continue

        metadata = {
            "schema_version": 1,
            "sampler_version": spec.sampler_version,
            "source_dataset": {
                "path": str(GPKG_PATH),
                "layer": LAYER_NAME,
                "crs": (
                    source_gdf.crs.to_string() if source_gdf.crs is not None else None
                ),
                "feature_count": int(len(source_gdf)),
                "total_bounds": list(total_bounds),
            },
            "sampling": {
                **spec.to_payload(),
                "attempt_index": attempt,
                "window_size_mode": anchor_metadata["window_size_mode"],
            },
            "anchor": anchor_metadata,
            "window": {
                "bounds": list(bounds),
                "width": float(bounds[2] - bounds[0]),
                "height": float(bounds[3] - bounds[1]),
            },
            "summary": summary,
        }
        return clipped, metadata

    raise RuntimeError(
        "Unable to sample a wetland window that satisfies the minimum-content "
        f"rule after {spec.max_attempts} attempts."
    )


def save_window_debug_plot(
    window_gdf: gpd.GeoDataFrame,
    metadata: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a lightweight debug image for one sampled wetland window."""

    import matplotlib.pyplot as plt

    total_bounds = tuple(metadata["source_dataset"]["total_bounds"])
    window_bounds = tuple(metadata["window"]["bounds"])

    overview_gdf = gpd.GeoDataFrame(
        {"kind": ["dataset", "window"]},
        geometry=[box(*total_bounds), box(*window_bounds)],
        crs=window_gdf.crs,
    )
    window_outline_gdf = gpd.GeoDataFrame(
        {"kind": ["window"]},
        geometry=[box(*window_bounds)],
        crs=window_gdf.crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    overview_gdf.iloc[[0]].boundary.plot(ax=axes[0], color="0.7", linewidth=1.0)
    overview_gdf.iloc[[1]].boundary.plot(ax=axes[0], color="crimson", linewidth=2.0)
    axes[0].set_title("Dataset Extent and Sampled Window")
    axes[0].set_aspect("equal")

    if not window_gdf.empty:
        window_gdf.plot(ax=axes[1], color="#2a9d8f", edgecolor="#1f5c56", linewidth=0.2)
    window_outline_gdf.boundary.plot(ax=axes[1], color="crimson", linewidth=1.5)
    axes[1].set_title("Clipped Wetlands Inside Sampled Window")
    axes[1].set_aspect("equal")

    summary = metadata["summary"]
    sampling = metadata["sampling"]
    fig.suptitle(
        "seed={seed} attempt={attempt} features={features} wetland_area={area:.3f}".format(
            seed=sampling["seed"],
            attempt=sampling["attempt_index"],
            features=summary["feature_count"],
            area=summary["total_wetland_area"],
        )
    )

    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve or create an immutable cached v2 wetland-window task "
            "from a deterministic sampling specification."
        )
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        required=False, 
        help="Deterministic sampling seed.")
    parser.add_argument(
        "--window-size-m",
        type=float,
        default=None,
        help="Square window size in projected meters. Applies to both width and height.",
    )
    parser.add_argument(
        "--window-width-m",
        type=float,
        default=500.0,
        help="Window width in projected meters. Default is a local 500 m slice.",
    )
    parser.add_argument(
        "--window-height-m",
        type=float,
        default=500.0,
        help="Window height in projected meters. Default is a local 500 m slice.",
    )
    parser.add_argument(
        "--window-width-fraction",
        type=float,
        default=None,
        help="Optional fraction of statewide width to include in the sampled window.",
    )
    parser.add_argument(
        "--window-height-fraction",
        type=float,
        default=None,
        help="Optional fraction of statewide height to include in the sampled window.",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=("wetland-feature", "statewide-uniform"),
        default="wetland-feature",
        help=(
            "How to place each candidate window. 'wetland-feature' centers it on a "
            "sampled wetland geometry; 'statewide-uniform' samples from the whole "
            "dataset extent."
        ),
    )
    parser.add_argument(
        "--min-feature-count",
        type=int,
        default=1,
        help="Minimum non-empty clipped wetland features required before accepting a window.",
    )
    parser.add_argument(
        "--min-total-wetland-area",
        type=float,
        default=0.0,
        help="Minimum total clipped wetland area required before accepting a window.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Maximum resampling attempts before failing.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional override for the v2 task cache root.",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        default=None,
        help=(
            "Optional PNG path for a debug view showing the sampled window "
            "within the statewide extent and the clipped wetland features. "
            "Defaults to <task_dir>/debug_window.png when the flag is given "
            "without a value."
        ),
        nargs="?",
        const=Path("debug_window.png"),
    )
    return parser


def _spec_from_args(args: argparse.Namespace) -> TaskSamplingSpec:
    if args.window_size_m is not None:
        window_width_m = args.window_size_m
        window_height_m = args.window_size_m
    else:
        window_width_m = args.window_width_m
        window_height_m = args.window_height_m

    use_fractional_windows = (
        args.window_width_fraction is not None or args.window_height_fraction is not None
    )
    if use_fractional_windows:
        if args.window_width_fraction is None or args.window_height_fraction is None:
            raise ValueError(
                "Provide both --window-width-fraction and --window-height-fraction."
            )
        window_width_m = None
        window_height_m = None

    return TaskSamplingSpec(
        seed=args.seed,
        window_width_m=window_width_m,
        window_height_m=window_height_m,
        window_width_fraction=args.window_width_fraction,
        window_height_fraction=args.window_height_fraction,
        anchor_mode=args.anchor_mode,
        min_feature_count=args.min_feature_count,
        min_total_wetland_area=args.min_total_wetland_area,
        max_attempts=args.max_attempts,
    )


def main(argv: list[str] | None = None) -> int:
    """Create or reuse one cached v2 task from CLI arguments."""

    from v2.task_cache import TASK_CACHE_ROOT, resolve_or_create_task, task_dir, task_id_for_spec

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    # Check seed, set seed to random if not specified
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    spec = _spec_from_args(args)
    cache_root = args.cache_root if args.cache_root is not None else TASK_CACHE_ROOT
    task_id = task_id_for_spec(spec)
    sampled_task_dir = task_dir(task_id, cache_root=cache_root)
    reused = (sampled_task_dir / "metadata.json").exists() and (
        sampled_task_dir / "window.geojson"
    ).exists()

    cached_task = resolve_or_create_task(spec, cache_root=cache_root)
    debug_image_path = None
    if args.debug_image is not None:
        debug_image_path = args.debug_image
        if debug_image_path == Path("debug_window.png"):
            debug_image_path = cached_task.task_dir / debug_image_path
        debug_image_path = save_window_debug_plot(
            cached_task.window_gdf,
            cached_task.metadata,
            debug_image_path,
        )

    print(
        json.dumps(
            {
                "status": "reused" if reused else "created",
                "task_id": cached_task.task_id,
                "task_dir": str(cached_task.task_dir),
                "window_geojson_path": str(cached_task.window_geojson_path),
                "metadata_path": str(cached_task.metadata_path),
                "manifest_path": str(cache_root / "manifest.json"),
                "sampling": cached_task.metadata["sampling"],
                "window": cached_task.metadata["window"],
                "summary": cached_task.metadata["summary"],
                "debug_image_path": str(debug_image_path) if debug_image_path else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SAMPLER_VERSION",
    "TaskSamplingSpec",
    "main",
    "save_window_debug_plot",
    "sample_wetland_window",
    "summarize_window",
    "window_meets_minimum_content",
]


if __name__ == "__main__":
    raise SystemExit(main())
