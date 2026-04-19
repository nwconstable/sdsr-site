"""Immutable cache helpers for sampled v2 wetland-window tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd

from v2.task_sampling import TaskSamplingSpec, sample_wetland_window

CACHE_SCHEMA_VERSION = 1
TASK_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "v2_tasks"
MANIFEST_PATH = TASK_CACHE_ROOT / "manifest.json"


@dataclass(frozen=True)
class CachedTask:
    task_id: str
    task_dir: Path
    window_geojson_path: Path
    metadata_path: Path
    window_gdf: gpd.GeoDataFrame
    metadata: dict[str, Any]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def spec_fingerprint(spec: TaskSamplingSpec) -> str:
    canonical = json.dumps(spec.to_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def task_id_for_spec(spec: TaskSamplingSpec) -> str:
    return f"task-{spec_fingerprint(spec)[:16]}"


def task_dir(task_id: str, cache_root: Path = TASK_CACHE_ROOT) -> Path:
    return cache_root / task_id


def load_manifest(cache_root: Path = TASK_CACHE_ROOT) -> dict[str, Any]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "tasks": {},
        }
    return _json_load(manifest_path)


def _save_manifest(manifest: dict[str, Any], cache_root: Path = TASK_CACHE_ROOT) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    _json_dump(cache_root / "manifest.json", manifest)


def _write_window_geojson(window_gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.write_text(window_gdf.to_json(drop_id=False), encoding="utf-8")


def _read_window_geojson(path: Path, crs: str | None) -> gpd.GeoDataFrame:
    payload = _json_load(path)
    window_gdf = gpd.GeoDataFrame.from_features(payload["features"])
    if crs is not None:
        window_gdf.set_crs(crs, inplace=True)
    return window_gdf


def load_cached_task(task_id: str, cache_root: Path = TASK_CACHE_ROOT) -> CachedTask:
    sampled_task_dir = task_dir(task_id, cache_root=cache_root)
    metadata_path = sampled_task_dir / "metadata.json"
    geojson_path = sampled_task_dir / "window.geojson"
    if not metadata_path.exists() or not geojson_path.exists():
        raise FileNotFoundError(
            f"Cached task {task_id} is missing expected artifacts in {sampled_task_dir}"
        )
    metadata = _json_load(metadata_path)
    window_gdf = _read_window_geojson(
        geojson_path, crs=metadata["source_dataset"].get("crs")
    )
    return CachedTask(
        task_id=task_id,
        task_dir=sampled_task_dir,
        window_geojson_path=geojson_path,
        metadata_path=metadata_path,
        window_gdf=window_gdf,
        metadata=metadata,
    )


def cache_sampled_task(
    spec: TaskSamplingSpec,
    window_gdf: gpd.GeoDataFrame,
    metadata: dict[str, Any],
    cache_root: Path = TASK_CACHE_ROOT,
) -> CachedTask:
    task_id = task_id_for_spec(spec)
    sampled_task_dir = task_dir(task_id, cache_root=cache_root)
    metadata_path = sampled_task_dir / "metadata.json"
    geojson_path = sampled_task_dir / "window.geojson"

    if metadata_path.exists() and geojson_path.exists():
        return load_cached_task(task_id, cache_root=cache_root)

    cache_root.mkdir(parents=True, exist_ok=True)
    sampled_task_dir.mkdir(parents=True, exist_ok=True)

    task_metadata = dict(metadata)
    task_metadata["task_id"] = task_id
    task_metadata["spec_fingerprint"] = spec_fingerprint(spec)
    task_metadata["artifacts"] = {
        "window_geojson": geojson_path.name,
        "metadata": metadata_path.name,
    }

    _write_window_geojson(window_gdf, geojson_path)
    _json_dump(metadata_path, task_metadata)

    manifest = load_manifest(cache_root=cache_root)
    manifest.setdefault("tasks", {})[task_id] = {
        "task_id": task_id,
        "spec_fingerprint": task_metadata["spec_fingerprint"],
        "task_dir": sampled_task_dir.name,
        "window_geojson": str(Path(sampled_task_dir.name) / geojson_path.name),
        "metadata": str(Path(sampled_task_dir.name) / metadata_path.name),
        "source_dataset": task_metadata["source_dataset"],
        "sampling": task_metadata["sampling"],
        "window": task_metadata["window"],
        "summary": task_metadata["summary"],
    }
    _save_manifest(manifest, cache_root=cache_root)
    return load_cached_task(task_id, cache_root=cache_root)


def resolve_or_create_task(
    spec: TaskSamplingSpec,
    gdf: gpd.GeoDataFrame | None = None,
    cache_root: Path = TASK_CACHE_ROOT,
) -> CachedTask:
    """Return the immutable cached task for *spec*, creating it if needed."""

    task_id = task_id_for_spec(spec)
    sampled_task_dir = task_dir(task_id, cache_root=cache_root)
    if (sampled_task_dir / "metadata.json").exists() and (
        sampled_task_dir / "window.geojson"
    ).exists():
        return load_cached_task(task_id, cache_root=cache_root)

    window_gdf, metadata = sample_wetland_window(spec, gdf=gdf)
    return cache_sampled_task(spec, window_gdf, metadata, cache_root=cache_root)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedTask",
    "MANIFEST_PATH",
    "TASK_CACHE_ROOT",
    "cache_sampled_task",
    "load_cached_task",
    "load_manifest",
    "resolve_or_create_task",
    "spec_fingerprint",
    "task_dir",
    "task_id_for_spec",
]
