"""PRDC precision and coverage on normalized video features."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be a 2D array, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("features contain a zero-norm row")
    return features / norms


def pairwise_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1, keepdims=True).T
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def kth_neighbor_radii(features: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(features) <= k:
        raise ValueError(f"k={k} requires at least {k + 1} reference samples")
    distances = pairwise_l2(features, features)
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, kth=k - 1, axis=1)[:, k - 1]


def precision_coverage(
    teacher_features: np.ndarray,
    student_features: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, float | int]:
    """Compute teacher-normalized PRDC precision and coverage.

    Precision is the fraction of student samples inside at least one teacher
    k-nearest-neighbor ball. Coverage is the fraction of teacher samples whose
    nearest student sample falls inside that teacher sample's k-NN radius.
    """
    teacher = normalize(teacher_features)
    student = normalize(student_features)
    if teacher.shape[1] != student.shape[1]:
        raise ValueError(
            f"feature dimensions differ: teacher={teacher.shape}, student={student.shape}"
        )
    radii = kth_neighbor_radii(teacher, k)
    distances = pairwise_l2(teacher, student)
    inside_teacher_support = distances <= radii[:, None]
    return {
        "precision": float(inside_teacher_support.any(axis=0).mean()),
        "coverage": float((distances.min(axis=1) <= radii).mean()),
        "k": int(k),
        "teacher_samples": int(len(teacher)),
        "student_samples": int(len(student)),
    }


def _metadata_from_npz(cache: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "metadata" not in cache.files:
        raise ValueError(f"feature cache is missing metadata: {path}")
    value = cache["metadata"]
    if getattr(value, "shape", None) == ():
        value = value.item()
    metadata = json.loads(str(value))
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid metadata object in {path}")
    return metadata


def load_feature_cache(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        if "features" not in cache.files:
            raise ValueError(f"feature cache is missing features: {path}")
        features = cache["features"].astype(np.float64)
        metadata = _metadata_from_npz(cache, path)
    if int(metadata.get("row_count", -1)) != len(features):
        raise ValueError(
            f"feature row count does not match metadata in {path}: "
            f"{len(features)} vs {metadata.get('row_count')}"
        )
    return features, metadata


def ensure_compatible_feature_protocols(
    teacher_metadata: dict[str, Any],
    student_metadata: dict[str, Any],
) -> str:
    teacher_hash = teacher_metadata.get("protocol_sha256")
    student_hash = student_metadata.get("protocol_sha256")
    if not teacher_hash or not student_hash:
        raise ValueError("both feature caches must contain protocol_sha256")
    if teacher_hash != student_hash:
        raise ValueError(
            "teacher and student features were not produced by the same extractor protocol: "
            f"teacher={teacher_hash}, student={student_hash}"
        )
    teacher_keys = teacher_metadata.get("sample_keys_sha256")
    student_keys = student_metadata.get("sample_keys_sha256")
    if not teacher_keys or not student_keys:
        raise ValueError("both feature caches must contain sample_keys_sha256")
    if teacher_keys != student_keys:
        raise ValueError(
            "teacher and student feature caches use different prompt/seed sets: "
            f"teacher={teacher_keys}, student={student_keys}"
        )
    return str(teacher_hash)


def compute_from_caches(
    teacher_cache: str | Path,
    student_cache: str | Path,
    *,
    k: int = 5,
) -> dict[str, Any]:
    teacher_features, teacher_metadata = load_feature_cache(teacher_cache)
    student_features, student_metadata = load_feature_cache(student_cache)
    protocol_hash = ensure_compatible_feature_protocols(
        teacher_metadata, student_metadata
    )
    metrics = precision_coverage(teacher_features, student_features, k=k)
    return {
        "schema_version": 1,
        "feature_encoder": teacher_metadata.get("encoder"),
        "feature_model": teacher_metadata.get("model_id"),
        "feature_protocol_sha256": protocol_hash,
        "teacher_manifest_sha256": teacher_metadata.get("manifest_sha256"),
        "student_manifest_sha256": student_metadata.get("manifest_sha256"),
        **metrics,
    }


def write_metrics(
    result: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{json_path.name}.", dir=json_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, json_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)
