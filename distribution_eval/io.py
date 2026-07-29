"""Manifest and artifact I/O shared by the evaluation stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


JOB_FIELDS = ("prompt_id", "seed", "prompt")
SAMPLE_FIELDS = (*JOB_FIELDS, "method", "video_path")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["prompt_id"]), int(row["seed"])


def validate_jobs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("job manifest is empty")
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        missing = [field for field in JOB_FIELDS if field not in row]
        if missing:
            raise ValueError(f"job row {index} is missing {missing}")
        key = sample_key(row)
        if key in seen:
            raise ValueError(f"duplicate prompt/seed key in jobs: {key}")
        seen.add(key)
        if not str(row["prompt"]).strip():
            raise ValueError(f"job row {index} has an empty prompt")


def validate_sample_manifest(
    rows: list[dict[str, Any]],
    *,
    require_files: bool = True,
    require_latents: bool = False,
) -> None:
    if not rows:
        raise ValueError("sample manifest is empty")
    seen: set[tuple[str, int, int]] = set()
    for index, row in enumerate(rows):
        missing = [field for field in SAMPLE_FIELDS if field not in row]
        if require_latents and "latent_path" not in row:
            missing.append("latent_path")
        if missing:
            raise ValueError(f"sample row {index} is missing {missing}")
        key = (str(row["method"]), *sample_key(row))
        if key in seen:
            raise ValueError(f"duplicate method/prompt/seed key: {key}")
        seen.add(key)
        if require_files:
            fields = ["video_path"]
            if require_latents:
                fields.append("latent_path")
            for field in fields:
                path = Path(str(row[field]))
                if not path.is_file():
                    raise FileNotFoundError(f"sample row {index}: missing {field}: {path}")
                if path.stat().st_size == 0:
                    raise ValueError(f"sample row {index}: empty {field}: {path}")


def validate_cached_metadata(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    artifact: str | Path,
) -> None:
    """Reject a cached artifact whose provenance differs from this run."""
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"refusing to reuse incompatible cached artifact {artifact}: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )


def ensure_aligned(
    teacher_rows: list[dict[str, Any]],
    student_rows: list[dict[str, Any]],
) -> None:
    teacher = {sample_key(row) for row in teacher_rows}
    student = {sample_key(row) for row in student_rows}
    if teacher != student:
        missing_student = sorted(teacher - student)[:10]
        missing_teacher = sorted(student - teacher)[:10]
        raise ValueError(
            "teacher/student prompt-seed sets differ: "
            f"missing_student={missing_student}, missing_teacher={missing_teacher}"
        )
