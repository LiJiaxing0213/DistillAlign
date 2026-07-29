"""Resolve local or Hugging Face checkpoint specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


HF_URL = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/(?:resolve|blob)/"
    r"(?P<revision>[^/]+)/(?P<filename>.+)$"
)


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: Path
    source: str
    repo_id: str | None = None
    filename: str | None = None
    revision: str | None = None


def parse_hf_spec(spec: str) -> tuple[str, str, str | None] | None:
    """Parse hf://org/repo/path or org/repo::path checkpoint syntax."""
    if spec.startswith("hf://"):
        parsed = urlparse(spec)
        components = [parsed.netloc, *parsed.path.strip("/").split("/")]
        components = [part for part in components if part]
        if len(components) < 3:
            raise ValueError(
                "HF checkpoint must use hf://ORG/REPO/PATH, for example "
                "hf://example/project/checkpoints/model.pt"
            )
        repo_id = "/".join(components[:2])
        filename = "/".join(components[2:])
        revision = None
        if parsed.query:
            for item in parsed.query.split("&"):
                key, _, value = item.partition("=")
                if key == "revision" and value:
                    revision = unquote(value)
        return repo_id, unquote(filename), revision
    if "::" in spec and not spec.startswith(("http://", "https://")):
        repo_id, filename = spec.split("::", 1)
        if repo_id.count("/") != 1 or not filename:
            raise ValueError("HF shorthand must use ORG/REPO::PATH")
        return repo_id, filename, None
    match = HF_URL.match(spec)
    if match:
        return (
            match.group("repo"),
            unquote(match.group("filename")),
            unquote(match.group("revision")),
        )
    return None


def resolve_checkpoint(
    spec: str | Path,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> ResolvedCheckpoint:
    value = str(spec)
    local = Path(value).expanduser()
    if local.is_file():
        return ResolvedCheckpoint(local.resolve(), source="local")
    if local.exists():
        raise ValueError(f"checkpoint must be a file, got: {local}")

    parsed = parse_hf_spec(value)
    if parsed is None:
        raise FileNotFoundError(
            f"checkpoint not found: {value}. Use a local file, "
            "hf://ORG/REPO/PATH, ORG/REPO::PATH, or a Hugging Face resolve URL."
        )
    repo_id, filename, parsed_revision = parsed
    effective_revision = revision or parsed_revision or "main"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for remote checkpoints") from exc
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=effective_revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    return ResolvedCheckpoint(
        Path(downloaded).resolve(),
        source="huggingface",
        repo_id=repo_id,
        filename=filename,
        revision=effective_revision,
    )
