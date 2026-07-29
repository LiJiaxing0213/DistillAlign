"""Strict V-JEPA2 feature extraction for distribution evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .io import canonical_hash, read_jsonl, sha256_file, validate_sample_manifest


DEFAULT_MODEL_ID = "facebook/vjepa2-vith-fpc64-256"
DEFAULT_MODEL_REVISION = "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5"


def select_uniform_frames(
    frames: torch.Tensor,
    *,
    num_frames: int = 8,
    clip_frames: int | None = 81,
) -> torch.Tensor:
    """Crop to a fixed prefix and uniformly select a fixed number of frames."""
    if frames.ndim != 4:
        raise ValueError(f"expected video [T,C,H,W], got {tuple(frames.shape)}")
    if clip_frames is not None:
        if clip_frames < 1:
            raise ValueError("clip_frames must be positive or None")
        frames = frames[:clip_frames]
    if len(frames) < num_frames:
        raise ValueError(
            f"video has only {len(frames)} frames after cropping; need {num_frames}"
        )
    indices = torch.linspace(0, len(frames) - 1, num_frames).round().long()
    return frames[indices]


def pool_vjepa_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Concatenate token mean and sample std, then L2-normalize.

    V-JEPA2 ViT-H has a 1280-dimensional hidden state. This pooling produces
    the 2560-dimensional representation used by the evaluation protocol.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens [B,N,D], got {tuple(tokens.shape)}")
    mean = tokens.float().mean(dim=1)
    std = tokens.float().std(dim=1, correction=1)
    pooled = torch.cat([mean, std], dim=-1)
    return torch.nn.functional.normalize(pooled, dim=-1)


def read_video(path: str | Path) -> torch.Tensor:
    try:
        from torchvision.io import read_video
        video, _, _ = read_video(
            str(path),
            pts_unit="sec",
            output_format="TCHW",
        )
        if video.numel() == 0:
            raise RuntimeError("empty video")
    except Exception:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError(
                "video decoding requires torchvision/PyAV or imageio"
            ) from exc
        array = iio.imread(path)
        if array.ndim == 3:
            array = array[None]
        video = torch.from_numpy(array).permute(0, 3, 1, 2)
    if video.numel() == 0:
        raise ValueError(f"decoded an empty video: {path}")
    if video.dtype != torch.uint8:
        if video.is_floating_point() and float(video.max()) <= 1.0:
            video = (video * 255.0).round()
        video = video.clamp(0, 255).to(torch.uint8)
    return video


class VJEPA2Embedder:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_MODEL_REVISION,
        cache_dir: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        try:
            import transformers
            from transformers import VJEPA2Model, VJEPA2VideoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "V-JEPA2 extraction requires a Transformers release with VJEPA2Model"
            ) from exc
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype: {dtype}")
        if str(device).startswith("cpu") and dtype != "float32":
            dtype = "float32"
        self.device = torch.device(device)
        self.dtype_name = dtype
        model_dtype = dtype_map[dtype]
        load_kwargs = {
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir else None,
        }
        self.processor = VJEPA2VideoProcessor.from_pretrained(model_id, **load_kwargs)
        try:
            self.model = VJEPA2Model.from_pretrained(
                model_id, dtype=model_dtype, **load_kwargs
            )
        except TypeError:
            self.model = VJEPA2Model.from_pretrained(
                model_id, torch_dtype=model_dtype, **load_kwargs
            )
        self.model = self.model.eval().to(self.device)
        self.model_id = model_id
        self.revision = revision
        self.commit_hash = getattr(self.model.config, "_commit_hash", None)
        self.transformers_version = transformers.__version__

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> np.ndarray:
        # Keep the exact TCHW uint8 input convention used for the released
        # paper features. The processor handles layout conversion and scaling.
        video = frames.to(torch.uint8)
        inputs = self.processor(video, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        if hasattr(self.model, "get_vision_features"):
            tokens = self.model.get_vision_features(**inputs)
        else:
            tokens = self.model(**inputs).last_hidden_state
        feature = pool_vjepa_tokens(tokens)
        return feature[0].cpu().numpy().astype(np.float32)


def _feature_metadata(
    *,
    manifest: Path,
    rows: list[dict[str, Any]],
    row_count: int,
    embedder: VJEPA2Embedder,
    num_frames: int,
    clip_frames: int | None,
) -> dict[str, Any]:
    implementation_sha256 = sha256_file(Path(__file__))
    protocol = {
        "schema_version": 1,
        "encoder": "vjepa2",
        "model_id": embedder.model_id,
        "model_revision": embedder.revision,
        "model_commit": embedder.commit_hash,
        "frame_selection": "linspace",
        "num_frames": int(num_frames),
        "clip_frames": int(clip_frames) if clip_frames is not None else None,
        "pooling": "token_mean_sample_std_concat_l2",
        "processor": type(embedder.processor).__name__,
        "transformers_version": embedder.transformers_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "inference_dtype": embedder.dtype_name,
        "implementation_sha256": implementation_sha256,
    }
    return {
        **protocol,
        "protocol_sha256": canonical_hash(protocol),
        "row_count": int(row_count),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "sample_keys_sha256": canonical_hash(
            sorted(
                (int(row["prompt_id"]), int(row["seed"]), str(row["prompt"]))
                for row in rows
            )
        ),
        "extractor_source_sha256": implementation_sha256,
        "torch_version": protocol["torch_version"],
        "numpy_version": protocol["numpy_version"],
        "inference_dtype": protocol["inference_dtype"],
    }


def extract_manifest(
    manifest: str | Path,
    output: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    cache_dir: str | Path | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    num_frames: int = 8,
    clip_frames: int | None = 81,
    log_every: int = 16,
) -> dict[str, Any]:
    manifest = Path(manifest)
    output = Path(output)
    rows = read_jsonl(manifest)
    validate_sample_manifest(rows, require_files=True)
    embedder = VJEPA2Embedder(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        dtype=dtype,
    )
    features: list[np.ndarray] = []
    row_keys: list[str] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        frames = select_uniform_frames(
            read_video(row["video_path"]),
            num_frames=num_frames,
            clip_frames=clip_frames,
        )
        features.append(embedder(frames))
        row_keys.append(
            json.dumps(
                {
                    "method": row["method"],
                    "prompt_id": int(row["prompt_id"]),
                    "seed": int(row["seed"]),
                },
                sort_keys=True,
            )
        )
        if index == 0 or (index + 1) % log_every == 0 or index + 1 == len(rows):
            completed = index + 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(rows) - completed)
            print(
                f"V-JEPA2: {completed}/{len(rows)} "
                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                flush=True,
            )
    feature_array = np.stack(features).astype(np.float32)
    metadata = _feature_metadata(
        manifest=manifest,
        rows=rows,
        row_count=len(rows),
        embedder=embedder,
        num_frames=num_frames,
        clip_frames=clip_frames,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".npz", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            features=feature_array,
            row_keys=np.asarray(row_keys, dtype=np.str_),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output} with shape {feature_array.shape}", flush=True)
    return metadata
