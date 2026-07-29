"""Video and latent serialization helpers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

# Latent metadata stores torch.__version__ (a TorchVersion object); PyTorch >=2.6
# rejects it under the weights_only=True default unless explicitly allowlisted.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


def save_video(video: torch.Tensor, path: str | Path, *, fps: int = 16) -> None:
    """Save [B,T,C,H,W] or [T,C,H,W] float video in [0,1]."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to save videos") from exc
    if video.ndim == 5:
        if len(video) != 1:
            raise ValueError("save_video expects batch size 1")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"expected [T,C,H,W], got {tuple(video.shape)}")
    frames = (
        video.detach()
        .float()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.shape[1] % 2 or frames.shape[2] % 2:
        raise ValueError("H.264 yuv420p output requires even frame dimensions")
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = int(frames.shape[2])
            stream.height = int(frames.shape[1])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18"}
            for array in frames:
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(array), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"video encoder produced an empty file: {temporary}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_latent(
    latent: torch.Tensor,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent": latent.detach().cpu().float(),
            "metadata": metadata,
        },
        path,
    )


def load_latent(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        latent = obj
        metadata: dict[str, Any] = {}
    elif isinstance(obj, dict):
        latent = obj.get("latent")
        if latent is None:
            latent = obj.get("__endpoint_latent__")
        if latent is None:
            raise KeyError(f"latent file has no latent tensor: {path}")
        metadata = dict(obj.get("metadata") or obj.get("__meta__") or {})
    else:
        raise TypeError(f"unsupported latent file type in {path}: {type(obj)}")
    if latent.ndim != 5:
        raise ValueError(f"expected latent [B,T,C,H,W], got {tuple(latent.shape)}")
    return latent.float(), metadata
