"""Sample a Self-Forcing-compatible checkpoint and retain endpoint latents."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch

from .checkpoint import resolve_checkpoint
from .io import (
    canonical_hash,
    read_jsonl,
    sha256_file,
    validate_cached_metadata,
    validate_jobs,
    write_jsonl,
)
from .video import load_latent, save_latent, save_video


ROOT = Path(__file__).resolve().parents[1]


def _load_checkpoint(path: Path, state_key: str) -> tuple[dict[str, torch.Tensor], str]:
    load_kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": True,
    }
    try:
        obj = torch.load(path, mmap=True, **load_kwargs)
    except TypeError:
        obj = torch.load(path, **load_kwargs)
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint must contain a state dict, got {type(obj)}")
    selected = state_key
    if state_key == "auto":
        if "generator_ema" in obj:
            selected = "generator_ema"
        elif "generator" in obj:
            selected = "generator"
        else:
            selected = "root"
    state = obj if selected == "root" else obj.get(selected)
    if not isinstance(state, dict):
        raise KeyError(f"checkpoint has no state dict named {selected!r}")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        elif key.startswith("_fsdp_wrapped_module."):
            key = key.replace("_fsdp_wrapped_module.", "", 1)
        cleaned[key] = value
    return cleaned, selected


def _load_config(config_path: Path, model_root: Path):
    from omegaconf import OmegaConf

    default = OmegaConf.load(ROOT / "configs/default_config.yaml")
    config = OmegaConf.merge(default, OmegaConf.load(config_path))
    if not hasattr(config, "model_kwargs"):
        config.model_kwargs = {}
    config.model_kwargs.model_root = str(model_root.resolve())
    if not config.model_kwargs.get("model_name"):
        config.model_kwargs.model_name = "Wan2.1-T2V-1.3B"
    return config


def _implementation_fingerprint() -> str:
    paths = [
        Path(__file__),
        ROOT / "pipeline/causal_inference.py",
        ROOT / "utils/wan_wrapper.py",
        ROOT / "utils/scheduler.py",
        ROOT / "wan/modules/causal_model.py",
        ROOT / "wan/modules/model.py",
        ROOT / "wan/modules/vae.py",
    ]
    return canonical_hash(
        {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    )


@torch.inference_mode()
def sample_checkpoint(
    *,
    checkpoint: str,
    jobs_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path = "configs/self_forcing_dmd.yaml",
    model_root: str | Path = "wan_models",
    checkpoint_cache: str | Path | None = None,
    checkpoint_revision: str | None = None,
    state_key: str = "auto",
    method: str = "student_raw",
    device: str = "cuda",
    fps: int = 16,
    low_memory: bool | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Path:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    jobs = read_jsonl(jobs_path)
    validate_jobs(jobs)
    jobs = [row for index, row in enumerate(jobs) if index % num_shards == shard_index]
    if not jobs:
        raise ValueError(f"shard {shard_index}/{num_shards} has no jobs")

    print(f"student: resolving checkpoint for {len(jobs)} jobs", flush=True)
    resolved = resolve_checkpoint(
        checkpoint,
        cache_dir=checkpoint_cache,
        revision=checkpoint_revision,
    )
    config_path = Path(config_path).resolve()
    config = _load_config(config_path, Path(model_root))
    config_sha256 = sha256_file(config_path)
    default_config_sha256 = sha256_file(ROOT / "configs/default_config.yaml")
    from omegaconf import OmegaConf

    merged_config_sha256 = canonical_hash(
        OmegaConf.to_container(config, resolve=True)
    )
    print("student: hashing checkpoint for provenance", flush=True)
    checkpoint_sha256 = sha256_file(resolved.path)
    print(f"student: checkpoint sha256={checkpoint_sha256}", flush=True)
    implementation_sha256 = _implementation_fingerprint()
    from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
    from pipeline import CausalInferencePipeline

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("student sampling currently requires CUDA")
    print("student: loading Wan2.1-1.3B pipeline", flush=True)
    pipeline = CausalInferencePipeline(config, device=torch_device)
    print(f"student: loading state dict from {resolved.path}", flush=True)
    state, selected_state_key = _load_checkpoint(resolved.path, state_key)
    pipeline.generator.load_state_dict(state, strict=True)
    del state
    pipeline = pipeline.to(dtype=torch.bfloat16)
    use_low_memory = get_cuda_free_memory_gb(gpu) < 40 if low_memory is None else low_memory
    if use_low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=torch_device)
    pipeline.generator.to(device=torch_device)
    pipeline.vae.to(device=torch_device)

    shape = list(config.get("image_or_video_shape", [1, 21, 16, 60, 104]))
    if len(shape) != 5:
        raise ValueError(f"image_or_video_shape must have 5 dimensions, got {shape}")
    shape[0] = 1
    output_dir = Path(output_dir).resolve()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, job in enumerate(jobs):
        prompt_id = int(job["prompt_id"])
        seed = int(job["seed"])
        prompt = str(job["prompt"])
        stem = f"p{prompt_id:04d}_s{seed}"
        video_path = output_dir / "videos" / f"{stem}.mp4"
        latent_path = output_dir / "latents" / f"{stem}.pt"
        metadata = {
            "stage": "student_raw",
            "method": method,
            "prompt_id": prompt_id,
            "seed": seed,
            "checkpoint_spec": checkpoint,
            "resolved_checkpoint": str(resolved.path),
            "checkpoint_source": resolved.source,
            "checkpoint_state_key": selected_state_key,
            "checkpoint_bytes": resolved.path.stat().st_size,
            "checkpoint_mtime_ns": resolved.path.stat().st_mtime_ns,
            "checkpoint_sha256": checkpoint_sha256,
            "config": str(config_path.resolve()),
            "config_sha256": config_sha256,
            "default_config_sha256": default_config_sha256,
            "merged_config_sha256": merged_config_sha256,
            "implementation_sha256": implementation_sha256,
            "model_name": str(config.model_kwargs.model_name),
            "model_root": str(Path(model_root).resolve()),
            "latent_shape": shape,
            "low_memory": use_low_memory,
            "fps": fps,
            "torch_version": torch.__version__,
            "prompt_sha256": canonical_hash(prompt),
        }
        if video_path.is_file() and latent_path.is_file():
            _, cached_metadata = load_latent(latent_path)
            validate_cached_metadata(
                cached_metadata,
                metadata,
                artifact=latent_path,
            )
            if video_path.stat().st_size == 0:
                raise ValueError(f"cached video is empty: {video_path}")
        else:
            # The pipeline draws fresh noise between denoising steps, so seed
            # both the explicit generator and PyTorch's process-wide RNG.
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device=torch_device)
            generator.manual_seed(seed)
            noise = torch.randn(
                shape,
                generator=generator,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            video, latent = pipeline.inference(
                noise=noise,
                text_prompts=[prompt],
                return_latents=True,
                initial_latent=None,
                low_memory=use_low_memory,
            )
            save_video(video, video_path, fps=fps)
            save_latent(latent, latent_path, metadata=metadata)
            if hasattr(pipeline.vae.model, "clear_cache"):
                pipeline.vae.model.clear_cache()
        rows.append(
            {
                **metadata,
                "prompt": prompt,
                "video_path": str(video_path),
                "latent_path": str(latent_path),
            }
        )
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (len(jobs) - completed)
        print(
            f"student sample: {completed}/{len(jobs)} "
            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
            flush=True,
        )

    suffix = "" if num_shards == 1 else f".shard{shard_index:03d}-of-{num_shards:03d}"
    manifest = output_dir / f"manifest{suffix}.jsonl"
    write_jsonl(manifest, rows)
    run_config = output_dir / f"run_config{suffix}.json"
    run_config.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "resolved_checkpoint": str(resolved.path),
                "checkpoint_state_key": selected_state_key,
                "checkpoint_bytes": resolved.path.stat().st_size,
                "checkpoint_mtime_ns": resolved.path.stat().st_mtime_ns,
                "checkpoint_sha256": checkpoint_sha256,
                "jobs": str(Path(jobs_path).resolve()),
                "config": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "default_config_sha256": default_config_sha256,
                "merged_config_sha256": merged_config_sha256,
                "implementation_sha256": implementation_sha256,
                "model_root": str(Path(model_root).resolve()),
                "model_name": str(config.model_kwargs.model_name),
                "latent_shape": shape,
                "fps": fps,
                "torch_version": torch.__version__,
                "method": method,
                "low_memory": use_low_memory,
                "num_shards": num_shards,
                "shard_index": shard_index,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
