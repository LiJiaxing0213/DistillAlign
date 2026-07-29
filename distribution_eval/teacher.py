"""Wan2.1 teacher sampling and teacher-normalized re-noise refinement."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .io import (
    canonical_hash,
    read_jsonl,
    validate_cached_metadata,
    validate_sample_manifest,
    write_jsonl,
)
from .video import load_latent, save_latent, save_video


FALLBACK_NEGATIVE_PROMPT = (
    "oversaturated, overexposed, static, blurry details, subtitles, painting, "
    "low quality, compression artifacts, malformed anatomy, cluttered background"
)


def refinement_schedule(strength: float, teacher_steps: int) -> tuple[int, np.ndarray]:
    if not 0.0 < strength <= 1.0:
        raise ValueError("re-noise strength must be in (0, 1]")
    if teacher_steps < 1:
        raise ValueError("teacher_steps must be positive")
    remaining_steps = max(1, int(round(teacher_steps * strength)))
    sigmas = np.linspace(strength, 0.0, remaining_steps + 1).copy()[:-1]
    return remaining_steps, sigmas


def _model_dir(model_root: Path, model_name: str) -> Path:
    path = (model_root / model_name).resolve()
    required = [
        path / "config.json",
        path / "Wan2.1_VAE.pth",
        path / "models_t5_umt5-xxl-enc-bf16.pth",
        path / "google/umt5-xxl/tokenizer.json",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete Wan model directory {path}; missing: {missing}"
        )
    if not list(path.glob("*.safetensors")):
        raise FileNotFoundError(
            f"Wan transformer weights are missing from {path}; run scripts/download_models.sh"
        )
    return path


def _model_fingerprint(model_dir: Path) -> str:
    files = [
        model_dir / "config.json",
        model_dir / "Wan2.1_VAE.pth",
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "google/umt5-xxl/tokenizer.json",
        *sorted(model_dir.glob("*.safetensors")),
        *sorted(model_dir.glob("*.index.json")),
    ]
    return canonical_hash(
        [
            {
                "path": str(path.relative_to(model_dir)),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ]
    )


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        root / "utils/wan_wrapper.py",
        root / "wan/utils/fm_solvers_unipc.py",
        root / "wan/modules/model.py",
        root / "wan/modules/vae.py",
    ]
    from .io import sha256_file

    return canonical_hash(
        {str(path.relative_to(root)): sha256_file(path) for path in paths}
    )


@torch.inference_mode()
def _encode_prompts(
    prompts: list[str],
    *,
    negative_prompt: str,
    model_name: str,
    model_root: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    from utils.wan_wrapper import WanTextEncoder

    encoder = WanTextEncoder(model_name=model_name, model_root=model_root)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for index, prompt in enumerate(dict.fromkeys(prompts)):
        cache[prompt] = {
            key: value.detach().cpu()
            for key, value in encoder(text_prompts=[prompt]).items()
        }
        print(f"teacher text encoding: {index + 1}/{len(set(prompts))}", flush=True)
    negative = {
        key: value.detach().cpu()
        for key, value in encoder(text_prompts=[negative_prompt]).items()
    }
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return cache, negative


class WanTeacherEngine:
    def __init__(
        self,
        *,
        model_name: str,
        model_root: Path,
        device: str,
        guidance_scale: float,
        teacher_steps: int,
        teacher_shift: float,
        prompt_cache: dict[str, dict[str, torch.Tensor]],
        negative_cache: dict[str, torch.Tensor],
    ) -> None:
        from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper

        self.device = torch.device(device)
        self.guidance_scale = float(guidance_scale)
        self.teacher_steps = int(teacher_steps)
        self.teacher_shift = float(teacher_shift)
        self.prompt_cache = prompt_cache
        self.negative_cache = negative_cache
        self.generator = WanDiffusionWrapper(
            model_name=model_name,
            model_root=model_root,
            timestep_shift=teacher_shift,
            is_causal=False,
        ).to(device=self.device, dtype=torch.bfloat16).eval()
        self.vae = WanVAEWrapper(
            model_name=model_name,
            model_root=model_root,
        ).to(device=self.device, dtype=torch.bfloat16).eval()

    def _condition(self, prompt: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        conditional = {
            key: value.to(self.device)
            for key, value in self.prompt_cache[prompt].items()
        }
        unconditional = {
            key: value.to(self.device)
            for key, value in self.negative_cache.items()
        }
        return conditional, unconditional

    @torch.inference_mode()
    def _solve(
        self,
        latent: torch.Tensor,
        *,
        prompt: str,
        sigmas: np.ndarray | None,
        steps: int,
        shift: float,
        progress_label: str,
    ) -> tuple[torch.Tensor, list[float]]:
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        # Match Wan2.1's reference sampler: scheduler state remains FP32 while
        # only the diffusion-network forward runs under BF16 autocast.
        latent = latent.to(device=self.device, dtype=torch.float32)
        conditional, unconditional = self._condition(prompt)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        if sigmas is None:
            scheduler.set_timesteps(steps, device=self.device, shift=shift)
        else:
            scheduler.set_timesteps(sigmas=sigmas, device=self.device, shift=1)
        used_timesteps: list[float] = []
        frame_count = latent.shape[1]
        solver_started = time.perf_counter()
        total_steps = len(scheduler.timesteps)
        for step_index, timestep_value in enumerate(scheduler.timesteps, start=1):
            used_timesteps.append(float(timestep_value.cpu()))
            timestep = timestep_value * torch.ones(
                [latent.shape[0], frame_count],
                device=self.device,
                dtype=timestep_value.dtype,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flow_conditional, _ = self.generator(latent, conditional, timestep)
                flow_unconditional, _ = self.generator(latent, unconditional, timestep)
            flow = flow_unconditional + self.guidance_scale * (
                flow_conditional - flow_unconditional
            )
            latent = scheduler.step(
                flow, timestep_value, latent, return_dict=False
            )[0]
            if step_index == 1 or step_index % 5 == 0 or step_index == total_steps:
                elapsed = time.perf_counter() - solver_started
                eta = elapsed / step_index * (total_steps - step_index)
                print(
                    f"teacher {progress_label}: {step_index}/{total_steps} "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )
        return latent, used_timesteps

    @torch.inference_mode()
    def sample_reference(
        self,
        *,
        prompt: str,
        seed: int,
        shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, list[float]]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return self._solve(
            noise,
            prompt=prompt,
            sigmas=None,
            steps=self.teacher_steps,
            shift=self.teacher_shift,
            progress_label="reference",
        )

    @torch.inference_mode()
    def refine(
        self,
        clean_latent: torch.Tensor,
        *,
        prompt: str,
        seed: int,
        strength: float,
        noise_seed_offset: int,
    ) -> tuple[torch.Tensor, list[float], int]:
        remaining_steps, sigmas = refinement_schedule(strength, self.teacher_steps)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + noise_seed_offset)
        noise = torch.randn(
            clean_latent.shape,
            generator=generator,
            dtype=torch.float32,
        )
        start = (1.0 - strength) * clean_latent.float() + strength * noise
        refined, timesteps = self._solve(
            start,
            prompt=prompt,
            sigmas=sigmas,
            steps=remaining_steps,
            shift=1,
            progress_label="refine",
        )
        return refined, timesteps, remaining_steps

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        video = self.vae.decode_to_pixel(latent.to(self.device, dtype=torch.bfloat16))
        return (video * 0.5 + 0.5).clamp(0, 1)


def teacher_normalize(
    *,
    raw_manifest: str | Path,
    output_dir: str | Path,
    model_root: str | Path = "wan_models",
    teacher_model_name: str = "Wan2.1-T2V-14B",
    negative_prompt: str = FALLBACK_NEGATIVE_PROMPT,
    strength: float = 0.9,
    teacher_steps: int = 25,
    teacher_shift: float = 8.0,
    guidance_scale: float = 5.0,
    noise_seed_offset: int = 0,
    device: str = "cuda",
    fps: int = 16,
    generate_refine: bool = True,
    generate_reference: bool = True,
) -> tuple[Path | None, Path | None]:
    if not generate_refine and not generate_reference:
        raise ValueError("nothing to do: refine and reference are both disabled")
    rows = read_jsonl(raw_manifest)
    validate_sample_manifest(rows, require_files=True, require_latents=True)
    refinement_schedule(strength, teacher_steps)
    model_root = Path(model_root)
    model_dir = _model_dir(model_root, teacher_model_name)
    model_fingerprint = _model_fingerprint(model_dir)
    implementation_fingerprint = _implementation_fingerprint()
    negative_prompt_sha256 = canonical_hash(negative_prompt)
    print(
        f"teacher: {len(rows)} jobs, model={model_dir}, strength={strength}",
        flush=True,
    )
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("Wan2.1-14B refinement currently requires CUDA")
    print("teacher: loading text encoder", flush=True)
    prompt_cache, negative_cache = _encode_prompts(
        [str(row["prompt"]) for row in rows],
        negative_prompt=negative_prompt,
        model_name=teacher_model_name,
        model_root=model_root,
        device=torch_device,
    )
    print("teacher: loading diffusion model and VAE", flush=True)
    engine = WanTeacherEngine(
        model_name=teacher_model_name,
        model_root=model_root,
        device=device,
        guidance_scale=guidance_scale,
        teacher_steps=teacher_steps,
        teacher_shift=teacher_shift,
        prompt_cache=prompt_cache,
        negative_cache=negative_cache,
    )

    output_dir = Path(output_dir).resolve()
    # Single-purpose invocations write directly into output_dir; the combined
    # mode keeps one subdirectory per artifact kind.
    if generate_refine and generate_reference:
        refine_root = output_dir / "student_refined"
        reference_root = output_dir / "teacher_reference"
    else:
        refine_root = output_dir
        reference_root = output_dir
    refined_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    strength_tag = int(round(strength * 100))
    teacher_tag = teacher_model_name.lower().replace(".", "p").replace("-", "_")
    refined_method = f"student_renoise{strength_tag:02d}_{teacher_tag}"
    teacher_method = f"{teacher_tag}_{teacher_steps}step"
    for index, row in enumerate(rows):
        prompt_id = int(row["prompt_id"])
        seed = int(row["seed"])
        prompt = str(row["prompt"])
        stem = f"p{prompt_id:04d}_s{seed}"
        clean_latent, _ = load_latent(row["latent_path"])

        if generate_refine:
            refined_video = refine_root / "videos" / f"{stem}.mp4"
            refined_latent = refine_root / "latents" / f"{stem}.pt"
            source_latent = Path(str(row["latent_path"])).resolve()
            remaining_steps, _ = refinement_schedule(strength, teacher_steps)
            expected_refined_metadata = {
                "stage": "teacher_normalized",
                "method": refined_method,
                "source_method": row["method"],
                "source_latent_path": str(source_latent),
                "source_latent_bytes": source_latent.stat().st_size,
                "source_latent_mtime_ns": source_latent.stat().st_mtime_ns,
                "prompt_id": prompt_id,
                "seed": seed,
                "prompt_sha256": canonical_hash(prompt),
                "renoise_strength": strength,
                "noise_seed": seed + noise_seed_offset,
                "teacher_model": teacher_model_name,
                "teacher_model_fingerprint_sha256": model_fingerprint,
                "teacher_steps_base": teacher_steps,
                "teacher_remaining_steps": remaining_steps,
                "teacher_refinement_shift": 1.0,
                "guidance_scale": guidance_scale,
                "negative_prompt_sha256": negative_prompt_sha256,
                "noise_seed_offset": noise_seed_offset,
                "implementation_sha256": implementation_fingerprint,
                "fps": fps,
                "torch_version": torch.__version__,
            }
            refined_metadata: dict[str, Any]
            if refined_video.is_file() and refined_latent.is_file():
                _, refined_metadata = load_latent(refined_latent)
                validate_cached_metadata(
                    refined_metadata,
                    expected_refined_metadata,
                    artifact=refined_latent,
                )
                if refined_video.stat().st_size == 0:
                    raise ValueError(f"cached video is empty: {refined_video}")
            else:
                refined, timesteps, remaining_steps = engine.refine(
                    clean_latent,
                    prompt=prompt,
                    seed=seed,
                    strength=strength,
                    noise_seed_offset=noise_seed_offset,
                )
                refined_metadata = {
                    **expected_refined_metadata,
                    "teacher_timesteps": timesteps,
                }
                save_video(engine.decode(refined), refined_video, fps=fps)
                save_latent(refined, refined_latent, metadata=refined_metadata)
            refined_rows.append(
                {
                    **refined_metadata,
                    "prompt": prompt,
                    "video_path": str(refined_video),
                    "latent_path": str(refined_latent),
                }
            )

        if generate_reference:
            teacher_video = reference_root / "videos" / f"{stem}.mp4"
            teacher_latent = reference_root / "latents" / f"{stem}.pt"
            expected_teacher_metadata = {
                "stage": "teacher_reference",
                "method": teacher_method,
                "prompt_id": prompt_id,
                "seed": seed,
                "prompt_sha256": canonical_hash(prompt),
                "latent_shape": list(clean_latent.shape),
                "teacher_model": teacher_model_name,
                "teacher_model_fingerprint_sha256": model_fingerprint,
                "teacher_steps": teacher_steps,
                "teacher_shift": teacher_shift,
                "guidance_scale": guidance_scale,
                "negative_prompt_sha256": negative_prompt_sha256,
                "implementation_sha256": implementation_fingerprint,
                "fps": fps,
                "torch_version": torch.__version__,
            }
            teacher_metadata: dict[str, Any]
            if teacher_video.is_file() and teacher_latent.is_file():
                _, teacher_metadata = load_latent(teacher_latent)
                validate_cached_metadata(
                    teacher_metadata,
                    expected_teacher_metadata,
                    artifact=teacher_latent,
                )
                if teacher_video.stat().st_size == 0:
                    raise ValueError(f"cached video is empty: {teacher_video}")
            else:
                reference, timesteps = engine.sample_reference(
                    prompt=prompt,
                    seed=seed,
                    shape=tuple(clean_latent.shape),
                )
                teacher_metadata = {
                    **expected_teacher_metadata,
                    "teacher_timesteps": timesteps,
                }
                save_video(engine.decode(reference), teacher_video, fps=fps)
                save_latent(reference, teacher_latent, metadata=teacher_metadata)
            teacher_rows.append(
                {
                    **teacher_metadata,
                    "prompt": prompt,
                    "video_path": str(teacher_video),
                    "latent_path": str(teacher_latent),
                }
            )
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (len(rows) - completed)
        print(
            f"teacher normalization: {completed}/{len(rows)} "
            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
            flush=True,
        )

    refined_manifest = None
    if generate_refine:
        refined_manifest = refine_root / "manifest.jsonl"
        write_jsonl(refined_manifest, refined_rows)
    teacher_manifest = None
    if generate_reference:
        teacher_manifest = reference_root / "manifest.jsonl"
        write_jsonl(teacher_manifest, teacher_rows)
    if generate_refine and generate_reference:
        config_name = "teacher_normalization_config.json"
    elif generate_refine:
        config_name = "teacher_refine_config.json"
    else:
        config_name = "teacher_sample_config.json"
    run_config = output_dir / config_name
    run_config.write_text(
        json.dumps(
            {
                "raw_manifest": str(Path(raw_manifest).resolve()),
                "model_root": str(model_root.resolve()),
                "teacher_model_name": teacher_model_name,
                "teacher_model_fingerprint_sha256": model_fingerprint,
                "strength": strength,
                "teacher_steps": teacher_steps,
                "teacher_shift": teacher_shift,
                "guidance_scale": guidance_scale,
                "negative_prompt_sha256": negative_prompt_sha256,
                "noise_seed_offset": noise_seed_offset,
                "implementation_sha256": implementation_fingerprint,
                "fps": fps,
                "torch_version": torch.__version__,
                "generate_refine": generate_refine,
                "generate_reference": generate_reference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return refined_manifest, teacher_manifest
