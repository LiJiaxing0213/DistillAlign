"""Command-line entry points for teacher-normalized distribution evaluation."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .io import read_jsonl, validate_jobs, validate_sample_manifest, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
PAPER_SEEDS = (
    11, 22, 33, 42, 44, 55, 66, 77,
    88, 123, 456, 789, 2024, 3407, 7777, 9999,
)


def csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def command_make_jobs(args: argparse.Namespace) -> None:
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    prompts = prompts[args.prompt_offset:]
    if args.num_prompts > 0:
        prompts = prompts[:args.num_prompts]
    if not prompts:
        raise ValueError("no prompts selected")
    rows = [
        {"prompt_id": prompt_id + args.prompt_offset, "seed": seed, "prompt": prompt}
        for prompt_id, prompt in enumerate(prompts)
        for seed in args.seeds
    ]
    validate_jobs(rows)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} jobs to {args.output}")


def command_sample(args: argparse.Namespace) -> None:
    from .sampling import sample_checkpoint

    manifest = sample_checkpoint(
        checkpoint=args.checkpoint,
        jobs_path=args.jobs,
        output_dir=args.output_dir,
        config_path=args.config,
        model_root=args.model_root,
        checkpoint_cache=args.checkpoint_cache,
        checkpoint_revision=args.checkpoint_revision,
        state_key=args.state_key,
        method=args.method,
        device=args.device,
        fps=args.fps,
        low_memory=args.low_memory,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(manifest)


def _negative_prompt(args: argparse.Namespace) -> str:
    if args.negative_prompt:
        return args.negative_prompt
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(args.config)
        value = config.get("negative_prompt")
        if value:
            return str(value)
    except Exception:
        pass

    from .teacher import FALLBACK_NEGATIVE_PROMPT

    return FALLBACK_NEGATIVE_PROMPT


def command_teacher_refine(args: argparse.Namespace) -> None:
    from .teacher import teacher_normalize

    refined, _ = teacher_normalize(
        raw_manifest=args.raw_manifest,
        output_dir=args.output_dir,
        model_root=args.model_root,
        teacher_model_name=args.teacher_model,
        negative_prompt=_negative_prompt(args),
        strength=args.renoise_strength,
        teacher_steps=args.teacher_steps,
        guidance_scale=args.guidance_scale,
        noise_seed_offset=args.noise_seed_offset,
        device=args.device,
        fps=args.fps,
        generate_refine=True,
        generate_reference=False,
    )
    print(refined)


def command_teacher_sample(args: argparse.Namespace) -> None:
    from .teacher import teacher_normalize

    _, reference = teacher_normalize(
        raw_manifest=args.raw_manifest,
        output_dir=args.output_dir,
        model_root=args.model_root,
        teacher_model_name=args.teacher_model,
        negative_prompt=_negative_prompt(args),
        teacher_steps=args.teacher_steps,
        teacher_shift=args.teacher_shift,
        guidance_scale=args.guidance_scale,
        device=args.device,
        fps=args.fps,
        generate_refine=False,
        generate_reference=True,
    )
    print(reference)


def command_extract(args: argparse.Namespace) -> None:
    from .features import extract_manifest

    metadata = extract_manifest(
        args.manifest,
        args.output,
        model_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        num_frames=args.num_frames,
        clip_frames=None if args.clip_frames == 0 else args.clip_frames,
        log_every=args.log_every,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def command_metrics(args: argparse.Namespace) -> None:
    from .metrics import compute_from_caches, write_metrics

    result = compute_from_caches(
        args.teacher_features,
        args.student_features,
        k=args.k,
    )
    write_metrics(result, args.output, args.csv_output)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_merge(args: argparse.Namespace) -> None:
    rows = []
    for manifest in args.manifests:
        rows.extend(read_jsonl(manifest))
    validate_sample_manifest(rows, require_files=True)
    rows.sort(key=lambda row: (int(row["prompt_id"]), int(row["seed"])))
    write_jsonl(args.output, rows)
    print(f"merged {len(rows)} rows into {args.output}")


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def command_run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "checkpoint": args.checkpoint,
        "checkpoint_cache": str(args.checkpoint_cache.resolve())
        if args.checkpoint_cache is not None
        else None,
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_state_key": args.state_key,
        "student_method": args.method,
        "student_low_memory": args.low_memory,
        "student_refine": not args.no_refine,
        "jobs": str(args.jobs.resolve()),
        "renoise_strength": args.renoise_strength,
        "teacher_model": args.teacher_model,
        "teacher_steps": args.teacher_steps,
        "teacher_shift": args.teacher_shift,
        "guidance_scale": args.guidance_scale,
        "vjepa_model": args.vjepa_model,
        "vjepa_revision": args.vjepa_revision,
        "vjepa_dtype": args.vjepa_dtype,
        "num_frames": args.num_frames,
        "clip_frames": args.clip_frames,
        "prdc_k": args.k,
    }
    (output / "pipeline_config.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    python = sys.executable

    if args.raw_manifest is None:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --raw-manifest is supplied")
        raw_manifest = output / "student_raw/manifest.jsonl"
        sample_command = [
            python, "-m", "distribution_eval.cli", "sample",
            "--checkpoint", args.checkpoint,
            "--jobs", str(args.jobs),
            "--output-dir", str(output / "student_raw"),
            "--config", str(args.config),
            "--model-root", str(args.model_root),
            "--method", args.method,
            "--state-key", args.state_key,
            "--device", args.device,
            "--fps", str(args.fps),
        ]
        if args.checkpoint_cache is not None:
            sample_command.extend(["--checkpoint-cache", str(args.checkpoint_cache)])
        if args.checkpoint_revision is not None:
            sample_command.extend(["--checkpoint-revision", args.checkpoint_revision])
        if args.low_memory is True:
            sample_command.append("--low-memory")
        elif args.low_memory is False:
            sample_command.append("--no-low-memory")
        _run(sample_command, dry_run=args.dry_run)
    else:
        raw_manifest = args.raw_manifest.resolve()

    skip_reference = args.teacher_manifest is not None or args.teacher_features is not None
    teacher_common = [
        "--raw-manifest", str(raw_manifest),
        "--model-root", str(args.model_root),
        "--teacher-model", args.teacher_model,
        "--config", str(args.config),
        "--teacher-steps", str(args.teacher_steps),
        "--guidance-scale", str(args.guidance_scale),
        "--device", args.device,
        "--fps", str(args.fps),
    ]
    if not args.no_refine:
        _run(
            [
                python, "-m", "distribution_eval.cli", "teacher-refine",
                "--output-dir", str(output / "student_refined"),
                *teacher_common,
                "--renoise-strength", str(args.renoise_strength),
            ],
            dry_run=args.dry_run,
        )
    if not skip_reference:
        _run(
            [
                python, "-m", "distribution_eval.cli", "teacher-sample",
                "--output-dir", str(output / "teacher_reference"),
                *teacher_common,
                "--teacher-shift", str(args.teacher_shift),
            ],
            dry_run=args.dry_run,
        )

    if args.no_refine:
        student_manifest = raw_manifest
        student_features = output / "student_raw/student_raw_vjepa2.npz"
    else:
        student_manifest = output / "student_refined/manifest.jsonl"
        student_features = output / "student_refined/student_refined_vjepa2.npz"
    extract_common = [
        "--model", args.vjepa_model,
        "--revision", args.vjepa_revision,
        "--cache-dir", str(args.model_cache),
        "--device", args.device,
        "--dtype", args.vjepa_dtype,
        "--num-frames", str(args.num_frames),
        "--clip-frames", str(args.clip_frames),
    ]
    _run(
        [
            python, "-m", "distribution_eval.cli", "extract",
            "--manifest", str(student_manifest),
            "--output", str(student_features),
            *extract_common,
        ],
        dry_run=args.dry_run,
    )

    if args.teacher_features is not None:
        teacher_features = args.teacher_features.resolve()
    else:
        teacher_manifest = (
            args.teacher_manifest.resolve()
            if args.teacher_manifest is not None
            else output / "teacher_reference/manifest.jsonl"
        )
        teacher_features = output / "teacher_reference/teacher_reference_vjepa2.npz"
        _run(
            [
                python, "-m", "distribution_eval.cli", "extract",
                "--manifest", str(teacher_manifest),
                "--output", str(teacher_features),
                *extract_common,
            ],
            dry_run=args.dry_run,
        )

    _run(
        [
            python, "-m", "distribution_eval.cli", "metrics",
            "--teacher-features", str(teacher_features),
            "--student-features", str(student_features),
            "--k", str(args.k),
            "--output", str(output / "precision_coverage.json"),
            "--csv-output", str(output / "precision_coverage.csv"),
        ],
        dry_run=args.dry_run,
    )


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/self_forcing_dmd.yaml")
    parser.add_argument("--model-root", type=Path, default=ROOT / "wan_models")
    parser.add_argument("--checkpoint-cache", type=Path)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--state-key", choices=["auto", "generator", "generator_ema", "root"], default="auto")
    parser.add_argument("--method", default="student_raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=16)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--low-memory", action="store_true", dest="low_memory")
    memory.add_argument("--no-low-memory", action="store_false", dest="low_memory")
    parser.set_defaults(low_memory=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)


def add_teacher_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=ROOT / "wan_models")
    parser.add_argument("--teacher-model", default="Wan2.1-T2V-14B")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/self_forcing_dmd.yaml")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--teacher-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=16)


def add_teacher_refine_arguments(parser: argparse.ArgumentParser) -> None:
    add_teacher_common_arguments(parser)
    parser.add_argument("--renoise-strength", type=float, default=0.9)
    parser.add_argument("--noise-seed-offset", type=int, default=0)


def add_teacher_sample_arguments(parser: argparse.ArgumentParser) -> None:
    add_teacher_common_arguments(parser)
    parser.add_argument("--teacher-shift", type=float, default=8.0)


def add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    from .features import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--clip-frames", type=int, default=81, help="Use 0 for the full decoded clip.")
    parser.add_argument("--log-every", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    from .features import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_jobs = subparsers.add_parser("make-jobs", help="Build the prompt/seed eval set (JSONL).")
    make_jobs.add_argument("--prompts", type=Path, default=ROOT / "prompts/distribution_eval_16.txt")
    make_jobs.add_argument("--output", type=Path, default=ROOT / "outputs/eval_jobs.jsonl")
    make_jobs.add_argument(
        "--seeds",
        type=csv_ints,
        default=list(PAPER_SEEDS),
    )
    make_jobs.add_argument("--num-prompts", type=int, default=16)
    make_jobs.add_argument("--prompt-offset", type=int, default=0)
    make_jobs.set_defaults(func=command_make_jobs)

    sample = subparsers.add_parser("sample", help="Sample the student and save endpoint latents.")
    add_sample_arguments(sample)
    sample.set_defaults(func=command_sample)

    refine = subparsers.add_parser("teacher-refine", help="Re-noise student latents and refine them with the teacher.")
    add_teacher_refine_arguments(refine)
    refine.set_defaults(func=command_teacher_refine)

    teacher_sample = subparsers.add_parser("teacher-sample", help="Sample the teacher reference on the same prompt/seed grid.")
    add_teacher_sample_arguments(teacher_sample)
    teacher_sample.set_defaults(func=command_teacher_sample)

    extract = subparsers.add_parser("extract", help="Extract strict V-JEPA2 features.")
    add_extract_arguments(extract)
    extract.set_defaults(func=command_extract)

    metrics = subparsers.add_parser("metrics", help="Compute PRDC precision and coverage.")
    metrics.add_argument("--teacher-features", type=Path, required=True)
    metrics.add_argument("--student-features", type=Path, required=True)
    metrics.add_argument("--k", type=int, default=5)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--csv-output", type=Path)
    metrics.set_defaults(func=command_metrics)

    merge = subparsers.add_parser("merge-manifests", help="Merge sample manifest shards.")
    merge.add_argument("--manifests", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(func=command_merge)

    run = subparsers.add_parser("run", help="Run sampling, normalization, encoding, and metrics.")
    run.add_argument("--checkpoint")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--raw-manifest", type=Path)
    run.add_argument("--teacher-manifest", type=Path)
    run.add_argument("--teacher-features", type=Path)
    run.add_argument("--config", type=Path, default=ROOT / "configs/self_forcing_dmd.yaml")
    run.add_argument("--model-root", type=Path, default=ROOT / "wan_models")
    run.add_argument("--model-cache", type=Path, default=ROOT / ".cache/huggingface")
    run.add_argument("--checkpoint-cache", type=Path)
    run.add_argument("--checkpoint-revision")
    run.add_argument("--state-key", choices=["auto", "generator", "generator_ema", "root"], default="auto")
    run.add_argument("--method", default="student_raw")
    run.add_argument("--teacher-model", default="Wan2.1-T2V-14B")
    run.add_argument("--renoise-strength", type=float, default=0.9)
    run.add_argument("--teacher-steps", type=int, default=25)
    run.add_argument("--teacher-shift", type=float, default=8.0)
    run.add_argument("--guidance-scale", type=float, default=5.0)
    run.add_argument("--vjepa-model", default=DEFAULT_MODEL_ID)
    run.add_argument("--vjepa-revision", default=DEFAULT_MODEL_REVISION)
    run.add_argument("--vjepa-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    run.add_argument("--num-frames", type=int, default=8)
    run.add_argument("--clip-frames", type=int, default=81)
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--device", default="cuda")
    run.add_argument("--fps", type=int, default=16)
    run.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip teacher re-noise refinement and encode raw student videos "
        "directly (post-DMD protocol). Refinement is on by default.",
    )
    run_memory = run.add_mutually_exclusive_group()
    run_memory.add_argument("--low-memory", action="store_true", dest="low_memory")
    run_memory.add_argument("--no-low-memory", action="store_false", dest="low_memory")
    run.set_defaults(low_memory=None)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
