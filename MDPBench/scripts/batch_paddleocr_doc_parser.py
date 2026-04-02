#!/usr/bin/env python3
"""Batch wrapper for `paddleocr doc_parser`.

Example:
  /home/zhangli/miniconda3/envs/paddleocrvl/bin/python batch_paddleocr_doc_parser.py \
    --input_dir /path/to/images \
    --output_dir /path/to/out \
    --pipeline_version v1.5 \
    --vl_rec_backend vllm-server \
    --vl_rec_server_url http://127.0.0.1:8080/v1

Notes:
- Uses `sys.executable -m paddleocr` so it runs with the current Python environment.
- Creates one output subfolder per image to avoid overwriting.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def iter_images(input_dir: Path, recursive: bool, exts: tuple[str, ...]):
    if recursive:
        it = input_dir.rglob("*")
    else:
        it = input_dir.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def safe_rel_stem(path: Path, base_dir: Path) -> Path:
    rel = path.relative_to(base_dir)
    return rel.with_suffix("")


def has_any_output_files(out_dir: Path, marker_name: str) -> bool:
    if not out_dir.exists() or not out_dir.is_dir():
        return False
    for p in out_dir.iterdir():
        if p.name == marker_name:
            continue
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch run: paddleocr doc_parser on a folder of images")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing images")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder to write results")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subfolders")
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="Image extension to include (repeatable), e.g. --ext .png --ext .jpg. Default: common image types.",
    )

    parser.add_argument(
        "--pipeline_version",
        default="v1.5",
        choices=["v1", "v1.5"],
        help="PaddleOCR-VL pipeline version (v1.5 is default)",
    )

    parser.add_argument("--vl_rec_backend", default="vllm-server", help="e.g. vllm-server")
    parser.add_argument("--vl_rec_server_url", default="http://127.0.0.1:8080/v1", help="e.g. http://127.0.0.1:8080/v1")
    parser.add_argument("--vl_rec_max_concurrency", type=int, default=None)
    parser.add_argument("--vl_rec_api_model_name", default=None, help="Model name on the VLM server (optional)")
    parser.add_argument("--vl_rec_api_key", default=None, help="API key for the VLM server (optional)")

    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute")
    parser.add_argument("--continue_on_error", action="store_true", help="Skip failures and continue")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Disable resume and re-run all images")
    parser.add_argument(
        "--strict_resume_marker",
        action="store_true",
        help="Only skip when resume marker exists (do not use non-empty-dir heuristic)",
    )
    parser.add_argument(
        "--resume_marker_name",
        default=".doc_parser.done",
        help="Resume marker file name written after a successful image run",
    )
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir not found or not a directory: {input_dir}")

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in (args.ext or DEFAULT_EXTS))

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(input_dir, args.recursive, exts))
    images.sort()

    if not images:
        print(f"No images found under: {input_dir} (exts={exts}, recursive={args.recursive})")
        return 0

    failures: list[str] = []
    skipped = 0

    for idx, img_path in enumerate(images, start=1):
        rel_stem = safe_rel_stem(img_path, input_dir)
        out_subdir = output_dir / rel_stem
        out_subdir.mkdir(parents=True, exist_ok=True)
        marker_path = out_subdir / args.resume_marker_name

        if args.resume:
            done_by_marker = marker_path.exists()
            done_by_non_empty = (not args.strict_resume_marker) and has_any_output_files(
                out_subdir, args.resume_marker_name
            )
            if done_by_marker or done_by_non_empty:
                skipped += 1
                reason = "marker" if done_by_marker else "non-empty output"
                print(f"[{idx}/{len(images)}] SKIP {img_path} ({reason})")
                continue

        cmd = [
            sys.executable,
            "-m",
            "paddleocr",
            "doc_parser",
            "-i",
            str(img_path),
            "--save_path",
            str(out_subdir),
            "--pipeline_version",
            args.pipeline_version,
            "--vl_rec_backend",
            args.vl_rec_backend,
            "--use_layout_detection",
            "False",
            "--prompt_label",
            "formula",
        ]
        if args.vl_rec_server_url:
            cmd += ["--vl_rec_server_url", args.vl_rec_server_url]
        if args.vl_rec_max_concurrency is not None:
            cmd += ["--vl_rec_max_concurrency", str(args.vl_rec_max_concurrency)]
        if args.vl_rec_api_model_name:
            cmd += ["--vl_rec_api_model_name", args.vl_rec_api_model_name]
        if args.vl_rec_api_key:
            cmd += ["--vl_rec_api_key", args.vl_rec_api_key]

        print(f"[{idx}/{len(images)}] {img_path}")
        if args.dry_run:
            print(" ".join(cmd))
            continue

        env = os.environ.copy()

        try:
            subprocess.run(cmd, check=True, env=env)
            marker_path.write_text(f"ok\nimage={img_path}\n", encoding="utf-8")
        except subprocess.CalledProcessError as e:
            failures.append(f"{img_path}: exit={e.returncode}")
            print(f"FAILED: {img_path} (exit={e.returncode})")
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"- {f}")
        return 2

    print(f"\nDone. Results in: {output_dir} (processed={len(images) - skipped - len(failures)}, skipped={skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
