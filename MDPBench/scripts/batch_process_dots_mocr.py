#!/usr/bin/env python3
"""Run dots.mocr through a vLLM server and write MDPBench-compatible Markdown.

Start the server first, for example:
  vllm serve /path/to/dots-mocr/weights/DotsMOCR --served-model-name dots-mocr ...

The script deliberately uses DotsMOCRParser's official post-processing.  Its
layout JSON is converted to Markdown, rather than being written directly as a
prediction file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dots-repo",
        type=Path,
        required=True,
        help="Clone of https://github.com/rednote-hilab/dots.mocr.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="dots-mocr")
    parser.add_argument("--prompt", default="prompt_layout_all_en")
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--no-fitz-preprocess",
        action="store_true",
        help="Do not render image inputs through the official fitz preprocessing path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if not args.dots_repo.is_dir():
        raise SystemExit(f"dots.mocr repository does not exist: {args.dots_repo}")

    # The repository is intentionally imported from its checkout, so this
    # script does not depend on editing the benchmark environment's packages.
    sys.path.insert(0, str(args.dots_repo))
    from dots_mocr.parser import DotsMOCRParser

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"No supported images in {args.input_dir}")

    parser = DotsMOCRParser(
        ip=args.host,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        output_dir=str(args.output_dir),
        use_hf=False,
    )

    failures: list[tuple[str, str]] = []
    for image_path in tqdm(image_paths, desc="dots.mocr"):
        prediction_path = args.output_dir / f"{image_path.stem}.md"
        if prediction_path.exists() and not args.overwrite:
            continue
        try:
            # parse_image writes <stem>.md directly into output_dir.  The
            # filename is exactly what MDPBench's evaluator looks up.
            parser.parse_image(
                str(image_path),
                image_path.stem,
                args.prompt,
                str(args.output_dir),
                fitz_preprocess=not args.no_fitz_preprocess,
            )
        except Exception as exc:  # keep a long evaluation resumable
            failures.append((image_path.name, repr(exc)))
            print(f"FAILED {image_path.name}: {exc}", file=sys.stderr)

    failure_log = args.output_dir / "failures.txt"
    if failures:
        failure_log.write_text(
            "\n".join(f"{name}\t{error}" for name, error in failures) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"{len(failures)} images failed; see {failure_log}")
    failure_log.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
