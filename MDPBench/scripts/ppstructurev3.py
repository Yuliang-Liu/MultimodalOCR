from __future__ import annotations

import argparse
import multiprocessing as mp
import traceback
from pathlib import Path

from paddleocr import PPStructureV3


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"input path not found: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images: list[Path] = []
    for p in iterator:
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            images.append(p)
    return sorted(images)


def _worker_loop(conn, pipeline_kwargs: dict) -> None:
    pipeline = PPStructureV3(**pipeline_kwargs)
    while True:
        msg = conn.recv()
        if msg is None:
            break

        image_path = msg["image_path"]
        output_dir = msg["output_dir"]
        try:
            # 支持传入 predict 级别的可选参数（如只启用公式识别）
            predict_kwargs = msg.get("predict_kwargs") or {}
            output = pipeline.predict(image_path, **predict_kwargs)
            for res in output:
                res.save_to_json(save_path=output_dir)
                res.save_to_markdown(save_path=output_dir)
            conn.send({"ok": True})
        except Exception:
            conn.send({"ok": False, "error": traceback.format_exc()})


def _start_worker(pipeline_kwargs: dict, predict_kwargs: dict):
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_worker_loop, args=(child_conn, pipeline_kwargs), daemon=True)
    proc.start()
    return proc, parent_conn


def main() -> int:
    parser = argparse.ArgumentParser(description="PPStructureV3 batch inference")
    parser.add_argument("--input", required=True, help="输入：图片文件或包含图片的文件夹")
    parser.add_argument("--output", required=True, help="输出文件夹（每张图一个子目录）")
    parser.add_argument("--device", default=None, help="推理设备，例如 cpu / gpu。为空则使用默认")
    parser.add_argument("--lang", default=None, help="语言，例如 en。为空则使用默认")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="单张图片推理超时秒数，>0 则超时跳过并重启 worker（用于避免卡死）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="开启断点续传：如果输出文件夹中已存在处理结果，则跳过该图片",
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        default=0,
        help="每处理 N 张图片重启一次 worker（0 表示不主动重启）",
    )
    parser.add_argument(
        "--fail-list",
        default=True,
        help="保存失败/超时图片路径的 txt 文件（可选）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归遍历输入文件夹（包含子文件夹）",
    )
    parser.add_argument(
        "--formula-only",
        action="store_true",
        help="只运行公式识别模块（其它模块将被禁用）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs: dict = {}
    if args.device:
        pipeline_kwargs["device"] = args.device
    if args.lang:
        pipeline_kwargs["lang"] = args.lang
    pipeline_kwargs["use_doc_unwarping"] = True

    # 构建 predict 级别的 kwargs（例如只运行公式识别）
    predict_kwargs: dict = {}
    if getattr(args, "formula_only", False):
        # 传递自定义 yaml 配置文件以在此基础上禁用特定的模型加载（如：layout）
        pipeline_kwargs["paddlex_config"] = "/home/zhangli/Multy-lingualOCRBench/scripts/PP-StructureV3.yaml"
        predict_kwargs.update(
            {
                "use_formula_recognition": True,
                "use_table_recognition": False,
                "use_region_detection": False,
                "use_chart_recognition": False,
                "use_seal_recognition": False,
                "use_doc_unwarping": False,
                "use_doc_orientation_classify": False,
                "use_textline_orientation": False,
            }
        )

    images = _iter_images(input_path, recursive=args.recursive)
    if not images:
        print(f"No images found under: {input_path}")
        return 1

    fail_lines: list[str] = []
    worker_proc, worker_conn = _start_worker(pipeline_kwargs, predict_kwargs)
    processed_since_restart = 0

    for i, image_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] Processing: {image_path}")

        per_image_out = output_root / image_path.stem
        per_image_out.mkdir(parents=True, exist_ok=True)

        if getattr(args, "resume", False) and per_image_out.exists():
            # 检查文件夹内是否已经存在 .json 或 .md 文件
            # 这里的判断标准可以根据 PPStructure 实际生成的文件名进行调整
            if any(per_image_out.glob("*.json")) or any(per_image_out.glob("*.md")):
                print(f"[{i}/{len(images)}] 已处理过，跳过: {image_path}")
                continue

        if not worker_proc.is_alive():
            try:
                worker_proc.join(timeout=0.2)
            except Exception:
                pass
            worker_proc, worker_conn = _start_worker(pipeline_kwargs, predict_kwargs)
            processed_since_restart = 0

        # 将 predict 级别的 kwargs 一并发送给 worker
        worker_conn.send(
            {
                "image_path": str(image_path),
                "output_dir": str(per_image_out),
                "predict_kwargs": predict_kwargs,
            }
        )

        if args.timeout and args.timeout > 0:
            if not worker_conn.poll(args.timeout):
                fail_lines.append(f"TIMEOUT\t{image_path}")
                print(f"  -> TIMEOUT after {args.timeout}s, skipping and restarting worker")
                worker_proc.terminate()
                worker_proc.join(timeout=2.0)
                worker_proc, worker_conn = _start_worker(pipeline_kwargs, predict_kwargs)
                processed_since_restart = 0
                continue

        result = worker_conn.recv()
        if not result.get("ok", False):
            fail_lines.append(f"ERROR\t{image_path}\t{result.get('error','')}")
            print("  -> ERROR, skipping (details recorded)")

        processed_since_restart += 1
        if args.restart_every and args.restart_every > 0 and processed_since_restart >= args.restart_every:
            worker_proc.terminate()
            worker_proc.join(timeout=2.0)
            worker_proc, worker_conn = _start_worker(pipeline_kwargs, predict_kwargs)
            processed_since_restart = 0

    try:
        if worker_proc.is_alive():
            worker_conn.send(None)
            worker_proc.join(timeout=2.0)
    except Exception:
        pass

    if args.fail_list and fail_lines:
        fail_path = Path(args.fail_list)
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
        print(f"Failures saved to: {fail_path}")

    print(f"Done. Outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())