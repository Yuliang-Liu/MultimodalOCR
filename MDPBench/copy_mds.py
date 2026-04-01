import os
import shutil
from pathlib import Path

img_dir = Path("/home/zhibolin/MultimodalOCR/MDPBench/demo_data/MDPBench_demo")
src_md_dir = Path("/home/zhibolin/Multy-lingualOCRBench/model_results/Gemini-3-pro-preview_public")
dst_md_dir = Path("/home/zhibolin/MultimodalOCR/MDPBench/demo_data/Gemini3-pro-preview_demo_result")

dst_md_dir.mkdir(parents=True, exist_ok=True)

copied = 0
for img_path in img_dir.iterdir():
    if img_path.is_file():
        stem = img_path.stem
        src_md_path = src_md_dir / f"{stem}.md"
        if src_md_path.exists():
            shutil.copy2(src_md_path, dst_md_dir / f"{stem}.md")
            copied += 1

print(f"Copied {copied} files to {dst_md_dir}")
