from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from PIL import Image
from pathlib import Path
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

def process_folder(input_dir: str, output_dir: str, batch_size: int = 4):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    image_files = [f for f in sorted(input_path.iterdir()) if f.suffix.lower() in IMAGE_EXTS]

    if not image_files:
        print(f"No image files found: {input_dir}")
        return

    print(f"Found {len(image_files)} images. Connecting to the vLLM service...")
    manager = InferenceManager(method="vllm")

    remaining_files = []
    for f in image_files:
        out_file = output_path / (f.stem + ".md")
        if not out_file.exists():
            remaining_files.append(f)

    if not remaining_files:
        print("All images have already been processed. Nothing to do.")
        return

    for i in tqdm(range(0, len(remaining_files), batch_size), desc="Processing Images"):
        batch_files = remaining_files[i:i + batch_size]

        batch = [
            BatchInputItem(
                image=Image.open(f).convert("RGB"),
                prompt_type="ocr_layout"
            )
            for f in batch_files
        ]

        results = manager.generate(batch)

        for f, result in zip(batch_files, results):
            out_file = output_path / (f.stem + ".md")
            out_file.write_text(result.markdown, encoding="utf-8")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch OCR images to Markdown")
    parser.add_argument("input_dir", help="Input image directory")
    parser.add_argument("output_dir", help="Output Markdown directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images per batch (default: 4)")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.batch_size)
