import argparse
import os
import time
from typing import Optional

import torch
from PIL import Image
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

DEFAULT_PROMPT = "You are an advanced hybrid OCR engine capable of processing multilingual text mixed with mathematical notation. Your goal is to transcribe the content with high fidelity.Strict Rules: 1. Multilingual Precision: Transcribe text exactly as it appears in the original language. Do not translate, summarize, or correct original spelling errors. 2. Math Formatting: Identify all mathematical expressions and convert them into LaTeX. 3. Use single dollar signs ($x$) for inline math (formulas within a sentence). 4. Use double dollar signs ($$x$$) for display math (standalone formulas on their own lines). 5. Layout & Structure: Use Markdown to preserve the visual structure (headers, paragraphs, lists). 6. Output Only: Output the transcribed text directly without any conversational filler."


def load_model(model_path: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def qwen3_ocr(model, processor, img_path: str, prompt: str, max_new_tokens: int) -> Optional[str]:
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"\nError reading {img_path}: {exc}")
        return None

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0] if output_text else ""
    except Exception as exc:
        print(f"\nError processing {img_path}: {exc}")
        return None


def process_folder(input_folder: str, output_folder: str, prompt: str, model_path: str, max_new_tokens: int, sleep_s: float):
    os.makedirs(output_folder, exist_ok=True)

    print("Initializing Qwen3-VL model...")
    model, processor = load_model(model_path)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_files = os.listdir(input_folder)
    image_files = [
        filename for filename in all_files if os.path.splitext(filename)[1].lower() in valid_extensions
    ]
    image_files.sort()

    total_files = len(image_files)
    print(f"Found {total_files} images in {input_folder}")

    for filename in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(input_folder, filename)
        file_stem = os.path.splitext(filename)[0]
        output_filename = f"{file_stem}.md"
        output_path = os.path.join(output_folder, output_filename)

        if os.path.exists(output_path):
            continue

        print(f"Processing: {filename}")
        result = qwen3_ocr(model, processor, img_path, prompt, max_new_tokens)

        if result is not None:
            with open(output_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(result)
            print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")

        if sleep_s > 0:
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Batch process images using Qwen3-VL for OCR.")
    parser.add_argument("--input_dir", type=str, help="Path to the folder containing input images")
    parser.add_argument("--output_dir", type=str, help="Path to the folder to save markdown outputs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Custom prompt for the model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Local path or hub id for Qwen3-VL model",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens per image")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests (seconds)")

    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        print("Error: --input_dir and --output_dir are required.")
        return

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    process_folder(
        args.input_dir,
        args.output_dir,
        args.prompt,
        args.model_path,
        args.max_new_tokens,
        args.sleep,
    )


if __name__ == "__main__":
    main()