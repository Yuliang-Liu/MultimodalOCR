import argparse
import os
import time
import base64
import io
from PIL import Image
from typing import Optional
from openai import OpenAI

DEFAULT_PROMPT = """
You are an advanced hybrid OCR engine capable of processing multilingual text mixed with mathematical notation. Your goal is to transcribe the content with high fidelity. Strict Rules:

1. Multilingual Precision: Transcribe text exactly as it appears in the original language. Do not translate, summarize, or correct original spelling errors.

2. Math Formatting: Identify all mathematical expressions and convert them into LaTeX.

3. Inline Math: Use single dollar signs ($x$) for inline math (formulas within a sentence).

4. Display Math: Use double dollar signs ($$x$$) for display math (standalone formulas on their own lines).

5. Layout & Structure: Use Markdown to preserve the visual structure (headers, paragraphs, lists).

6. Table Formatting: Use HTML tags (e.g., <table>, <tr>, <th>, <td>) to generate any tables found in the text.

7. Output Only: Output the transcribed text directly without any conversational filler.
"""

def encode_image(image_path):
    """
    Reads the image file and converts it to a base64 string, as well as providing its MIME type.
    Scales down the image if its total pixels exceed 35,000,000 (Doubao limit is 36,000,000).
    Also re-encodes if file size is too large (e.g., > 10MB to be safe).
    """
    SUPPORTED_FORMATS = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp', 'GIF': 'image/gif'}
    
    with Image.open(image_path) as img:
        img_format = img.format if img.format in SUPPORTED_FORMATS else 'JPEG'
        
        MAX_PIXELS = 35_000_000  # Safe margin below 36,000,000
        MAX_SIZE = 8 * 1024 * 1024  # 8MB safe limit
        file_size = os.path.getsize(image_path)
        
        current_pixels = img.width * img.height
        
        if img.format in SUPPORTED_FORMATS and file_size <= MAX_SIZE and current_pixels <= MAX_PIXELS:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8"), SUPPORTED_FORMATS[img.format]
        
        # Need to process image
        if img.mode in ("RGBA", "P") and img_format == "JPEG":
            img = img.convert("RGB")
        
        quality = 95
        scale_factor = 1.0
        
        # Calculate initial scale factor to bring pixels under MAX_PIXELS
        if current_pixels > MAX_PIXELS:
            # Add a slight extra margin while scaling
            scale_factor = (MAX_PIXELS / current_pixels) ** 0.5 * 0.95
        
        while True:
            temp_img = img
            if scale_factor < 1.0:
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                temp_img = img.resize(new_size, resample)
                
            buffer = io.BytesIO()
            if img_format in ['JPEG', 'WEBP']:
                temp_img.save(buffer, format=img_format, quality=quality)
            else:
                temp_img.save(buffer, format=img_format)
                
            size_bytes = buffer.tell()
            
            # Check both byte size and pixel limit (we already clamped pixel limit above, so just file size loop)
            if size_bytes <= MAX_SIZE:
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8"), SUPPORTED_FORMATS[img_format]
                
            if img_format in ['JPEG', 'WEBP'] and quality > 30:
                quality -= 15
            else:
                scale_factor *= 0.8
                quality = 85

def doubao_ocr(client, model_name, img_path: str, prompt: str) -> Optional[str]:
    try:
        base64_image, mime_type = encode_image(img_path)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"\nError processing {img_path}: {exc}")
        return None

def process_folder(input_folder: str, output_folder: str, prompt: str, model_name: str, sleep_s: float):
    os.makedirs(output_folder, exist_ok=True)

    print("Initializing Doubao client...")
    api_key = os.getenv('ARK_API_KEY')
    if not api_key:
        print("Error: ARK_API_KEY environment variable is not set.")
        return

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_files = os.listdir(input_folder)
    image_files = [
        filename for filename in all_files if os.path.splitext(filename)[1].lower() in valid_extensions
    ]
    image_files.sort()

    total_files = len(image_files)
    print(f"Found {total_files} images in {input_folder}")

    # Simple progress loop without tqdm if not installed, but try to use if available
    try:
        from tqdm import tqdm
        iterator = tqdm(image_files, desc="Processing Images")
        using_tqdm = True
    except ImportError:
        def tqdm(iterable, desc=None):
            return iterable
        iterator = image_files
        using_tqdm = False

    for filename in iterator:
        img_path = os.path.join(input_folder, filename)
        file_stem = os.path.splitext(filename)[0]
        output_filename = f"{file_stem}.md"
        output_path = os.path.join(output_folder, output_filename)

        if os.path.exists(output_path):
            continue

        if not using_tqdm: # If not using tqdm, print progress
            print(f"Processing: {filename}")
            
        result = doubao_ocr(client, model_name, img_path, prompt)

        if result is not None:
            with open(output_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(result)
            if not using_tqdm:
                print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")

        if sleep_s > 0:
            time.sleep(sleep_s)

def main():
    parser = argparse.ArgumentParser(description="Batch process images using Doubao for OCR.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the folder to save markdown outputs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Custom prompt for the model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="doubao-seed-2-0-pro-260215",
        help="Model name for Doubao API",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between requests (seconds)")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    process_folder(
        args.input_dir,
        args.output_dir,
        args.prompt,
        args.model_name,
        args.sleep,
    )

if __name__ == "__main__":
    main()