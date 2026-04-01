import argparse
import os
import time
import base64
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
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def kimi_ocr(client, model_name, img_path: str, prompt: str) -> Optional[str]:
    try:
        base64_image = encode_image(img_path)
        
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext == '.png':
            mime_type = "image/png"
        elif ext == '.webp':
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg" 
            
        image_url = f"data:{mime_type};base64,{base64_image}"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
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

    print("Initializing Kimi client...")
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("Error: MOONSHOT_API_KEY environment variable is not set.")
        return

    client = OpenAI(
        base_url=os.getenv('BASE_URL'),
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

        if not using_tqdm:
            print(f"Processing: {filename}")
            
        result = kimi_ocr(client, model_name, img_path, prompt)

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
    parser = argparse.ArgumentParser(description="Batch process images using Kimi for OCR.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the folder to save markdown outputs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Custom prompt for the model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="kimi-k2.5",
        help="Model name for Kimi API",
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
 