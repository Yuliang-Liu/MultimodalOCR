import os
import argparse
import base64
import mimetypes
import time
import io
from PIL import Image
from anthropic import Anthropic

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

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
    SUPPORTED_FORMATS = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp', 'GIF': 'image/gif'}
    
    with Image.open(image_path) as img:
        # 获取图片的真实格式
        img_format = img.format if img.format in SUPPORTED_FORMATS else 'JPEG'
        
        MAX_SIZE = 3.5 * 1024 * 1024
        file_size = os.path.getsize(image_path)
        
        if img.format in SUPPORTED_FORMATS and file_size <= MAX_SIZE:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8"), SUPPORTED_FORMATS[img.format]
        
        if img.mode in ("RGBA", "P") and img_format == "JPEG":
            img = img.convert("RGB")
            
        quality = 95
        scale_factor = 1.0
        
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
            if size_bytes <= MAX_SIZE:
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8"), SUPPORTED_FORMATS[img_format]
                
            if img_format in ['JPEG', 'WEBP'] and quality > 30:
                quality -= 15
            else:
                scale_factor *= 0.8
                quality = 85

def process_image(client, img_path, prompt, model_name="claude-sonnet-4-6"):
    try:
        base64_image, mime_type = encode_image(img_path)
        
        response = client.messages.create(
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image,
                            }
                        }
                    ],
                }
            ],
            model=model_name,
        )
        if isinstance(response.content, list):
            text_content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    text_content += block.text
                elif isinstance(block, dict) and 'text' in block:
                    text_content += block['text']
            return text_content
        else:
            return str(response.content)
            
    except Exception as e:
        print(f"\\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prompt):
    os.makedirs(output_folder, exist_ok=True)

    print("Initializing Anthropic Client...")
    api_key = os.getenv('API_KEY')
        

    client = Anthropic(
        base_url=os.getenv('BASE_URL'),
        api_key=api_key
    )

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    if not os.path.exists(input_folder):
         print(f"Error: Input directory {input_folder} does not exist.")
         return

    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
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
        result = process_image(client, img_path, prompt)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")
        
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process images using Claude Sonnet 5.6 for OCR.")
    parser.add_argument('--input_dir', type=str, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, help='Path to the folder to save markdown outputs')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Custom prompt for the model')

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        process_folder(args.input_dir, args.output_dir, args.prompt)
    else:
        parser.print_help()