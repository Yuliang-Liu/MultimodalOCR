import os
import argparse
import base64
import mimetypes
import time
from openai import OpenAI

# 尝试导入 tqdm 用于显示进度条，如果没有安装则使用普通 range
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

# 设置默认的Prompt
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
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt52(client, img_path, prompt):
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        mime_type = 'image/jpeg' 

    try:
        base64_image = encode_image(img_path)
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:{mime_type};base64,{base64_image}",
                        },
                    ],
                }
            ],
            model="gpt-5.2-2025-12-11",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prompt):
    os.makedirs(output_folder, exist_ok=True)

    print("Initializing OpenAI Client...")
    client = OpenAI(
        base_url=os.getenv('BASE_URL'),
        api_key= os.getenv('API_KEY'),
    )

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
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
        result = gpt52(client, img_path, prompt)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")
        
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process images using GPT-5.2 for OCR.")
    parser.add_argument('--input_dir', type=str, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, help='Path to the folder to save markdown outputs')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Custom prompt for the model')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        process_folder(args.input_dir, args.output_dir, args.prompt)