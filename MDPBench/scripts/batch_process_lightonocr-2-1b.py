import os
import argparse
import base64
import mimetypes
import time
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

DEFAULT_PROMPT = """Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers.Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."""

MODEL_NAME = "lightonai/LightOnOCR-2-1B"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_lightonocr(client, img_path, prompt):
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        mime_type = 'image/jpeg'

    try:
        base64_image = encode_image(img_path)
        
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}", 
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prompt):
    os.makedirs(output_folder, exist_ok=True)

    print(f"Initializing OpenAI Client for {MODEL_NAME}...")
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8800/v1",
        timeout=3600
    )

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    if not os.path.exists(input_folder):
         print(f"Error: Input directory '{input_folder}' does not exist.")
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
        result = call_lightonocr(client, img_path, prompt)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Batch process images using {MODEL_NAME}.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to save markdown outputs')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Custom prompt for the model')

    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.prompt)
