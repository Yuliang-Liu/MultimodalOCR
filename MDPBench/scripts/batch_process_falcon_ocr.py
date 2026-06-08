import os
import argparse
import base64
import mimetypes
import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_falcon_ocr(img_path, url, skip_layout):
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        mime_type = 'image/jpeg' 

    try:
        base64_image = encode_image(img_path)
        data_uri = f"data:{mime_type};base64,{base64_image}"
        
        payload = {
            "images": [data_uri],
            "skip_layout": skip_layout
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        resp_json = response.json()
        
        # Return only markdown_result. Also handle the legacy string response format.
        if isinstance(resp_json, dict):
            return resp_json.get("markdown_result", "")
        elif isinstance(resp_json, str):
            import ast
            try:
                data = ast.literal_eval(resp_json)
                return data.get("markdown_result", "")
            except:
                pass
        return ""
            
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, url, skip_layout):
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    image_files.sort() 
    
    total_files = len(image_files)
    print(f"Found {total_files} images in {input_folder}")
    print(f"Target API: {url} (skip_layout: {skip_layout})")

    for filename in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(input_folder, filename)
        
        file_stem = os.path.splitext(filename)[0]
        output_filename = f"{file_stem}.md"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            continue
        
        result = call_falcon_ocr(img_path, url, skip_layout)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
        else:
            print(f"Failed to process/empty result for: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process images using local Falcon-OCR via API.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to save markdown outputs')
    parser.add_argument('--url', type=str, default='http://localhost:5002/falconocr/parse', help='Falcon OCR Server API URL')
    parser.add_argument('--skip_layout', action='store_true', help='Set this flag to skip layout detection (default is false)')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        process_folder(args.input_dir, args.output_dir, args.url, args.skip_layout)
