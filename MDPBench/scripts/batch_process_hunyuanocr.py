import os
import argparse
import base64
import mimetypes
import time
from openai import OpenAI

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

# Default Prompt for HunyuanOCR
DEFAULT_PROMPT = """Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers.Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."""

def encode_image(image_path):
    """
    Encode image to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_hunyuanocr(client, img_path, prompt):
    """
    Call HunyuanOCR model to process a single image
    """
    # Guess mime type
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        mime_type = 'image/jpeg' # Default fallback

    try:
        base64_image = encode_image(img_path)
        
        # Prepare messages
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
            model="tencent/HunyuanOCR",
            messages=messages,
            temperature=0.0,
            extra_body={
                "top_k": 1,
                "repetition_penalty": 1.0
            },
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prompt):
    """
    Traverse folder and process images
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize Client for HunyuanOCR
    print("Initializing OpenAI Client for HunyuanOCR...")
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=3600
    )

    # Note: Hunyuan script had a very long timeout, keeping it high.

    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    if not os.path.exists(input_folder):
         print(f"Error: Input directory '{input_folder}' does not exist.")
         return

    # Get all image files
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    image_files.sort() # Sort by filename
    
    total_files = len(image_files)
    print(f"Found {total_files} images in {input_folder}")

    for filename in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(input_folder, filename)
        
        # Build output file path (same name .md)
        file_stem = os.path.splitext(filename)[0]
        output_filename = f"{file_stem}.md"
        output_path = os.path.join(output_folder, output_filename)
        
        # If output exists, skip (resume support)
        if os.path.exists(output_path):
            # print(f"Skipping {filename}, output already exists.")
            continue
        
        # Process image
        print(f"Processing: {filename}")
        result = call_hunyuanocr(client, img_path, prompt)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")
        
        # Avoid too frequent requests (optional for local, but good practice)
        # time.sleep(0.1) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process images using HunyuanOCR.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to save markdown outputs')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Custom prompt for the model')

    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.prompt)
