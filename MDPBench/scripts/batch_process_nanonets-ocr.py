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

# 设置默认的Prompt (保留 Nanonets 原始 Prompt)
DEFAULT_PROMPT = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."

def encode_image(image_path):
    """
    将图片文件编码为 base64 字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_page_with_nanonets_s(client, img_path, prompt):
    """
    调用 Nanonets-OCR-s 模型处理单个图片
    """
    # 根据文件扩展名猜测 MIME 类型
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        mime_type = 'image/png' # 默认回退到 png
    
    # 编码图片
    try:
        base64_image = encode_image(img_path)
        
        response = client.chat.completions.create(
            model="nanonets/Nanonets-OCR-s",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=15000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prompt):
    """
    遍历文件夹处理图片
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 初始化客户端
    print("Initializing OpenAI Client for Nanonets...")
    client = OpenAI(
        api_key="123",
        base_url="http://localhost:8000/v1"
    )

    # 支持的图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # 获取所有图片文件
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    image_files.sort() # 按文件名排序
    
    total_files = len(image_files)
    print(f"Found {total_files} images in {input_folder}")

    for filename in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(input_folder, filename)
        
        # 构建输出文件路径 (同名 .md)
        file_stem = os.path.splitext(filename)[0]
        output_filename = f"{file_stem}.md"
        output_path = os.path.join(output_folder, output_filename)
        
        # 如果输出文件已存在，跳过（支持断点续传）
        if os.path.exists(output_path):
            continue
        
        # 处理图片
        print(f"Processing: {filename}")
        result = ocr_page_with_nanonets_s(client, img_path, prompt)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved: {output_filename}")
        else:
            print(f"Failed to process: {filename}")
        
        # 避免请求过于频繁，稍微sleep一下
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process images using Nanonets for OCR.")
    parser.add_argument('--input_dir', type=str, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, help='Path to the folder to save markdown outputs')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Custom prompt for the model')

    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        parser.print_help()
    elif not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        process_folder(args.input_dir, args.output_dir, args.prompt)
