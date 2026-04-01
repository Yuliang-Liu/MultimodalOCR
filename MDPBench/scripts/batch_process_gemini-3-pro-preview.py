import os
import argparse
import mimetypes
import time
import threading
from google import genai
from google.genai import types

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

def gemini3(client, img_path, prompt):
    """
    调用 Gemini 3 模型处理单个图片
    """
    # 根据文件扩展名猜测 MIME 类型
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type is None:
        mime_type = 'image/jpeg' # 默认回退到 jpeg

    # 读取图片数据
    try:
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        result_container = []
        exception_container = []

        def _call_api():
            try:
                response = client.models.generate_content(
                    model='gemini-3-pro-preview',
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    ),
                )
                result_container.append(response)
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=_call_api, daemon=True)
        thread.start()
        thread.join(timeout=60)

        if thread.is_alive():
            print(f"\nTimeout (60s) processing {img_path}")
            return None

        if exception_container:
            raise exception_container[0]

        return result_container[0].text
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None

def process_folder(input_folder, output_folder, prompt, recursive=False):
    """
    遍历文件夹处理图片
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 初始化客户端
    print("Initializing Gemini Client...")
    client = genai.Client(
        api_key= os.getenv("CLOSEAI_API_KEY"),
        vertexai=True, # 优先使用vertexai协议访问，稳定性更高
        http_options={
            "base_url": "https://api.openai-proxy.org/google"
        },
    )

    # 支持的图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.JPEG', '.JPG', '.PNG'}

    # 收集图片: recursive=False 仅当前目录；recursive=True 包含子目录
    image_files = []
    if recursive:
        for root, _, files in os.walk(input_folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in valid_extensions:
                    abs_path = os.path.join(root, name)
                    rel_path = os.path.relpath(abs_path, input_folder)
                    image_files.append((abs_path, rel_path))
    else:
        for name in os.listdir(input_folder):
            abs_path = os.path.join(input_folder, name)
            if not os.path.isfile(abs_path):
                continue
            if os.path.splitext(name)[1].lower() in valid_extensions:
                image_files.append((abs_path, name))

    image_files.sort(key=lambda x: x[1])  # 按相对路径排序

    total_files = len(image_files)
    print(f"Found {total_files} images in {input_folder}")

    for img_path, relative_path in tqdm(image_files, desc="Processing Images"):
        # 构建输出文件路径 (同名 .md)
        rel_stem = os.path.splitext(relative_path)[0]
        output_path = os.path.join(output_folder, f"{rel_stem}.md")
        output_dirname = os.path.dirname(output_path)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)
        output_filename = os.path.basename(output_path)
        display_name = relative_path
        
        # 如果输出文件已存在，跳过（支持断点续传）
        if os.path.exists(output_path):
            # print(f"Skipping {filename}, output already exists.") # 减少刷屏，如需调试可取消注释
            continue
        
        # 处理图片
        print(f"Processing: {display_name}")
        result = gemini3(client, img_path, prompt)
        
        if result:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Saved: {os.path.relpath(output_path, output_folder)}")
        else:
            print(f"Failed to process: {display_name}")
        
        # 避免请求过于频繁，稍微sleep一下
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch process images using Gemini 3 for OCR.")
    parser.add_argument('--input_dir', type=str, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, help='Path to the folder to save markdown outputs')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Custom prompt for the model')
    parser.add_argument('--recursive', action='store_true', help='Recursively search images in subfolders')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        process_folder(args.input_dir, args.output_dir, args.prompt, recursive=args.recursive)