import os
import glob

def main():
    img_dir = "/home/zhibolin/Multy-lingualOCRBench/demo_data/MDPBench_img_public"
    
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} does not exist.")
        return
        
    all_files = os.listdir(img_dir)
    deleted_count = 0
    kept_count = 0
    
    for f in all_files:
        f_lower = f.lower()
        if 'indoor' in f_lower or 'outdoor' in f_lower:
            file_path = os.path.join(img_dir, f)
            os.remove(file_path)
            deleted_count += 1
        else:
            kept_count += 1
            
    print(f"删除了 {deleted_count} 个带有 indoor 或 outdoor 后缀的图像文件。")
    print(f"保留了 {kept_count} 个原图文件。")

if __name__ == '__main__':
    main()
