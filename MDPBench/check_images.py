import os
import re
from collections import defaultdict

def main():
    img_dir = "/home/zhibolin/Multy-lingualOCRBench/demo_data/MDPBench_img_public"
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} does not exist.")
        return

    files = os.listdir(img_dir)
    
    # 按照语言_类型_数字的分组前缀，如 ar_academicpaper_25
    groups = defaultdict(list)
    pattern = re.compile(r'^([a-zA-Z\-]+_[a-zA-Z]+_\d+)')
    
    for f in files:
        if f.startswith('.'): continue
        match = pattern.match(f)
        if match:
            base_name = match.group(1)
            groups[base_name].append(f)
        else:
            groups['UNKNOWN'].append(f)
            
    invalid_groups = []
    
    for base, f_list in groups.items():
        if base == 'UNKNOWN':
            continue
            
        originals = []
        indoors = []
        outdoors = []
        
        for f in f_list:
            # 兼容有些拼写缺下划线的情况例如 indoorlight
            if 'indoor' in f.lower():
                indoors.append(f)
            elif 'outdoor' in f.lower():
                outdoors.append(f)
            else:
                originals.append(f)
                
        # 验证是否符合 1原图, 2 indoor, 1 outdoor 的规则
        if len(originals) != 1 or len(indoors) != 2 or len(outdoors) != 1:
            invalid_groups.append((base, originals, indoors, outdoors))
            
    if not invalid_groups:
        print("所有图像组均符合【1原图, 2张indoor, 1张outdoor】的规范。")
    else:
        print(f"发现 {len(invalid_groups)} 组不符合【1原图, 2张indoor, 1张outdoor】的规范！")
        report_path = "abnormal_images_report.txt"
        with open(report_path, "w", encoding="utf-8") as f_out:
            f_out.write(f"共有 {len(invalid_groups)} 组不合规数据:\n\n")
            for g in invalid_groups:
                base, orgs, inds, outs = g
                f_out.write(f"======== 前缀: {base} ========\n")
                f_out.write(f"-> 原图({len(orgs)}张): {orgs}\n")
                f_out.write(f"-> indoor({len(inds)}张): {inds}\n")
                f_out.write(f"-> outdoor({len(outs)}张): {outs}\n\n")
                
        print(f"已将详细排查结果保存到本目录: {report_path}")
        
if __name__ == '__main__':
    main()
