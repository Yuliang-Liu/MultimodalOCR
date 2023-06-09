import re
import os
def has_chinese_characters(string):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(string))
def is_all_chinese(text):
    """
    判断一个字符串是否仅仅包含中文
    """
    pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
    return pattern.match(text) is not None
if __name__ =='__main__':
    file_path = "/home/zhangli/GPT4/MutimodelOCR/data/ReCTS/annotation.txt"
    out = open("/home/zhangli/GPT4/MutimodelOCR/data/ReCTS/ann.txt",'w')
    with open(file_path, 'r') as file:
        data = file.readlines()
        for line in data:
            text = line.strip().split()[1]
            path = os.path.join("/home/zhangli/GPT4/MutimodelOCR/data/ReCTS/crops",line.strip().split()[0])
            if is_all_chinese(text) and os.path.exists(path):
                out.write(line.strip())
                out.write('\n')
    out.close()