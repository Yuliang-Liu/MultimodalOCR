import json
import os
import re
def has_chinese_characters(string):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(string))
if __name__ == "__main__":
    ann_file = "/home/zhangli/OCRData/data/TextVQA/ESTVQA/annotations/train.json"
    #img_file = "/home/zhangli/GPT4/MutimodelOCR/data/ESTVQA/images/train"
    cn_list = []
    en_list= []
    with open(ann_file,'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            if has_chinese_characters(data[i]['annotation'][0]['question']):
                cn_list.append(data[i])
            else:
                en_list.append(data[i])
    with open('./cn_train.json', 'w', encoding='utf-8') as f:
        json.dump(cn_list, f, ensure_ascii=False)
    with open('./en_train.json', 'w', encoding='utf-8') as f:
        json.dump(en_list, f, ensure_ascii=False)
    
