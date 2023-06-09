from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
import re
def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s
class ocrDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ocr",
        dataset_name = "ct80"
    ):
        self.image_dir_path = image_dir_path
        self.dataset_name = dataset_name
        file_path = os.path.join(image_dir_path, f'{dataset_name}/test_label.txt')
        file = open(file_path, "r")
        self.lines = file.readlines()
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        image_id = self.lines[idx].split()[0]
        img_path = os.path.join(self.image_dir_path,f'{self.dataset_name}/{image_id}')
        answers = self.lines[idx].split()[1]
        return {
            "image_path": img_path,
            "gt_answers": answers}
class IAMDataset(Dataset):
    def __init__(self, image_dir_path = './data/IAM') -> None:
        ann_path = image_dir_path + '/xml'
        self.images = []
        self.answers = []
        for filename in os.listdir(ann_path):
            if filename.endswith('.xml'):
                # 读取xml文件
                xml_file = os.path.join(ann_path, filename)
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # 对读取的xml文件进行操作
                # 例如，输出xml文件中的所有元素
                for word in root.iter('word'):
                    text = word.get('text')
                    img_id = word.get('id')
                    img_path = image_dir_path+'/'+filename.split('-')[0]+'/'+filename.split('.')[0]+'/'+img_id+'.png'
                    text = remove_special_chars(text)
                    if len(text)>0:
                        self.images.append(img_path)
                        self.answers.append(text)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        answers = self.answers[idx]
        return {
            "image_path": img_path,
            "gt_answers": answers}
class ReCTSDataset(Dataset):
    def __init__(
        self,
        dir_path= "./data/ReCTS",
    ):
        self.image_dir_path = os.path.join(dir_path, 'crops')
        file_path = os.path.join(dir_path, 'test_label.txt')
        file = open(file_path, "r")
        self.lines = file.readlines()
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        image_id = self.lines[idx].split()[0]
        img_path = os.path.join(self.image_dir_path, image_id)
        answers = self.lines[idx].split()[1]
        return {
            "image_path": img_path,
            "gt_answers": answers}
if __name__ == "__main__":
    '''data = IAMDataset('/home/zhangli/GPT4/MutimodelOCR/data/IAM')
    print(len(data))
    data = iter(data)
    batch = next(data)
    import pdb;pdb.set_trace()'''
    data = ReCTSDataset('/home/zhangli/GPT4/MutimodelOCR/data/ReCTS')
    print(len(data))
    data = iter(data)
    batch = next(data)
    print(batch)


