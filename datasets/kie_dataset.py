from torch.utils.data import Dataset
import os
import json
class SROIEDataset(Dataset):
    def __init__(
        self,
        dir_path= "./data/SROIE",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".txt") and '(' not in file_name:
                file_path = os.path.join(dir_path, file_name)
                img_path = file_path.replace('.txt', '.jpg')
                with open(file_path) as f:
                    content = f.read()
                    info = json.loads(content)
                    if 'company' in info.keys():
                        self.question_list.append("what is the name of the company that issued this invoice?")#llava 0.12
                        #self.question_list.append("what is the company information in the image?")#llava 0.08
                        self.answer_list.append(info['company'])
                        self.image_list.append(img_path)
                    if 'date' in info.keys():
                        self.question_list.append("when was this invoice issued?")
                        #self.question_list.append("what is the date information in the image?")
                        self.answer_list.append(info['date'])
                        self.image_list.append(img_path)

                    if 'address' in info.keys():
                        self.question_list.append("where was this invoice issued?")
                        #self.question_list.append("what is the address information in the image?")
                        self.answer_list.append(info['address'])
                        self.image_list.append(img_path)

                    if 'total' in info.keys():
                        self.question_list.append("what is the total amount of this invoice?")
                        #self.question_list.append("what is the total information in the image?")
                        self.answer_list.append(info['total'])
                        self.image_list.append(img_path)
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
        
class FUNSDDataset(Dataset):
    def __init__(self, ann_dir_path= "./data/FUNSD/testing_data/annotations"):
        questions = []
        answers = []
        images = []
        for file_name in os.listdir(ann_dir_path):
            file_path = os.path.join(ann_dir_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)['form']
                #去除空的linking
                json_data = [d for d in json_data if "linking" in d and len(d["linking"])>0]
                question_list = [d for d in json_data if d.get('label') == 'question']
                answer_list = [d for d in json_data if d.get('label') == 'answer']
                
                for i in range(len(question_list)):
                    link = question_list[i]['linking']
                    gt_answer = ""
                    for j in range(len(link)):
                        for k in range(len(answer_list)):
                            if answer_list[k]['id'] == link[j][1]:
                                if len(gt_answer)>0:
                                    gt_answer = gt_answer + ' ' + answer_list[k]['text']
                                else:
                                    gt_answer = gt_answer + answer_list[k]['text']
                    if len(gt_answer)>0:
                        questions.append(f"what is \"{question_list[i]['text']}\" information in the image?")
                        answers.append(gt_answer)
                        images.append(file_path.replace('annotations','images').replace('.json','.png'))
        self.questions = questions
        self.answers = answers
        self.images = images
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        question = self.questions[idx]
        answers = self.answers[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
class POIEDataset(Dataset):
    def __init__(
        self,
        dir_path= "./data/POIE/test.txt",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        with open(dir_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                import pdb;pdb.set_trace()
                dict = json.loads(line)
                for key, value in dict['entity_dict'].items():
                    self.image_list.append(dir_path.replace("test.txt", dict['file_name']))
                    self.question_list.append(key)
                    self.answer_list.append(value)
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
if __name__ == "__main__":
    data = POIEDataset("/home/zhangli/GPT4/MutimodelOCR/data/POIE/test.txt")
    data = iter(data)
    batch = next(data)