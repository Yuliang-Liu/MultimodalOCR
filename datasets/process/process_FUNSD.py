import json
import os
if __name__ == "__main__":
    ann_dir_path = '/home/zhangli/GPT4/MutimodelOCR/data/FUNSD/training_data/annotations'
    questions = []
    answers = []
    for file_name in os.listdir(ann_dir_path):
        file_path = os.path.join(ann_dir_path, file_name)
        with open(file_path, 'r') as f:
            json_data = json.load(f)['form']
            #去除空的linking
            json_data = [d for d in json_data if "linking" in d and len(d["linking"])>0]
            question_list = [d for d in json_data if d.get('label') == 'question']
            answer_list = [d for d in json_data if d.get('label') == 'answer']
            unique_question_list = [d for i, d in enumerate(question_list) if d['text'] not in [x['text'] for x in json_data[:i]]]
            for i in range(len(unique_question_list)):
                link = unique_question_list[i]['linking']
                gt_answer = ""
                for j in range(len(link)):
                    for k in range(len(answer_list)):
                        if answer_list[k]['id'] == link[j][1]:
                            gt_answer = gt_answer + answer_list[k]['text']
                questions.append(unique_question_list[i]['text'])
                answers.append(gt_answer)
                    