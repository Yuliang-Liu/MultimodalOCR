import pathlib
from argparse import ArgumentParser
import json
from tqdm import tqdm
import os
import sys
from http import HTTPStatus
from dashscope import MultiModalConversation
import time
# You should follow the instructions here befor strat: https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start   
OCRBench_score = {"Regular Text Recognition":0,"Irregular Text Recognition":0,"Artistic Text Recognition":0,"Handwriting Recognition":0,
"Digit String Recognition":0,"Non-Semantic Text Recognition":0,"Scene Text-centric VQA":0,"Doc-oriented VQA":0,
"Key Information Extraction":0,"Handwritten Mathematical Expression Recognition":0}
AllDataset_score = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}
num_all = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}
def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def call_with_local_file(img_path, question, model_name):
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """
    local_file_path1 = f'file://{img_path}'
    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': local_file_path1
            },
            {
                'text': question
            },
        ]
    }]
    response = MultiModalConversation.call(model=model_name, messages=messages)
    # time.sleep(2) #For qwenvl-max you may need to add this line to avoid the limits.
    print(response)
    return response['output']['choices'][0]["message"]['content'][0]['text']


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./OCRBench/OCRBench.json")
    parser.add_argument("--model", type=str, default="qwen-vl-max")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    if os.path.exists(os.path.join(args.output_path,f"{args.model}.json")):
        data_path = os.path.join(args.output_path,f"{args.model}.json")
    else:
        data_path = args.OCRBench_file
    with open(data_path, "r") as f:
        data = json.load(f)
    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]['image_path'])
        question = data[i]['question']
        if data[i].get("predict", 0)!=0:
            print(f"{img_path} predict exist, continue.")
            continue
        try:
            response = call_with_local_file(img_path, question, args.model)
            data[i]['predict'] = response
        except:
            print("QwenVL api failed")
        save_json(data, os.path.join(args.output_path,f"{args.model}.json"))
    for i in range(len(data)):
        data_type = data[i]["type"]
        dataset_name = data[i]["dataset_name"]
        answers = data[i]["answers"]
        if data[i].get('predict',0)==0:
            continue
        predict = data[i]['predict']
        data[i]['result'] = 0
        if dataset_name == "HME100k":
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n"," ").replace(" ","")
                    predict = predict.strip().replace("\n"," ").replace(" ","")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.strip().replace("\n"," ").replace(" ","")
                predict = predict.strip().replace("\n"," ").replace(" ","")
                if answers in predict:
                    data[i]['result'] = 1
        else:
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n"," ")
                    predict = predict.lower().strip().replace("\n"," ")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.lower().strip().replace("\n"," ")
                predict = predict.lower().strip().replace("\n"," ")
                if answers in predict:
                    data[i]['result'] = 1
    save_json(data, os.path.join(args.output_path,f"{args.model}.json"))
    for i in range(len(data)):
        if data[i].get("result",100)==100:
            continue
        OCRBench_score[data[i]['type']] += data[i]['result']
    recognition_score = OCRBench_score['Regular Text Recognition']+OCRBench_score['Irregular Text Recognition']+OCRBench_score['Artistic Text Recognition']+OCRBench_score['Handwriting Recognition']+OCRBench_score['Digit String Recognition']+OCRBench_score['Non-Semantic Text Recognition']
    Final_score = recognition_score+OCRBench_score['Scene Text-centric VQA']+OCRBench_score['Doc-oriented VQA']+OCRBench_score['Key Information Extraction']+OCRBench_score['Handwritten Mathematical Expression Recognition']
    print("###########################OCRBench##############################")
    print(f"Text Recognition(Total 300):{recognition_score}")
    print("------------------Details of Recognition Score-------------------")
    print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
    print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
    print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
    print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
    print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
    print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
    print("----------------------------------------------------------------")
    print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
    print("----------------------------------------------------------------")
    print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
    print("----------------------------------------------------------------")
    print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
    print("----------------------------------------------------------------")
    print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
    print("----------------------Final Score-------------------------------")
    print(f"Final Score(Total 1000): {Final_score}")
