import sys
sys.path.append('./models/MiniGPT4')
sys.path.append('./models/mPLUG_Owl')
import argparse
#from models.BLIP2.BLIP2 import BLIP2
import more_itertools
from tqdm import tqdm
import datetime
import os
import json
import re
from datasets.vqa_dataset import textVQADataset, docVQADataset, ocrVQADataset, STVQADataset, ESTVQADataset
from datasets.ocr_dataset import ocrDataset, IAMDataset, ReCTSDataset
from datasets.kie_dataset import SROIEDataset,FUNSDDataset,POIEDataset
from datasets.formula_dataset import HMEDataset
from models.lavis.lavis import lavis
from models.LLaVA.LLaVA import LLaVA
from models.mPLUG_Owl.pipeline.mPLUG import mPLUG
from models.MiniGPT4.MiniGPT4 import MiniGPT4
from models.OpenFlamingo.OpenFlamingo import OpenFlamingo
from models.BLIP2.BLIP2 import BLIP2
from models.InstructBLIP.InstructBLIP import InstructBLIP
import torch
import numpy as np
def get_model(args):
    if args.model_name=='BLIP2':
        model = BLIP2("/home/zhangli/.cache/huggingface/hub/models--Salesforce--blip2-opt-6.7b/snapshots/f998da12f28eb37d7e7f080cfe3291d6d9d7e1fb", args.device)
        #model = lavis(args.BLIP2_model_name, args.BLIP2_model_type, args.device)
    elif args.model_name=='LLaVA':
        model = LLaVA(args.LLaVA_model_path, args.device)
    elif args.model_name=='MiniGPT4':
        model = MiniGPT4(args, args.device)
    elif args.model_name=='mPLUG':
        model = mPLUG(args.mPLUG_model_name, args.device)
    elif args.model_name=='OpenFlamingo':
        model = OpenFlamingo(args.llama_path, args.check_point, args.device)
    elif args.model_name == 'instructblip':
        model = InstructBLIP('blip2_vicuna_instruct',args.device)
    return model
def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False
def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s

class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, answer, gt_answers):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if type(gt_answers)==list:
            for i in range(len(gt_answers)):
                gt_answers[i] = gt_answers[i].replace("\n", " ")
                gt_answers[i] = gt_answers[i].replace("\t", " ")
                gt_answers[i] = gt_answers[i].strip()
                gt_answers[i] = self.processPunctuation(gt_answers[i])
                gt_answers[i] = self.processDigitArticle(gt_answers[i])
                if has_word(answer, gt_answers[i]):
                    return 1
            return 0
        else:
            gt_answers = gt_answers.replace("\n", " ")
            gt_answers= gt_answers.replace("\t", " ")
            gt_answers = gt_answers.strip()
            gt_answers = self.processPunctuation(gt_answers)
            gt_answers = self.processDigitArticle(gt_answers)
            if has_word(answer, gt_answers):
                return 1
            else:
                return 0

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText
def evaluate_VQA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=batch['question'])
        answer_dict={'question':batch['question'], 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    eval = VQAEval()
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer,gt_answers)==1:
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num
def evaluate_OCR(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    question='what is written in the image?',
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=question)
        answer_dict={'question':question, 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            gt_answers = remove_special_chars(gt_answers).lower()
            answer = remove_special_chars(answer).lower()
            if has_word(answer, gt_answers):
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num

def evaluate_ReCTS(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    #question='图像中的中文是什么？',
    question = 'What are the Chinese characters in the image?',
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=question)
        answer_dict={'question':question, 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            gt_answers = re.sub(r'[^\u4e00-\u9fa5\s]+', '', gt_answers)
            answer = re.sub(r'[^\u4e00-\u9fa5\s]+', '', answer)
            if gt_answers in answer:
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num

def evaluate_Formula(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    question='Please write out the expression of the formula in the image using LaTeX format.',
    batch_size=1,
    answer_path='./answers'
):
    #Please write out the expression of the formula in the image using LaTeX format.
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=question)
        answer_dict={'question':question, 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = re.sub(r'\s+', '', dict[i]['gt_answers'])
            answer = re.sub(r'\s+', '', dict[i]['answer'])
            if gt_answers in answer:
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num           

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    #OCR datasets
    parser.add_argument("--ocr_dir_path", type=str, default="./data")
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K svt IC13_857 IC15_1811 svtp ct80 cocotext ctw totaltext HOST WOST WordArt")
    #IAM
    parser.add_argument("--IAM_dir_path", type=str, default="./data/IAM")
    #ReCTS
    parser.add_argument("--ReCTS_dir_path", type=str, default="./data/ReCTS")

    #HME100k
    parser.add_argument("--HME_image_dir_path", type=str, default="./data/HME100K/test_images")
    parser.add_argument("--HME_ann_path", type=str, default="./data/HME100K/test_labels.txt")
    #textVQA
    parser.add_argument("--textVQA_image_dir_path", type=str, default="./data/textVQA/train_images")
    parser.add_argument("--textVQA_ann_path", type=str, default="./data/textVQA/TextVQA_0.5.1_val.json")

    #docVQA
    parser.add_argument("--docVQA_image_dir_path", type=str, default="./data/docVQA/val")
    parser.add_argument("--docVQA_ann_path", type=str, default="./data/docVQA/val/val_v1.0.json")

    #ocrVQA
    parser.add_argument("--ocrVQA_image_dir_path", type=str, default="./data/ocrVQA/images")
    parser.add_argument("--ocrVQA_ann_path", type=str, default="./data/ocrVQA/dataset.json")

    #STVQA
    parser.add_argument("--STVQA_image_dir_path", type=str, default="./data/STVQA")
    parser.add_argument("--STVQA_ann_path", type=str, default="./data/STVQA/train_task_3.json")
    #ESTVQA
    parser.add_argument("--ESTVQA_image_dir_path", type=str, default="./data/ESTVQA/images/train")
    parser.add_argument("--ESTVQA_CN_ann_path", type=str, default="./data/ESTVQA/annotations/cn_train.json")
    parser.add_argument("--ESTVQA_EN_ann_path", type=str, default="./data/ESTVQA/annotations/en_train.json")

    #SROIE
    parser.add_argument("--SROIE_dir_path", type=str, default="./data/SROIE")

    #FUNSD
    parser.add_argument("--FUNSD_dir_path", type=str, default="./data/FUNSD/testing_data/annotations")

    #POIE
    parser.add_argument("--POIE_dir_path", type=str, default="./data/POIE/test.txt")

    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_ocrVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )
    parser.add_argument(
        "--eval_STVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on STVQA."
    )
    parser.add_argument(
        "--eval_ESTVQA_CN",
        action="store_true",
        default=False,
        help="Whether to evaluate on ESTVQA_CN."
    )
    parser.add_argument(
        "--eval_ESTVQA_EN",
        action="store_true",
        default=False,
        help="Whether to evaluate on ESTVQA_EN."
    )
    parser.add_argument(
        "--eval_SROIE",
        action="store_true",
        default=False,
        help="Whether to evaluate on SROIE."
    )
    parser.add_argument(
        "--eval_FUNSD",
        action="store_true",
        default=False,
        help="Whether to evaluate on FUNSD."
    )
    parser.add_argument(
        "--eval_POIE",
        action="store_true",
        default=False,
        help="Whether to evaluate on POIE."
    )
    parser.add_argument(
        "--eval_HME",
        action="store_true",
        default=False,
        help="Whether to evaluate on HME100k."
    )
    parser.add_argument(
        "--eval_IAM",
        action="store_true",
        default=False,
        help="Whether to evaluate on IAM (handwritten)."
    )
    parser.add_argument(
        "--eval_ReCTS",
        action="store_true",
        default=False,
        help="Whether to evaluate on ReCTS (Chinese)."
    )
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="Whether to evaluate all datasets"
    )
    #BLIP2
    #parser.add_argument("--BLIP2_model_path", type=str, default="/home/zhangli/GPT4/BLIP2-flant5")
    parser.add_argument("--BLIP2_model_name", type=str, default="blip2_opt")#blip2_t5  blip2_opt blip2_vicuna_instruct
    parser.add_argument("--BLIP2_model_type", type=str, default="pretrain_opt6.7b")#pretrain_flant5xxl pretrain_opt6.7b vicuna13b
    #LLaVA
    parser.add_argument("--LLaVA_model_path", type=str, default="./models/LLaVA/model_weight")
    #miniGPT4
    parser.add_argument("--MiniGPT4_cfg_path", type=str, default="./models/MiniGPT4/eval_configs/minigpt4_eval.yaml")
    #mPLUG
    parser.add_argument("--mPLUG_model_name", type=str, default="MAGAer13/mplug-owl-llama-7b")
    #parser.add_argument("--mPLUG_tokenizer_path", type=str, default="./models/mPLUG_Owl/model_weight/tokenizer.model")
    #OpenFlamingo
    parser.add_argument("--llama_path", type=str, default="/home/zhangli/llama_models/llama/llama-7b")
    parser.add_argument("--check_point", type=str, default="/home/zhangli/code/open_flamingo/checkpoint/checkpoint.pt")

    parser.add_argument("--model_name", type=str, default="BLIP2")#mPLUG,miniGPT4,LLaVA
    parser.add_argument("--device", type=str, default="cuda:0")#2,3,7
    args = parser.parse_args()
    return args

def main(args):
    np.random.seed(0)
    max_sample_num = 5000
    model = get_model(args)
    '''ocr_dataset_name=['IIIT5K','svt','IC13_857','IC15_1811','svtp','ct80',
                  'cocotext','ctw','totaltext','HOST','WOST','WordArt']'''
    ocr_dataset_name = args.ocr_dataset_name.split()
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if args.eval_textVQA or args.eval_all:
        dataset = textVQADataset(args.textVQA_image_dir_path, args.textVQA_ann_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time)
        result['textVQA'] = acc
    if args.eval_docVQA or args.eval_all:
        dataset = docVQADataset(args.docVQA_image_dir_path, args.docVQA_ann_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'docVQA', time)
        result['docVQA'] = acc
    #Due to network issues, it's difficult to download the entire OCR-VQA dataset. 
    # Therefore, we will extract the first 5000 questions for testing.
    if args.eval_ocrVQA or args.eval_all:
        dataset = ocrVQADataset(args.ocrVQA_image_dir_path, args.ocrVQA_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'ocrVQA', time)
        result['ocrVQA'] = acc
    
    if args.eval_STVQA or args.eval_all:
        dataset = STVQADataset(args.STVQA_image_dir_path, args.STVQA_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'STVQA', time)
        result['STVQA'] = acc

    if args.eval_ESTVQA_CN or args.eval_all:
        dataset = ESTVQADataset(args.ESTVQA_image_dir_path, args.ESTVQA_CN_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'ESTVQA_CN', time)
        result['ESTVQA_CN'] = acc

    if args.eval_ESTVQA_EN or args.eval_all:
        dataset = ESTVQADataset(args.ESTVQA_image_dir_path, args.ESTVQA_EN_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'ESTVQA_EN', time)
        result['ESTVQA_EN'] = acc

    if args.eval_SROIE or args.eval_all:
        dataset = SROIEDataset(args.SROIE_dir_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'SROIE', time)
        result['SROIE'] = acc
    
    if args.eval_FUNSD or args.eval_all:
        dataset = FUNSDDataset(args.FUNSD_dir_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'FUNSD', time)
        result['FUNSD'] = acc
    if args.eval_POIE or args.eval_all:
        dataset = POIEDataset(args.POIE_dir_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'POIE', time)
        result['POIE'] = acc
    
    if args.eval_HME or args.eval_all:
        dataset = HMEDataset(args.HME_image_dir_path, args.HME_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_Formula(model, dataset, args.model_name, 'HME', time)
        result['HME'] = acc 
    if args.eval_IAM or args.eval_all:
        dataset = IAMDataset(args.IAM_dir_path)
        dataset = torch.utils.data.Subset(dataset, range(3000))
        acc = evaluate_OCR(model, dataset, args.model_name, 'IAM', time)
        result['IAM'] = acc
    if args.eval_ReCTS or args.eval_all:
        dataset = ReCTSDataset(args.ReCTS_dir_path)
        dataset = torch.utils.data.Subset(dataset, range(3000))
        acc = evaluate_ReCTS(model, dataset, args.model_name, 'ReCTS', time)
        result['ReCTS'] = acc   
    if args.eval_ocr or args.eval_all:
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(args.ocr_dir_path, ocr_dataset_name[i])
            acc = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time)
            result[ocr_dataset_name[i]] = acc
    result_path = os.path.join(os.path.join(args.answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))
if __name__ == "__main__":
    args = parse_args()
    main(args)