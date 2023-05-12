from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import argparse
import json

#dataset_name=['ct80','IC13_857','IC15_1811','IIIT5K','svt','svtp']
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--ocr_path", type=str, default="./data/")
    parser.add_argument("--ocr_dataset", type=str, default="textVQA")
    parser.add_argument("--answers-file", type=str, default="./answers_textvqa")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    model_name = "Salesforce/blip2-opt-6.7b"
    device = "cuda"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model.to(device)
    ans_file = open(args.answers_file + '/' + args.ocr_dataset + '.jsonl', "w", encoding="utf-8")
    with open(args.ocr_path+args.ocr_dataset+'/TextVQA_0.5.1_val.json', 'r') as f:
        data = json.load(f)
        for i in range(len(data['data'])):
            prompt =  data['data'][i]['question']
            image_file = args.ocr_path+args.ocr_dataset+'/train_images/'+data['data'][i]['image_id']+'.jpg'
            question_id = data['data'][i]['question_id']
            gt_answers = data['data'][i]['answers']

            image = Image.open(image_file)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            ans_file.write(json.dumps({
                                        "image_path": image_file,
                                        "question_id": question_id,
                                        "prompt": prompt,
                                        "answer": generated_text,
                                        "gt_answers":gt_answers,
                                        "model_name":model_name}) + "\n")
            ans_file.flush()
        ans_file.close()