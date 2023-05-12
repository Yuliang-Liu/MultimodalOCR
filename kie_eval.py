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
    parser.add_argument("--ocr_path", type=str, default="/path/to/GPT4/KIE_data/")
    parser.add_argument("--ocr_dataset", type=str, default="FUNSD")
    parser.add_argument("--answers-file", type=str, default="/path/to/GPT4/KIE_data/cmr/blip2answer")
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
    # prompt = "Question: what is written in the image? Answer:"
    ans_file = open(args.answers_file + '/' + args.ocr_dataset + '.jsonl', "w+", encoding="utf-8")
    with open(args.ocr_path+ args.ocr_dataset +'.txt', 'r') as file:
        for line in file:
            image_file = line.split()[0]
            
            question_start = line.find('question:') + len('question:')
            label_start = line.find('label:') + len('label:')
            question = line[question_start:line.find('label:')].strip()
            label = line[label_start:].strip()
            prompt = f"What is the '{question}' information written in this image ?"
            
            img_path = os.path.join(args.ocr_path+args.ocr_dataset, image_file)
            image = Image.open(img_path)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            ans_file.write(json.dumps({"prompt": prompt,
                                        "image_path": image_file,
                                        "label": label,
                                        "text": generated_text,
                                        "model_name":model_name}) + "\n")
            ans_file.flush()
        ans_file.close()