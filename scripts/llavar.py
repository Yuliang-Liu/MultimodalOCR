import json
from argparse import ArgumentParser
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, Queue, Manager

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava import conversation as conversation_lib
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from PIL import Image,ImageOps
# https://github.com/SALT-NLP/LLaVAR/blob/main/LLaVA/llava/eval/model_vqa.py

def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    image = image.resize((new_width, new_height))
    width_diff = target_size[0] - image.size[0]
    height_diff = target_size[1] - image.size[1]
    left_padding = 0
    top_padding = 0
    right_padding = width_diff - left_padding
    bottom_padding = height_diff - top_padding
    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=0)
    return padded_image

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

def split_list(lst, n):
    length = len(lst)
    avg = length // n  # 每份的大小
    result = []  # 存储分割后的子列表
    for i in range(n - 1):
        result.append(lst[i*avg:(i+1)*avg])
    result.append(lst[(n-1)*avg:])
    return result

def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./OCRBench_Images")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./OCRBench/OCRBench.json")
    parser.add_argument("--model_path", type=str, default="./model_weights/LLaVar")
    parser.add_argument("--save_name", type=str, default="llavar")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    return args
OCRBench_score = {"Regular Text Recognition":0,"Irregular Text Recognition":0,"Artistic Text Recognition":0,"Handwriting Recognition":0,
"Digit String Recognition":0,"Non-Semantic Text Recognition":0,"Scene Text-centric VQA":0,"Doc-oriented VQA":0,"Doc-oriented VQA":0,
"Key Information Extraction":0,"Handwritten Mathematical Expression Recognition":0}
AllDataset_score = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}
num_all = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}

def eval_worker(args, data, eval_id, output_queue):
    print(f"Process {eval_id} start.")
    device = f"cuda:{eval_id}"
    disable_torch_init()
    model_name = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device=device, dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).to(device)
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.to(device).half()
        model.model.vision_tower = [vision_tower]
    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]['image_path'])
        qs = data[i]['question']
        # qs = qs+"\nAnswer the question using a single word or phrase."
        if data[i].get("predict", 0)!=0:
            print(f"{img_path} predict exist, continue.")
            continue
        if mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        if args.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'
        # conv = default_conversation.copy()
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        # modified
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        image = Image.open(img_path)
        # if "REval" in args.image_folder:
        image = resize_image(image, (336, 336))

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = torch.as_tensor(inputs.input_ids).to(device)

        # new stopping implementation
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.tokenizer = tokenizer
                self.start_len = None
                self.input_ids = input_ids

            def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.start_len is None:
                    self.start_len = self.input_ids.shape[1]
                else:
                    outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        if keyword in outputs:
                            return True
                return False
        
        # keywords = ['###']
        # modified
        keywords = ['</s>']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                do_sample=False,
                temperature=0,
                max_new_tokens=200,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        
        # modified
        if args.conv_mode == 'simple_legacy' or args.conv_mode == 'simple':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

        if conv.sep_style == conversation_lib.SeparatorStyle.TWO:
            sep = conv.sep2
        else:
            sep = conv.sep

        try:
            index = outputs.index(sep)
        except ValueError:
            outputs += sep
            index = outputs.index(sep)
        outputs = outputs[:index].strip()
        data[i]['predict'] = outputs
    output_queue.put({eval_id: data})
    print(f"Process {eval_id} has completed.")

if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    args = _get_args()
    
    if os.path.exists(os.path.join(args.output_folder,f"{args.save_name}.json")):
        data_path = os.path.join(args.output_folder,f"{args.save_name}.json")
        print(f"output_path:{data_path} exist! Only generate the results that were not generated in {data_path}.")
    else:
        data_path = args.OCRBench_file

    with open(data_path, "r") as f:
        data = json.load(f)
    
    data_list = split_list(data, args.num_workers)
    output_queue = Manager().Queue()

    pool = Pool(processes=args.num_workers)
    for i in range(len(data_list)):
        pool.apply_async(eval_worker, args=(args, data_list[i], i, output_queue))
    pool.close()
    pool.join()

    results = {}
    while not output_queue.empty():
        result = output_queue.get()
        results.update(result)
    data = []
    for i in range(len(data_list)):
        data.extend(results[i])

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
    save_json(data, os.path.join(args.output_folder,f"{args.save_name}.json"))
    if len(data)==1000:
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
    else:
        for i in range(len(data)):
            num_all[data[i]['dataset_name']] += 1
            if data[i].get("result",100)==100:
                continue
            AllDataset_score[data[i]['dataset_name']] += data[i]['result']
        for key in AllDataset_score.keys():
            print(f"{key}: {AllDataset_score[key]/float(num_all[key])}")
