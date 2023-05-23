from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from PIL import Image
from ..process import pad_image, resize_image
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
class LLaVA:
    def __init__(self, model_path, device) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        patch_config(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device = device, dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    def generate(self, image, question, name = 'resize'):
        #llava   textVQA none 0.32   pad  0.25   resize 30.4    ct80  none 29.5   pad 63.9    resize  61.5  
        qs = question + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
        conv = conv_templates['simple'].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
        image = Image.open(image)
        if name == "pad":
            image = pad_image(image, (224,224))
        elif name == "resize":
            image = resize_image(image, (224,224))
            
        
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                do_sample=True,
                temperature=0.9,
                max_new_tokens=256,
                stopping_criteria=[stopping_criteria])
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break
            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
        return outputs