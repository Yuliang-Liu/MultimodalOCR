import torch
import numpy as np
import requests
from PIL import Image, ImageOps
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # 宽度大于高度，以宽度为基准进行 resize
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        # 高度大于宽度，以高度为基准进行 resize
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


def get_model(pretrained_ckpt, use_bf16=False):
    """Model Provider with tokenizer and processor. 

    Args:
        pretrained_ckpt (string): The path to pre-trained checkpoint.
        use_bf16 (bool, optional): Whether to use bfloat16 to load the model. Defaults to False.

    Returns:
        model: MplugOwl Model
        tokenizer: MplugOwl text tokenizer
        processor: MplugOwl processor (including text and image)
    """
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.half,
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    return model, tokenizer, processor


def do_generate(prompts, image_list, model, tokenizer, processor, use_bf16=False, **generate_kwargs):
    """The interface for generation

    Args:
        prompts (List[str]): The prompt text
        image_list (List[str]): Paths of images
        model (MplugOwlForConditionalGeneration): MplugOwlForConditionalGeneration
        tokenizer (MplugOwlTokenizer): MplugOwlTokenizer
        processor (MplugOwlProcessor): MplugOwlProcessor
        use_bf16 (bool, optional): Whether to use bfloat16. Defaults to False.

    Returns:
        sentence (str): Generated sentence.
    """
    inputs = processor(text=prompts, images=image_list, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence
class mPLUG:
    def __init__(self, base_model, device) -> None:
        model, tokenizer, processor = get_model(base_model, use_bf16=True)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.processor = processor
    def generate(self, image, question,name='resize'):
        prompts = [f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: {question}
AI: ''']
        image = Image.open(image)
        #ct80 none 0.3229166666666667 resize  0.8159722222222222
        if name == 'resize':
            image = resize_image(image,(224,224))
        image_list=[image]
        sentence = do_generate(
        prompts, image_list, self.model, 
        self.tokenizer, self.processor, use_bf16=True,
        max_length=512, top_k=1, do_sample=True)
        return sentence
  