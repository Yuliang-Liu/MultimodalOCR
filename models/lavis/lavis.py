import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from ..process import pad_image, resize_image
class lavis:
    def __init__(self, model_name, model_type, device) -> None:
        model, vis_processors, txt_processors = load_model_and_preprocess(name = model_name, model_type = model_type, is_eval=True, device=device)
        self.model_name = model_name
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.device = device
    def generate(self, image, question, name='resize'):
        if 'opt' in self.model_name:
            prompt = f'Question: {question} Answer:'
        elif 't5' in self.model_name:
            prompt = f'Question: {question} Short answer:'
        else:
            prompt = f'Question: {question} Answer:'
        image = Image.open(image).convert("RGB")
        if name == "pad":
            image = pad_image(image, (224,224))
        elif name == "resize":
            image = resize_image(image, (224,224))
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        prompt = self.txt_processors["eval"](prompt)
        answer = self.model.predict_answers(samples={"image": image, "text_input": prompt}, inference_method="generate", max_len=48, min_len=1)[0]
        return answer