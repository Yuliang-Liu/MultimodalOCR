from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from ..process import pad_image
#There are some issues with the Hugging Face version of the BLIP2-opt model.
class BLIP2:
    def __init__(self, model_path, device = "cuda") -> None:
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16).to(device)
        self.model.eval()
        self.device = device
    def generate(self, image, question, pad=True):
        prompt =f'Question: {question} Answer:'
        image = Image.open(image)
        if pad:
            image = pad_image(image, (224,224))
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text