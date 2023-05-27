from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from ..process import pad_image, resize_image
from PIL import Image
import re
def postprocess_vqa_generation(predictions):
    return re.split("Question|Answer", predictions, 1)[0]
class OpenFlamingo:
    def __init__(self, llama_path, check_point, device) -> None:
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path = llama_path,
            tokenizer_path = llama_path,
            cross_attn_every_n_layers=4
        )
        checkpoint = torch.load(check_point, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        #checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
        self.model = model.to(device)
        self.image_processor=image_processor
        self.tokenizer = tokenizer
        self.device = device
    def generate(self, image, question, name="resize"):
        self.tokenizer.padding_side = "left"
        lang_x = self.tokenizer(
        [f"<image>Question:{question} Answer:"],
        return_tensors="pt",
        ).to(self.device)
        len_input =  len(lang_x['input_ids'][0])
        image = Image.open(image)
        if name == "resize":
            image = resize_image(image, (224,224))
        vision_x = [self.image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)
        generated_text = self.model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=48,
        num_beams=3,
        )
        answer = self.tokenizer.decode(generated_text[0][len_input:], skip_special_tokens=True)
        '''process_function = (
            postprocess_vqa_generation)
        new_predictions = [
            process_function(out)
            for out in self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
        ]'''
        return answer
        