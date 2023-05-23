from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="<path to llama weights in HuggingFace format>",
    tokenizer_path="<path to llama tokenizer in HuggingFace format>",
    cross_attn_every_n_layers=4
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)