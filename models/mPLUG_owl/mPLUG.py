import argparse
import json
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from mplug_owl.configuration_mplug_owl import mPLUG_OwlConfig
from mplug_owl.modeling_mplug_owl import mPLUG_OwlForConditionalGeneration
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl.modeling_mplug_owl import ImageProcessor
from mplug_owl.tokenize_utils import tokenize_prompts
class mPLUG:
    def __init__(self, checkpoint_path=None, tokenizer_path=None) -> None:
        config = mPLUG_OwlConfig()
        self.model = mPLUG_OwlForConditionalGeneration(config=config).to(torch.bfloat16)
        self.model.eval()

        if checkpoint_path is not None:
            tmp_ckpt = torch.load(
                checkpoint_path, map_location='cpu')
            msg = self.model.load_state_dict(tmp_ckpt, strict=False)
            print(msg)

        assert tokenizer_path is not None
        self.tokenizer = LlamaTokenizer(
            tokenizer_path, pad_token='<unk>', add_bos_token=False)
        self.img_processor = ImageProcessor()
    def generate(self, image, question, max_length=512, top_k=1, do_sample=True, **generate_kwargs):
        prompts = [
        f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <image>
        Human: {question}
        AI: ''']
        tokens_to_generate = 0
        add_BOS = True
        context_tokens_tensor, context_length_tensorm, attention_mask = tokenize_prompts(
            prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS, tokenizer=self.tokenizer, ignore_dist=True)
        images = self.img_processor(image).to(torch.bfloat16).cuda()
        context_tokens_tensor = context_tokens_tensor.cuda()
        self.model.eval()
        with torch.no_grad():
            res = self.model.generate(input_ids=context_tokens_tensor, pixel_values=images,
                                attention_mask=attention_mask, max_lengt=max_length,top_k=top_k,do_sample=do_sample,**generate_kwargs)
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        return sentence
    