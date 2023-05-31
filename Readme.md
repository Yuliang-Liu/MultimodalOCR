# On the Hidden Mystery of OCR in Large Multimodal Models
[paper link] https://arxiv.org/abs/2305.07895

Updating the instruction of evaluating large multimodal models on ocr tasks.

Feel free to open issues for any suggestion or comment.

# Results

Results are available in answer_save folder. 

![image](https://github.com/echo840/MultimodalOCR/assets/87795401/523e0421-7eca-4d15-89f1-3f7348321055)

Visualization results

![rvk](https://github.com/echo840/MultimodalOCR/assets/87795401/21982aba-d063-4a52-a045-8d16e0e98f71)


# Data Download
| Data file | Size |
| --- | ---: |
|[text recognition](https://pan.baidu.com/s/1Ba950d94u8RQmtqvkLBk-A) code:iwyn | 1.37GB |

TextVQA, KIE and HME will be updated soon.

We assume that your symlinked `data` directory has the following structure:

```
data
|_ IC13_857
|_ IC15_1811
|_ ...
|_ ESTVQA
|_ textVQA
|_ ...
|_ FUNSD
|_ POIE
```


# Usage

eval on all datasets
```Shell
python eval.py --model_name LLaVA --eval_all
```

eval on one dataset
```Shell
python eval.py --model_name LLaVA --eval_textVQA
```
```Shell
python eval.py --model_name LLaVA --eval_ocr --ocr_dataset_name "ct80 IIIT5K"
```
The results will be saved at answer folder.

If you want to add a new model, please write its inference function under the folder "models", and update the get_model function in eval.py. An example inference code is as follows：

```Shell
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
```

# Related Projects
- [LLaVA](https://github.com/haotian-liu/LLaVA.git)
- [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4.git)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl.git)
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo.git)
- [Lavis](https://github.com/salesforce/LAVIS.git)
