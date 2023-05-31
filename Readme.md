# On the Hidden Mystery of OCR in Large Multimodal Models
[paper link] https://arxiv.org/abs/2305.07895


Updating the instruction of evaluating large multimodal models on ocr tasks.

Feel free to open issues for any suggestion or comment.

# Results

Results are available in answer_save folder. 

![image](https://github.com/echo840/MultimodalOCR/assets/87795401/523e0421-7eca-4d15-89f1-3f7348321055)

Visualization results
![rvk](https://github.com/echo840/MultimodalOCR/assets/87795401/21982aba-d063-4a52-a045-8d16e0e98f71)


# Dataset Download
Text recognition datasets: [text recognition](https://pan.baidu.com/s/1Ba950d94u8RQmtqvkLBk-A) code:iwyn 

TextVQA, KIE and HME will be updated soon.

We assume that your symlinked `data` directory has the following structure:

```
data
|_ IC13_857
|_ IC15_1811
|_ ...
|_ ESTVQA
|_ TextVQA
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
python eval.py --model_name LLaVA --eval_TextVQA
```
The results will be saved at answer folder.
if you want to add a new model, 
