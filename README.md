# On the Hidden Mystery of OCR in Large Multimodal Models 
<img src="./images/all_data.png" width="96%" height="96%">

> Large models have recently played a dominant role in natural language processing and multimodal vision-language learning. However, their effectiveness in text-related visual tasks remains relatively unexplored. In this paper, we  conducted a comprehensive evaluation of Large Multimodal Models, such as GPT4V and Gemini, in various text-related visual tasks including Text Recognition, Scene Text-Centric Visual Question Answering (VQA), Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER). To facilitate the assessment of Optical Character Recognition (OCR) capabilities in Large Multimodal Models, we propose OCRBench, a comprehensive evaluation benchmark. Our study encompasses 29 datasets, making it the most comprehensive OCR evaluation benchmark available. Furthermore, our study reveals both the strengths and weaknesses of these models, particularly in handling multilingual text, handwritten text, non-semantic text, and mathematical expression recognition. Most importantly, the baseline results showcased in this study could provide a foundational framework for the conception and assessment of innovative strategies targeted at enhancing zero-shot multimodal techniques.

**[Project Page [This Page]](https://github.com/Yuliang-Liu/MultimodalOCR)** | **[Paper](https://arxiv.org/abs/2305.07895)** | **[OCRBench Leaderboard](http://27.18.7.167:7682/)** |

# Data
| Data | Link | Description |
| --- | --- | --- |
| Full Test Json | [Full Test](./OCRBench/FullTest.json) | This file contains the test data used in Table 1 and Table 2 from [Paper](https://arxiv.org/abs/2305.07895). |
| OCRBench Json | [OCRBench](./OCRBench/OCRBench.json) | This file contains the test data in OCRBench used in Table3 from [Paper](https://arxiv.org/abs/2305.07895). |
| All Test Images |[All Images](https://drive.google.com/file/d/1U5AtLoJ7FrJe9yfcbssfeLmlKb7dTosc/view?usp=drive_link) | This file contains all the testing images used in [Paper](https://arxiv.org/abs/2305.07895), including OCRBench Images.|
| OCRBench Images | [OCRBench Images](https://drive.google.com/file/d/1a3VRJx3V3SdOmPr7499Ky0Ug8AwqGUHO/view?usp=drive_link) | This file only contains the images used in OCRBench. |
| Test Results | [Test Results](https://drive.google.com/drive/folders/15XlHCuNTavI1Ihqm4G7u3J34BHpkaqyE?usp=drive_link) | This file file contains the result files for the test models. |

# OCRBench

OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models. It comprises five components: Text Recognition, SceneText-Centric VQA, Document-Oriented VQA, Key Information Extraction, and Handwritten Mathematical Expression Recognition. The benchmark includes 1000 question-answer pairs, and all the answers undergo manual verification and correction to ensure a more precise evaluation. 

You can find the results of Large Multimodal Models in **[OCRBench Leaderboard](http://27.18.7.167:7682/)**, if you would like to include your model in the OCRBench leaderboard, please follow the evaluation instructions provided below and feel free to contact us via email at zhangli123@hust.edu.cn. We will update the leaderboard in time.

<img src="./images/GPT4V_Gemini.png" width="96%" height="96%">

# Evaluation
The test code for evaluating models in the paper can be found in [scripts](./scripts). If you want to evaluate other models, please edit the "TODO" things in [example](./example.py).

Example evaluation scripts:
```python

python ./scripts/monkey.py --image_folder ./data --OCRBench_file ./OCRBench/OCRBench.json --save_name Monkey_OCRBench --num_workers GPU_Nums # Test on OCRBench
python ./scripts/monkey.py --image_folder ./data --OCRBench_file ./OCRBench/FullTest.json --save_name Monkey_FullTest --num_workers GPU_Nums # Full Test

```

# Citation
If you wish to refer to the baseline results published here, please use the following BibTeX entries:
```BibTeX
@misc{liu2024hidden,
      title={On the Hidden Mystery of OCR in Large Multimodal Models}, 
      author={Yuliang Liu and Zhang Li and Biao Yang and Chunyuan Li and Xucheng Yin and Cheng-lin Liu and Lianwen Jin and Xiang Bai},
      year={2024},
      eprint={2305.07895},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



