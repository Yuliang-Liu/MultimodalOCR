# OCRBench & OCRBench v2

**This is the repository of the [OCRBench](./OCRBench/README.md) & [OCRBench v2](./OCRBench_v2/README.md).**

**OCRBench** is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models. It comprises five components: Text Recognition, SceneText-Centric VQA, Document-Oriented VQA, Key Information Extraction, and Handwritten Mathematical Expression Recognition. The benchmark includes 1000 question-answer pairs, and all the answers undergo manual verification and correction to ensure a more precise evaluation. More details can be found in [OCRBench README](./OCRBench/README.md).

<p align="center">
  <img src="./OCRBench/images/all_data.png" width="88%" height="80%">
</p>

**OCRBench v2** is a large-scale bilingual text-centric benchmark with currently the most comprehensive set of tasks (4× more tasks than the previous multi-scene benchmark OCRBench), the widest coverage of scenarios (31 diverse scenarios including street scene, receipt, formula, diagram, and so on), and thorough evaluation metrics, with a total of 10, 000 human-verified question-answering pairs and a high proportion of difficult samples. More details can be found in [OCRBench v2 README](./OCRBench_v2/README.md).

<p align="center">
    <img src="https://v1.ax1x.com/2024/12/30/7VhCnP.jpg" width="88%" height="80%">
<p>

# News 
* ```2024.12.31``` 🚀 [OCRBench v2](./OCRBench_v2/README.md) is released.
* ```2024.12.11``` 🚀 OCRBench has been accepted by [Science China Information Sciences](https://link.springer.com/article/10.1007/s11432-024-4235-6).
* ```2024.5.19 ``` 🚀 We realese [DTVQA](https://github.com/ShuoZhang2003/DT-VQA), to explore the Capabilities of Large Multimodal Models on Dense Text.
* ```2024.5.01 ``` 🚀 Thanks to [SWHL](https://github.com/Yuliang-Liu/MultimodalOCR/issues/29) for releasing [ChineseOCRBench](https://huggingface.co/datasets/SWHL/ChineseOCRBench).
* ```2024.3.26 ``` 🚀 OCRBench is now supported in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
* ```2024.3.12 ``` 🚀 We plan to construct OCRBench v2 to include more ocr tasks and data. Any contribution will be appreciated.
* ```2024.2.25 ``` 🚀 OCRBench is now supported in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).


# Other Related Multilingual Datasets
| Data | Link | Description |
| --- | --- | --- |
| EST-VQA Dataset (CVPR 2020, English and Chinese) | [Link](https://github.com/xinke-wang/EST-VQA) | On the General Value of Evidence, and Bilingual Scene-Text Visual Question Answering. |
| Swahili Dataset (ICDAR 2024) | [Link](https://arxiv.org/abs/2405.11437) | The First Swahili Language Scene Text Detection and Recognition Dataset. |
| Urdu Dataset (ICDAR 2024) | [Link](https://arxiv.org/abs/2405.12533) | Dataset and Benchmark for Urdu Natural Scenes Text Detection, Recognition and Visual Question Answering. |
| MTVQA (9 languages) | [Link](https://arxiv.org/abs/2405.11985) | MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering. |
| EVOBC (Oracle Bone Script Evolution Dataset) | [Link](https://arxiv.org/abs/2401.12467) | We systematically collected ancient characters from authoritative texts and websites spanning six historical stages. |
| HUST-OBC (Oracle Bone Script Character Dataset) | [Link](https://arxiv.org/abs/2401.15365) | For deciphering oracle bone script characters. |

# Citation
If you wish to refer to the baseline results published here, please use the following BibTeX entries:
```BibTeX
@article{Liu_2024,
    title={OCRBench: on the hidden mystery of OCR in large multimodal models},
    volume={67},
    ISSN={1869-1919},
    url={http://dx.doi.org/10.1007/s11432-024-4235-6},
    DOI={10.1007/s11432-024-4235-6},
    number={12},
    journal={Science China Information Sciences},
    publisher={Springer Science and Business Media LLC},
    author={Liu, Yuliang and Li, Zhang and Huang, Mingxin and Yang, Biao and Yu, Wenwen and Li, Chunyuan and Yin, Xu-Cheng and Liu, Cheng-Lin and Jin, Lianwen and Bai, Xiang},
    year={2024},
    month=dec }
  
@misc{fu2024ocrbenchv2improvedbenchmark,
    title={OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning}, 
    author={Ling Fu and Biao Yang and Zhebin Kuang and Jiajun Song and Yuzhe Li and Linghao Zhu and Qidi Luo and Xinyu Wang and Hao Lu and Mingxin Huang and Zhang Li and Guozhi Tang and Bin Shan and Chunhui Lin and Qi Liu and Binghong Wu and Hao Feng and Hao Liu and Can Huang and Jingqun Tang and Wei Chen and Lianwen Jin and Yuliang Liu and Xiang Bai},
    year={2024},
    eprint={2501.00321},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2501.00321}, 
}
```



