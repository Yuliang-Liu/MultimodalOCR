<h1 align="center">
MDPBench: A Benchmark for Multilingual Document Parsing in Real-World Scenarios
</h1>

<div align="center">
English | <a href="./README_zh-CN.md">简体中文</a>
<br>

[\[📜 arXiv\]](https://arxiv.org/abs/2603.28130) | [[Dataset (🤗Hugging Face)]](https://huggingface.co/datasets/Delores-Lin/MDPBench) | [[Source Code]](https://github.com/Yuliang-Liu/MultimodalOCR)

</div>
We introduce Multilingual Document Parsing Benchmark, the first benchmark for multilingual digital and photographed document parsing. Document parsing has made remarkable strides, yet almost exclusively on clean, digital, well-formatted pages in a handful of dominant languages. No systematic benchmark exists to evaluate how models perform on digital and photographed documents across diverse scripts and low-resource languages. MDPBench comprises 3,400 document images spanning 17 languages (Simplified Chinese, Traditional Chinese, English, Arabic, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Russian, Thai, Vietnamese), diverse scripts, and varied photographic conditions, with high-quality annotations produced through a rigorous pipeline of expert model labeling, manual correction, and human verification. To ensure fair comparison and prevent data leakage, we maintain separate public and private evaluation splits. Our comprehensive evaluation of both open-source and closed-source models uncovers a striking finding: while closed-source models (notably Gemini3-Pro) prove relatively robust, open-source alternatives suffer dramatic performance collapse, particularly on non-Latin scripts and real-world photographed documents, with an average drop of 17.8% on photographed documents and 14.0% on non-Latin scripts. These results reveal significant performance imbalances across languages and conditions, and point to concrete directions for building more inclusive, deployment-ready parsing systems.



## Main Results

<table style="width:100%; border-collapse: collapse; text-align: center;">
    <caption>Performance of general VLMs, specialized VLMs, and pipeline tools on MDPBench.</caption>
    <thead>
        <tr>
            <th rowspan="2">Model Type</th>
            <th rowspan="2">Model</th>
            <th colspan="3">Overall</th>
            <th colspan="10">Latin</th>
            <th colspan="9">Non-Latin</th>
            <th colspan="1">Private</th>
        </tr>
        <tr>
            <th>All</th>
            <th>Digit.</th>
            <th>Photo.</th>
            <th>Avg.</th>
            <th>DE</th>
            <th>EN</th>
            <th>ES</th>
            <th>FR</th>
            <th>ID</th>
            <th>IT</th>
            <th>NL</th>
            <th>PT</th>
            <th>VI</th>
            <th>Avg.</th>
            <th>AR</th>
            <th>HI</th>
            <th>JP</th>
            <th>KO</th>
            <th>RU</th>
            <th>TH</th>
            <th>ZH</th>
            <th>ZH-T</th>
            <th>All</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="8"><strong>General</strong><br><strong>VLMs</strong></td>
            <td>Gemini-3-pro-preview</td>
            <td><strong>86.4</strong></td>
            <td><ins>90.4</ins></td>
            <td><strong>85.1</strong></td>
            <td><strong>88.4</strong></td>
            <td><strong>91.2</strong></td>
            <td><strong>90.6</strong></td>
            <td><strong>83.4</strong></td>
            <td><strong>82.7</strong></td>
            <td><strong>91.5</strong></td>
            <td><strong>91.6</strong></td>
            <td><strong>87.7</strong></td>
            <td><strong>91.4</strong></td>
            <td><ins>85.9</ins></td>
            <td><strong>84.1</strong></td>
            <td><strong>89.4</strong></td>
            <td><strong>90.4</strong></td>
            <td><ins>74.8</ins></td>
            <td><ins>85.5</ins></td>
            <td><strong>84.9</strong></td>
            <td><strong>80.6</strong></td>
            <td><strong>85.1</strong></td>
            <td><strong>82.1</strong></td>
            <td><strong>89.8</strong></td>
        </tr>
        <tr>
            <td>kimi-K2.5</td>
            <td>77.5</td>
            <td>85.0</td>
            <td>75.0</td>
            <td>81.6</td>
            <td><ins>85.9</ins></td>
            <td>86.2</td>
            <td>72.7</td>
            <td>71.0</td>
            <td>80.6</td>
            <td>86.6</td>
            <td>77.4</td>
            <td>87.6</td>
            <td><strong>86.2</strong></td>
            <td>72.9</td>
            <td>75.8</td>
            <td>74.5</td>
            <td>72.5</td>
            <td>70.9</td>
            <td>61.8</td>
            <td>67.0</td>
            <td>81.7</td>
            <td>78.6</td>
            <td>81.2</td>
        </tr>
        <tr>
            <td>Doubao-2.0-pro</td>
            <td>74.2</td>
            <td>78.9</td>
            <td>72.8</td>
            <td>75.7</td>
            <td>82.8</td>
            <td>74.4</td>
            <td>69.0</td>
            <td>70.0</td>
            <td>73.3</td>
            <td>82.0</td>
            <td>69.9</td>
            <td>83.4</td>
            <td>76.5</td>
            <td>72.5</td>
            <td>81.3</td>
            <td>75.7</td>
            <td>65.8</td>
            <td>74.7</td>
            <td>63.3</td>
            <td>71.9</td>
            <td>71.9</td>
            <td>75.2</td>
            <td>79.5</td>
        </tr>
        <tr>
            <td>Claude-Sonnet-4.6</td>
            <td>73.1</td>
            <td>85.0</td>
            <td>69.3</td>
            <td>79.2</td>
            <td>79.8</td>
            <td>80.6</td>
            <td>72.8</td>
            <td>66.5</td>
            <td>82.3</td>
            <td>83.3</td>
            <td>76.7</td>
            <td>88.0</td>
            <td>83.1</td>
            <td>66.2</td>
            <td>67.8</td>
            <td>71.7</td>
            <td>63.4</td>
            <td>64.3</td>
            <td>70.8</td>
            <td>65.2</td>
            <td>61.3</td>
            <td>65.1</td>
            <td>77.6</td>
        </tr>
        <tr>
            <td>ChatGPT-5.2-2025-12-11</td>
            <td>68.6</td>
            <td>85.6</td>
            <td>63.0</td>
            <td>75.2</td>
            <td>70.8</td>
            <td>79.4</td>
            <td>71.4</td>
            <td>60.0</td>
            <td>77.7</td>
            <td>78.5</td>
            <td>71.6</td>
            <td>85.0</td>
            <td>82.1</td>
            <td>61.1</td>
            <td>64.9</td>
            <td>63.4</td>
            <td>55.8</td>
            <td>65.4</td>
            <td>60.7</td>
            <td>63.8</td>
            <td>56.3</td>
            <td>58.7</td>
            <td>74.0</td>
        </tr>
        <tr>
            <td>Qwen3-VL-Instruct-8b</td>
            <td>68.3</td>
            <td>78.4</td>
            <td>65.0</td>
            <td>73.6</td>
            <td>73.7</td>
            <td>71.4</td>
            <td>69.3</td>
            <td>66.2</td>
            <td>68.5</td>
            <td>79.1</td>
            <td>78.3</td>
            <td>82.2</td>
            <td>73.4</td>
            <td>62.5</td>
            <td>63.1</td>
            <td>58.4</td>
            <td>59.9</td>
            <td>61.9</td>
            <td>57.9</td>
            <td>62.0</td>
            <td>62.6</td>
            <td>73.8</td>
            <td>70.8</td>
        </tr>
        <tr>
            <td>Qwen3.5-Instruct-9B</td>
            <td>65.7</td>
            <td>74.8</td>
            <td>62.7</td>
            <td>72.5</td>
            <td>72.8</td>
            <td>72.0</td>
            <td>72.0</td>
            <td>64.4</td>
            <td>66.2</td>
            <td>77.6</td>
            <td>74.5</td>
            <td>79.1</td>
            <td>74.0</td>
            <td>58.2</td>
            <td>53.4</td>
            <td>56.2</td>
            <td>55.7</td>
            <td>60.3</td>
            <td>54.7</td>
            <td>56.7</td>
            <td>60.8</td>
            <td>67.5</td>
            <td>68.9</td>
        </tr>
        <tr>
            <td>InternVL-3.5-8B</td>
            <td>42.7</td>
            <td>59.7</td>
            <td>37.0</td>
            <td>53.4</td>
            <td>39.8</td>
            <td>64.2</td>
            <td>47.5</td>
            <td>42.7</td>
            <td>53.8</td>
            <td>60.6</td>
            <td>52.2</td>
            <td>63.2</td>
            <td>57.0</td>
            <td>30.6</td>
            <td>8.2</td>
            <td>9.0</td>
            <td>45.6</td>
            <td>30.3</td>
            <td>26.1</td>
            <td>10.8</td>
            <td>55.3</td>
            <td>59.3</td>
            <td>45.3</td>
        </tr>
        <tr>
            <td rowspan="13"><strong>Specialized</strong><br><strong>VLMs</strong></td>
            <td>dots.mocr</td>
            <td><ins>80.5</ins></td>
            <td><strong>90.5</strong></td>
            <td><ins>77.2</ins></td>
            <td><ins>81.7</ins></td>
            <td>82.6</td>
            <td><ins>87.4</ins></td>
            <td>71.3</td>
            <td>70.1</td>
            <td><ins>84.5</ins></td>
            <td><ins>89.3</ins></td>
            <td><ins>83.2</ins></td>
            <td>86.8</td>
            <td>79.9</td>
            <td><ins>79.2</ins></td>
            <td><ins>83.3</ins></td>
            <td><ins>83.6</ins></td>
            <td><strong>75.0</strong></td>
            <td>78.7</td>
            <td>71.2</td>
            <td><ins>77.9</ins></td>
            <td>84.6</td>
            <td><ins>79.6</ins></td>
            <td><ins>82.8</ins></td>
        </tr>
        <tr>
            <td>PaddleOCR-VL-1.5</td>
            <td>78.3</td>
            <td>87.4</td>
            <td>75.2</td>
            <td>81.2</td>
            <td>84.8</td>
            <td>83.0</td>
            <td>75.7</td>
            <td><ins>78.1</ins></td>
            <td>83.9</td>
            <td>85.2</td>
            <td>80.6</td>
            <td>80.2</td>
            <td>78.9</td>
            <td>74.9</td>
            <td>71.3</td>
            <td>67.7</td>
            <td>69.5</td>
            <td><strong>86.0</strong></td>
            <td><ins>76.0</ins></td>
            <td>68.4</td>
            <td><ins>84.8</ins></td>
            <td>75.7</td>
            <td>80.7</td>
        </tr>
        <tr>
            <td>dots.ocr</td>
            <td>76.5</td>
            <td>88.8</td>
            <td>72.3</td>
            <td>79.1</td>
            <td>79.7</td>
            <td>81.2</td>
            <td>69.2</td>
            <td>67.1</td>
            <td>82.5</td>
            <td>87.8</td>
            <td>78.8</td>
            <td>86.9</td>
            <td>79.1</td>
            <td>73.5</td>
            <td>75.9</td>
            <td>77.3</td>
            <td>70.6</td>
            <td>68.5</td>
            <td>66.8</td>
            <td>73.3</td>
            <td>79.1</td>
            <td>76.2</td>
            <td>79.7</td>
        </tr>
        <tr>
            <td>olmOCR2</td>
            <td>70.4</td>
            <td>79.9</td>
            <td>67.2</td>
            <td>76.7</td>
            <td>75.7</td>
            <td>77.3</td>
            <td>72.5</td>
            <td>68.9</td>
            <td>70.6</td>
            <td>81.0</td>
            <td>72.0</td>
            <td><ins>88.0</ins></td>
            <td>84.0</td>
            <td>63.3</td>
            <td>59.0</td>
            <td>60.8</td>
            <td>59.4</td>
            <td>70.6</td>
            <td>65.8</td>
            <td>59.2</td>
            <td>68.6</td>
            <td>63.4</td>
            <td>76.1</td>
        </tr>
        <tr>
            <td>PaddleOCR-VL</td>
            <td>69.6</td>
            <td>87.6</td>
            <td>63.6</td>
            <td>72.1</td>
            <td>78.2</td>
            <td>79.3</td>
            <td>62.9</td>
            <td>66.0</td>
            <td>77.4</td>
            <td>78.4</td>
            <td>67.9</td>
            <td>72.0</td>
            <td>66.6</td>
            <td>66.7</td>
            <td>65.8</td>
            <td>68.4</td>
            <td>59.9</td>
            <td>77.8</td>
            <td>56.9</td>
            <td>57.8</td>
            <td>78.2</td>
            <td>68.5</td>
            <td>70.9</td>
        </tr>
        <tr>
            <td>HunyuanOCR</td>
            <td>68.3</td>
            <td>80.2</td>
            <td>64.3</td>
            <td>72.4</td>
            <td>75.0</td>
            <td>73.1</td>
            <td>63.0</td>
            <td>66.1</td>
            <td>69.9</td>
            <td>80.3</td>
            <td>61.4</td>
            <td>81.9</td>
            <td>80.6</td>
            <td>63.7</td>
            <td>68.3</td>
            <td>73.1</td>
            <td>55.6</td>
            <td>68.9</td>
            <td>52.2</td>
            <td>60.7</td>
            <td>66.8</td>
            <td>64.2</td>
            <td>68.6</td>
        </tr>
        <tr>
            <td>GLM-OCR</td>
            <td>67.3</td>
            <td>77.9</td>
            <td>63.7</td>
            <td>78.7</td>
            <td>82.7</td>
            <td>84.5</td>
            <td><ins>75.8</ins></td>
            <td>76.2</td>
            <td>79.7</td>
            <td>82.8</td>
            <td>80.2</td>
            <td>77.4</td>
            <td>69.2</td>
            <td>54.3</td>
            <td>21.7</td>
            <td>39.6</td>
            <td>65.5</td>
            <td>61.2</td>
            <td>64.2</td>
            <td>27.4</td>
            <td>78.5</td>
            <td>76.7</td>
            <td>68.8</td>
        </tr>
        <tr>
            <td>MonkeyOCRv1.5</td>
            <td>65.0</td>
            <td>84.3</td>
            <td>58.6</td>
            <td>67.4</td>
            <td>70.8</td>
            <td>74.9</td>
            <td>55.6</td>
            <td>60.3</td>
            <td>73.8</td>
            <td>75.9</td>
            <td>66.3</td>
            <td>67.2</td>
            <td>61.4</td>
            <td>62.4</td>
            <td>60.1</td>
            <td>56.8</td>
            <td>57.0</td>
            <td>78.9</td>
            <td>51.7</td>
            <td>55.6</td>
            <td>74.8</td>
            <td>64.1</td>
            <td>69.0</td>
        </tr>
        <tr>
            <td>Nanonets-ocr2-3B</td>
            <td>64.2</td>
            <td>79.2</td>
            <td>59.3</td>
            <td>71.4</td>
            <td>76.7</td>
            <td>76.4</td>
            <td>61.8</td>
            <td>66.1</td>
            <td>68.4</td>
            <td>78.5</td>
            <td>74.1</td>
            <td>74.2</td>
            <td>66.0</td>
            <td>56.2</td>
            <td>60.2</td>
            <td>59.2</td>
            <td>52.1</td>
            <td>54.7</td>
            <td>45.5</td>
            <td>44.6</td>
            <td>68.3</td>
            <td>65.1</td>
            <td>67.6</td>
        </tr>
        <tr>
            <td>Nanonets-OCR-s</td>
            <td>63.7</td>
            <td>78.8</td>
            <td>58.7</td>
            <td>71.3</td>
            <td>75.1</td>
            <td>78.5</td>
            <td>61.2</td>
            <td>62.5</td>
            <td>70.3</td>
            <td>81.0</td>
            <td>69.6</td>
            <td>75.9</td>
            <td>67.5</td>
            <td>55.0</td>
            <td>59.5</td>
            <td>61.8</td>
            <td>55.9</td>
            <td>51.2</td>
            <td>43.5</td>
            <td>39.5</td>
            <td>67.4</td>
            <td>61.5</td>
            <td>66.6</td>
        </tr>
        <tr>
            <td>MonkeyOCR-pro-3B</td>
            <td>52.2</td>
            <td>68.0</td>
            <td>47.0</td>
            <td>65.1</td>
            <td>71.7</td>
            <td>77.9</td>
            <td>55.9</td>
            <td>62.1</td>
            <td>66.2</td>
            <td>74.5</td>
            <td>66.3</td>
            <td>71.1</td>
            <td>40.2</td>
            <td>37.6</td>
            <td>4.6</td>
            <td>4.2</td>
            <td>55.2</td>
            <td>60.5</td>
            <td>42.6</td>
            <td>9.1</td>
            <td>72.2</td>
            <td>52.4</td>
            <td>53.6</td>
        </tr>
        <tr>
            <td>DeepSeek-OCR</td>
            <td>51.8</td>
            <td>80.7</td>
            <td>42.2</td>
            <td>54.5</td>
            <td>55.0</td>
            <td>58.3</td>
            <td>44.1</td>
            <td>43.2</td>
            <td>60.9</td>
            <td>69.3</td>
            <td>52.4</td>
            <td>53.0</td>
            <td>54.1</td>
            <td>48.9</td>
            <td>56.9</td>
            <td>52.2</td>
            <td>49.1</td>
            <td>28.2</td>
            <td>36.2</td>
            <td>49.4</td>
            <td>59.7</td>
            <td>59.2</td>
            <td>54.5</td>
        </tr>
        <tr>
            <td>MinerU-2.5-VLM</td>
            <td>46.3</td>
            <td>61.9</td>
            <td>40.8</td>
            <td>63.0</td>
            <td>68.8</td>
            <td>78.4</td>
            <td>54.7</td>
            <td>57.3</td>
            <td>67.5</td>
            <td>75.2</td>
            <td>60.4</td>
            <td>58.8</td>
            <td>46.0</td>
            <td>27.4</td>
            <td>1.3</td>
            <td>9.0</td>
            <td>39.1</td>
            <td>14.7</td>
            <td>8.6</td>
            <td>11.3</td>
            <td>72.9</td>
            <td>62.2</td>
            <td>48.7</td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Pipeline</strong><br><strong>Tools</strong></td>
            <td>PP-StructureV3</td>
            <td>45.4</td>
            <td>56.2</td>
            <td>41.7</td>
            <td>59.8</td>
            <td>60.4</td>
            <td>68.7</td>
            <td>54.4</td>
            <td>49.8</td>
            <td>69.6</td>
            <td>68.9</td>
            <td>55.5</td>
            <td>58.4</td>
            <td>52.7</td>
            <td>28.9</td>
            <td>1.0</td>
            <td>7.7</td>
            <td>56.2</td>
            <td>15.4</td>
            <td>7.5</td>
            <td>11.9</td>
            <td>72.2</td>
            <td>59.1</td>
            <td>49.6</td>
        </tr>
        <tr>
            <td>MinerU-2.5-pipeline</td>
            <td>33.5</td>
            <td>57.6</td>
            <td>25.4</td>
            <td>46.5</td>
            <td>54.3</td>
            <td>58.3</td>
            <td>38.4</td>
            <td>43.6</td>
            <td>51.9</td>
            <td>56.5</td>
            <td>43.9</td>
            <td>44.0</td>
            <td>27.6</td>
            <td>18.7</td>
            <td>1.2</td>
            <td>5.3</td>
            <td>24.5</td>
            <td>6.8</td>
            <td>4.2</td>
            <td>6.4</td>
            <td>53.9</td>
            <td>47.2</td>
            <td>36.2</td>
        </tr>
    </tbody>
</table>

## Evaluation

### Environment Setup

```bash
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git
cd MultimodalOCR/MDPBench

conda create -n mdpbench python=3.10
conda activate mdpbench

pip install -r requirements.txt
```
For CDM, you need to set up the CDM environment according to the [README](./metrics/cdm/).

### End-to-End Evaluation on Public Set

Please follow the steps below to conduct the evaluation.

#### Step 1: Download the dataset

Download MDPBench (public) from Hugging Face or ModelScope.
Please install the required packages before downloading:

```bash
pip install -U "huggingface_hub[cli]" modelscope
```

```bash
# Download from Hugging Face (default)
python tools/download_dataset.py

# Download from ModelScope
python tools/download_dataset.py --source modelscope
```

#### Step 2: Run Model Inference

If you use the official code of a document parsing model for inference, please ensure that the inference results are saved in Markdown format. Each output file should have the same filename as the corresponding image, with the extension changed to .md. Below, we provide an example of running inference with Gemini-3-pro-preview:

```bash

export API_KEY="YOUR_API_KEY"
export BASE_URL="YOUR_BASE_URL"
python scripts/batch_process_gemini-3-pro-preview.py --input_dir MDPBench_dataset/MDPBench_img_public --output_dir result/Gemini3-pro-preview

```

#### Step 3: Edit the Configuration File

You should set `prediction.data_path` in [configs/end2end.yaml](./configs/end2end.yaml) to the directory where the model’s Markdown outputs are stored.

```yaml

# ----- Here are the lines to be modified -----

  dataset:

    dataset_name: end2end_dataset

    ground_truth:

      data_path: ./MDPBench_dataset/MDPBench_public.json

    prediction:

      data_path: ./result/Gemini3-pro-preview

```



#### Step 4: Compute the metrics for each file.

Run the following command to compute the score for each prediction.

```bash

python pdf_validation.py --config ./configs/end2end.yaml

```



#### Step 5: Calculate Final Scores

Upon completion of the evaluation, MDPBench will create a new folder in the result directory with the `_result` suffix to store the evaluation results.
Run the following command to obtain the overall scores of the model across different languages.

```bash

python tools/calculate_scores.py  --result_folder result/Gemini3-pro-preview_result

```

### End-to-End Evaluation on Private Set
To prevent data leakage and avoid sample-specific fine-tuning, we choose not to release the Private Set. If you would like to evaluate your model on MDPBench Private, please open an issue or contact us at [zhangli123@hust.edu.cn](mailto:zhangli123@hust.edu.cn), and please also provide your model’s inference code and the corresponding weight links.




## Acknowledgements

We would like to express our sincere appreciation to [OmniDocBench](https://github.com/opendatalab/OmniDocBench.git) for providing the evaluation pipeline! We also welcome any suggestions that can help us improve this benchmark.


## Citing MDPBench
If you find this benchmark useful, please cite:
```bibtex
@misc{li2026mdpbenchbenchmarkmultilingualdocument,
      title={MDPBench: A Benchmark for Multilingual Document Parsing in Real-World Scenarios}, 
      author={Zhang Li and Zhibo Lin and Qiang Liu and Ziyang Zhang and Shuo Zhang and Zidun Guo and Jiajun Song and Jiarui Zhang and Xiang Bai and Yuliang Liu},
      year={2026},
      eprint={2603.28130},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.28130}, 
}
```
