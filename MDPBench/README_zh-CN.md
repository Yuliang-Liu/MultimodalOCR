<h1 align="center">
MDPBench
</h1>

<div align="center">
<a href="./README.md">English</a> | 简体中文
<br>

[![arXiv](https://img.shields.io/badge/Arxiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.28130) 
[![HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-ffd21e.svg?logo=huggingface)](https://huggingface.co/datasets/Delores-Lin/MDPBench)
[![ModelScope](https://img.shields.io/badge/Dataset-ModelScope-blue.svg)](https://modelscope.cn/datasets/DeloresLin/MDPBench)

</div>

我们推出了多语言文档解析基准测试（Multilingual Document Parsing Benchmark，简称 MDPBench），这是首个专门针对多语言原生数字和拍摄文档解析的基准测试。尽管文档解析技术已经取得了长足的进步，但这些进展大部分局限于少数主流语言中排版整洁、格式良好的原生数字页面。目前还缺乏一个系统的基准来评估模型在涵盖多种书写系统和低资源语言的原生数字及拍摄文档上的表现。

MDPBench 包含 3,400 张文档图像，涵盖 17 种语言（简体中文、繁体中文、英文、阿拉伯文、德文、西班牙文、法文、印地文、印尼文、意大利文、日文、韩文、葡萄牙文、俄文、泰文和越南文）、多种书写系统以及各种复杂的拍摄条件。其高质量的标注数据是通过“专家模型标注、人工校对和人工验证”这一工作流程生成的。为了确保评估的公平性并防止数据泄露，我们将数据集划分为相互独立的公开和私有评估子集。

我们对开源和闭源模型进行的全面评估揭后发现：闭源模型（尤其是 Gemini-3-Pro）表现出相对较强的鲁棒性，但开源替代方案的性能却出现了断崖式下跌。这种现象在非拉丁字母系统和现实世界的拍摄文档上尤为明显——模型在拍摄文档上的性能平均下降了 17.8%，在非拉丁字母系统上的性能平均下降了 14.0%。这些结果暴露出当前模型在不同语言和条件下存在显著的性能失衡，同时也为构建更具包容性、可直接部署落地的文档解析系统指明了具体的方向。

## 核心实验结果

下表展示了各类通用视觉大模型 (General VLMs)、垂直专用视觉大模型 (Specialized VLMs) 以及管线工具 (Pipeline Tools) 在 MDPBench 上的总体性能。评估拆分展示了它们在数字原版(Digit.) / 拍摄(Photo.) ，以及各个拉丁/非拉丁语系分支上的解析指标均分：

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

## 评估流程

### 环境配置

请按照以下步骤准备评测环境：

```bash
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git
cd MultimodalOCR/MDPBench

# 强烈建议首先创建一个干净的虚拟环境
conda create -n mdpbench python=3.10
conda activate mdpbench

# 安装本评测所需的依赖包
pip install -r requirements.txt
```

对于公式，MDPBench使用CDM进行评测，请前往[README](./metrics/cdm/)中按照相应的教程配置CDM环境。


### 端到端评测


请按照下面的步骤使用MDPBench进行评测。


#### Step 1: 下载数据集



你可以使用 [tools/download_dataset.py](./tools/download_dataset.py) 脚本从Hugging Face Hub或ModelScope下载数据集。在下载之前，请先安装必要的下载库：

```bash
# 从 Hugging Face 下载（默认）
pip install huggingface_hub
python tools/download_dataset.py
```
```bash
# 从 ModelScope 下载
pip install modelscope
python tools/download_dataset.py --source modelscope
```



#### Step 2: 运行模型推理

首先，你可以使用 MDPBench [scripts](./scripts/)下提供的脚本对相应的模型进行推理
```bash

export API_KEY="YOUR_API_KEY"
export BASE_URL="YOUR_BASE_URL"
python scripts/batch_process_gemini-3-pro-preview.py --input_dir MDPBench_dataset/MDPBench_img_public --output_dir result/gemini-3-pro-preview

```

此外，在评测MonkeyOCR等其他模型时可以用官方的脚本对数据集的图片进行推理。模型推理结果应为 markdown 格式，并且存储在与图像文件名相同但扩展名为 .md 的文件目录中。

#### Step 3: 配置评测

在运行评测脚本之前，您需要在 [configs/end2end.yaml](./configs/end2end.yaml) 中的 `ground_truth` 的 `data_path` 中提供 `MDPBench_public.json` 的路径，在 `prediction` 的 `data_path` 中提供包含模型推理结果的目录路径，如下所示：



```yaml

# ----- Here are the lines to be modified -----

  dataset:

    dataset_name: end2end_dataset

    ground_truth:

      data_path: ./MDPBench_dataset/MDPBench_public.json

    prediction:

      data_path: ./result/Gemini3-pro-preview

```



#### Step 4: 运行评测脚本

运行下面的命令进行评测

```bash

python pdf_validation.py --config ./configs/end2end.yaml

```

脚本将自动读取配置中所列的模型输出路径与官方 Ground Truth 进行对比，根据内部 `dataset`, `task`, `metrics` 和 `registry` 相关模块输出详细的 JSON 指标结果。



#### Step 5: 计算最终分数 


评测完成后，MDPBench会在result文件夹下新增一个带`_result`后缀的文件夹用来存放评测结果。
你可以使用 [tools/calculate_scores.py](./tools/calculate_scores.py) 计算最终的各项得分，并输出为概览表格：

```bash

python tools/calculate_scores.py  --result_folder result/Gemini3-pro-preview_result

```


### 在私有数据集上进行端到端评测


为了防止数据泄露并避免针对特定样本进行微调，我们决定不公开私有集（Private Set）。如果您希望在 MDPBench 私有集上评估您的模型，请提交 Issue 或通过 [zhangli123@hust.edu.cn](mailto:zhangli123@hust.edu.cn) 与我们联系，同时请附上您模型的推理代码及相应的权重链接。


## 致谢



我们衷心感谢 [OmniDocBench](https://github.com/opendatalab/OmniDocBench.git) 提供的评测pipline！同时，我们也欢迎任何能够帮助我们改进该基准测试的建议。


## 引用
如果该基准测试工具对您有所启迪或帮助，请在您的工作中引入以下文献：
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
