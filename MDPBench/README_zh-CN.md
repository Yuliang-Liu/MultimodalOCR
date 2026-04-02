<h1 align="center">
MDPBench
</h1>

<div align="center">
<a href="./README.md">English</a> | 简体中文
<br>

[\[📜 arXiv\]](https://arxiv.org/abs/2603.28130) | [[Dataset (🤗Hugging Face)]](https://huggingface.co/datasets/Delores-Lin/MDPBench) | [[Source Code]](https://github.com/Yuliang-Liu/MultimodalOCR)

</div>

**MDPBench (Multilingual Document Parsing Benchmark)** 是首个专门针对多语言电子原生以及拍照文档解析构建的评测基准。尽管现有的文档解析技术已取得显著进展，但大多数测试验证仍局限于格式纯净、排版工整且由主流语言（如数字扫描版）构成的页面上。MDPBench 的出现填补了多样化语种与资源匮乏语言在真实世界拍照场景下的评估空白。

该基准测试台具备以下主要特征：
- **广泛的多语言支持**：包含 3,400 张文档图像，覆盖 17 种语言（简体中文、繁体中文、英语、阿拉伯语、德语、西班牙语、法语、印地语、印尼语、意大利语、日语、韩语、葡萄牙语、俄语、泰语、越南语）。
- **多样化的真实条件**：包含 850 份基于纯数字生成的原生文档页面与 2,550 份通过真实世界复杂条件拍摄的文档图像（每份数字版对应衍生包含多种变形与不同视角的3张照片）。
- **高标注质量**：数据首先由专家模型进行了标注生成，随后通过多轮人工校验纠正和复合审查以保证Ground Truth质量。它专为评估大量开源及闭源视觉大模型的能力而准备。
- **合理的验证划分机制**：内部严格分离了公开（Public）和私有（Private）双轨评估集合。
- **多维度端到端评估**：涵盖从宏观页面结构的阅读顺序（Reading Order）恢复，到具体的版面元素检测（Layout Detection），以及精细的字符识别错误率（Normalized Edit Distance）及符号错误分析指标。

## 目录
- [基准测试介绍](#基准测试介绍)
- [核心实验结果](#核心实验结果)
- [评估流程](#评估流程)
  - [环境配置与运行](#环境配置与运行)
  - [端到端评测 (End-to-End Evaluation)](#端到端评测)
- [致谢](#致谢)
- [引用](#引用)

## 基准测试介绍

MDPBench共计涉及 3,400 份文档图像，涵盖了 17 种不同语言（包含拉丁语系与非拉丁语系）以及 2 大主要文档场景（原生数字版与现实拍摄版）。MDPBench 具备极高的文档多样性与复杂的现实条件，全面涵盖了真实世界拍摄中常见的纸张弯折、形变折叠、光照不均以及不同视角的拍摄干扰。且所有原生数字图像页面均配备了详尽的高质量全量标注，支持对文档内的文本段落阅读顺序（Reading order）、高精度 OCR 文本识别、数学公式的 LaTeX 文本以及表格区（LaTeX和HTML结构）在内的各个要素模块进行深度评测。依托这些详尽的场景划分与标注属性，MDPBench 能够帮助使用者更有效地评估并定位当下多模态大模型在由于真实物理降质以及低资源多语言跨距所频繁导致的各种短板问题——诸如阅读顺序混乱、元素版面丢失、罕见语种错误分类以及严重的字符幻觉（Hallucinations）等乱象。

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

### 环境配置与运行

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

#### 用于公式验证的 CDM 指标配置
如果您希望在端到端评估中通过 CDM (Character Detection Metric 或类似基于 AST 渲染的指标) 高精度审核数学公式，您必须确保本地环境能够运行 NodeJS 的验证依赖（相关执行文件位于 `metrics/cdm` 中）


### 端到端评测



对于文档总体提取效果的评估，测试脚本在不同层面上会调用以下典型指标：计算文本阅读序列完整度的 **归一化编辑距离(Normalized Edit Distance)**，结构还原效果的 **TEDS (用于表格)** 以及 **CDM及编辑距离 (用于行内/独立公式)**。



#### Step 1: 下载数据集 (Download the dataset)



你可以使用 [tools/download_dataset.py](./tools/download_dataset.py) 脚本从Hugging Face Hub下载数据集。

```bash

python tools/download_dataset.py

```



#### Step 2: 运行模型推理 (Run Model Inference)



允许使用图像或 PDF 运行模型推理。模型推理结果应为 markdown 格式，并且存储在与图像文件名相同但扩展名为 .md 的文件目录中。以使用 Gemini-3.1-pro-preview 为例：



```bash

export API_KEY="YOUR_API_KEY"
export BASE_URL="YOUR_BASE_URL"
python scripts/batch_process_gemini-3-pro-preview.py --input_dir demo_data/MDPBench_demo --output_dir demo_data/Gemini3-pro-preview_demo_result

```



这个脚本会读取源文件输出包含预测结果的 markdown 文件，通常保存在诸如 [Gemini3-pro-preview_demo_result](./demo_data/Gemini3-pro-preview_demo_result/) 这样的目录中。



#### Step 3: 配置评测 (Configure Evaluation)



所有的评估输入都是通过配置文件进行配置的。我们在 `configs` 目录下为每项任务提供了模板，我们将在后续部分详细解释配置文件的内容。



简单来说，对于端到端评测 (end2end evaluation)，您需要在 `configs/end2end.yaml` 中的 `ground_truth` 的 `data_path` 中提供 `OmniDocBench.json` 的路径，在 `prediction` 的 `data_path` 中提供包含模型推理结果的目录路径，如下所示：



```yaml

# ----- Here are the lines to be modified -----

  dataset:

    dataset_name: end2end_dataset

    ground_truth:

      data_path: ./demo_data/MDPBench_demo.json

    prediction:

      data_path: ./demo_data/Gemini3-pro-preview_demo_result

```



#### Step 4: 运行评测脚本 (Run Validation Loop)



运行验证脚本，比较预测结果与地面真实数据集来计算指标：



```bash

python pdf_validation.py

```

脚本将自动读取配置中所列的模型输出路径与官方地面真值 Ground Truth 进行对比，根据内部 `dataset`, `task`, `metrics` 和 `registry` 相关模块将详细的 JSON 指标结果输出到你的 `save_dir` 目录下。



#### Step 5: 计算最终分数 (Calculate Final Scores)



你可以使用 [tools/calculate_scores.py](./tools/calculate_scores.py) 将 JSON 指标文件读取为一个分数概览表格：

```bash

python tools/calculate_scores.py Gemini-3-pro-preview_demo_result --result_folder result

```

这会自动打印格式化好的表格内容。


## 致谢



MDPBench 的开发建立在 [OmniDocBench](https://github.com/opendatalab/OmniDocBench.git) 扎实的基础之上。我们衷心感谢他们对文档解析社区所做出的杰出贡献以及开源精神！


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
