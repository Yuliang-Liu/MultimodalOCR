/*
 * Display names and first-party destinations for the initial MDPBench results.
 * `name` is the evaluation identifier preserved in leaderboard.json; `label`
 * is what readers see. Entries using a project page (rather than a versioned
 * checkpoint) retain their exact evaluation identifier as the label.
 */
const MODEL_METADATA = {
  "gemini-3-pro-preview": { label: "Gemini 3 Pro Preview" },
  "kimi-k2.5": {
    label: "Kimi K2.5",
    links: [
      { label: "GitHub", url: "https://github.com/MoonshotAI/Kimi-K2.5" },
      { label: "Hugging Face", url: "https://huggingface.co/moonshotai/Kimi-K2.5" }
    ]
  },
  "doubao-seed-2-0-pro-260215": { label: "Doubao-Seed-2.0-Pro (260215)" },
  "claude-sonnet-4-6": { label: "Claude Sonnet 4.6" },
  "gpt-5.2-2025-12-11": { label: "GPT-5.2 (2025-12-11)" },
  "Qwen3-VL-8B-Instruct": {
    label: "Qwen3-VL-8B-Instruct",
    links: [
      { label: "GitHub", url: "https://github.com/QwenLM/Qwen3-VL" },
      { label: "Hugging Face", url: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct" }
    ]
  },
  "Qwen3.5-Instruct-9B": {
    label: "Qwen3.5-9B",
    links: [
      { label: "Hugging Face", url: "https://huggingface.co/Qwen/Qwen3.5-9B" }
    ]
  },
  "InternVL3_5-8B": {
    label: "InternVL3_5-8B",
    links: [
      { label: "GitHub", url: "https://github.com/OpenGVLab/InternVL" },
      { label: "Hugging Face", url: "https://huggingface.co/OpenGVLab/InternVL3_5-8B" }
    ]
  },
  "MonkeyOCRv2-B-Parsing": {
    label: "MonkeyOCRv2-B-Parsing",
    links: [
      { label: "GitHub", url: "https://github.com/Yuliang-Liu/MonkeyOCRv2" },
      { label: "Hugging Face", url: "https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing" }
    ]
  },
  "MonkeyOCRv2-S-Parsing": {
    label: "MonkeyOCRv2-S-Parsing",
    links: [
      { label: "GitHub", url: "https://github.com/Yuliang-Liu/MonkeyOCRv2" },
      { label: "Hugging Face", url: "https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing" }
    ]
  },
  "dots.mocr": {
    label: "dots.mocr",
    links: [
      { label: "GitHub", url: "https://github.com/rednote-hilab/dots.mocr" },
      { label: "Hugging Face", url: "https://huggingface.co/rednote-hilab/dots.mocr" }
    ]
  },
  "chandra-ocr-2": {
    label: "Chandra OCR 2",
    links: [
      { label: "GitHub", url: "https://github.com/datalab-to/chandra" },
      { label: "Hugging Face", url: "https://huggingface.co/datalab-to/chandra-ocr-2" }
    ]
  },
  "PaddleOCR-VL-1.5": {
    label: "PaddleOCR-VL-1.5",
    links: [
      { label: "GitHub", url: "https://github.com/PaddlePaddle/PaddleOCR" },
      { label: "Hugging Face", url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5" }
    ]
  },
  "Mistral-OCR-4.0": { label: "Mistral OCR 4.0" },
  "HunyuanOCR-1.5": {
    label: "HunyuanOCR-1.5",
    links: [
      { label: "GitHub", url: "https://github.com/Tencent-Hunyuan/HunyuanOCR" },
      { label: "Hugging Face", url: "https://huggingface.co/tencent/HunyuanOCR" }
    ]
  },
  "dots.ocr": {
    label: "dots.ocr",
    links: [
      { label: "GitHub", url: "https://github.com/rednote-hilab/dots.ocr" },
      { label: "Hugging Face", url: "https://huggingface.co/rednote-hilab/dots.ocr" }
    ]
  },
  "PaddleOCR-VL-1.6": {
    label: "PaddleOCR-VL-1.6",
    links: [
      { label: "GitHub", url: "https://github.com/PaddlePaddle/PaddleOCR" },
      { label: "Hugging Face", url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6" }
    ]
  },
  "MinerU-2.5-Pro-1.2B": {
    label: "MinerU2.5-Pro-1.2B",
    links: [
      { label: "GitHub", url: "https://github.com/opendatalab/MinerU" },
      { label: "Hugging Face", url: "https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B" }
    ]
  },
  "olmOCR2": {
    label: "olmOCR 2",
    links: [
      { label: "GitHub", url: "https://github.com/allenai/olmocr" },
      { label: "Hugging Face", url: "https://huggingface.co/allenai/olmOCR-2-7B-1025" }
    ]
  },
  "PaddleOCR-VL": {
    label: "PaddleOCR-VL-0.9B",
    links: [
      { label: "GitHub", url: "https://github.com/PaddlePaddle/PaddleOCR" },
      { label: "Hugging Face", url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL" }
    ]
  },
  "HunyuanOCR": { label: "HunyuanOCR" },
  "GLM-OCR": {
    label: "GLM-OCR",
    links: [
      { label: "GitHub", url: "https://github.com/zai-org/GLM-OCR" },
      { label: "Hugging Face", url: "https://huggingface.co/zai-org/GLM-OCR" }
    ]
  },
  "MonkeyOCRv1.5": {
    label: "MonkeyOCR v1.5",
    links: [{ label: "GitHub", url: "https://github.com/Yuliang-Liu/MonkeyOCR" }]
  },
  "Nanonets-OCR2-3B": {
    label: "Nanonets-OCR2-3B",
    links: [
      { label: "GitHub", url: "https://github.com/NanoNets/Nanonets-OCR2" },
      { label: "Hugging Face", url: "https://huggingface.co/nanonets/Nanonets-OCR2-3B" }
    ]
  },
  "LightOnOCR-2-1B": {
    label: "LightOnOCR-2-1B",
    links: [{ label: "Hugging Face", url: "https://huggingface.co/lightonai/LightOnOCR-2-1B" }]
  },
  "Nanonets-OCR-s": {
    label: "Nanonets-OCR-s",
    links: [
      { label: "GitHub", url: "https://github.com/NanoNets/Nanonets-OCR2" },
      { label: "Hugging Face", url: "https://huggingface.co/nanonets/Nanonets-OCR-s" }
    ]
  },
  "FalconOCR": {
    label: "Falcon-OCR",
    links: [
      { label: "GitHub", url: "https://github.com/tiiuae/Falcon-Perception" },
      { label: "Hugging Face", url: "https://huggingface.co/tiiuae/Falcon-OCR" }
    ]
  },
  "Unlimited-OCR": { label: "Unlimited-OCR" },
  "MonkeyOCR-pro-3B": {
    label: "MonkeyOCR-pro-3B",
    links: [
      { label: "GitHub", url: "https://github.com/Yuliang-Liu/MonkeyOCR" },
      { label: "Hugging Face", url: "https://huggingface.co/echo840/MonkeyOCR-pro-3B" }
    ]
  },
  "DeepSeek-OCR": {
    label: "DeepSeek-OCR",
    links: [
      { label: "GitHub", url: "https://github.com/deepseek-ai/DeepSeek-OCR" },
      { label: "Hugging Face", url: "https://huggingface.co/deepseek-ai/DeepSeek-OCR" }
    ]
  },
  "MinerU-2.5-VLM": {
    label: "MinerU2.5-VLM",
    links: [{ label: "GitHub", url: "https://github.com/opendatalab/MinerU" }]
  },
  "PP-StructureV3": {
    label: "PP-StructureV3",
    links: [{ label: "GitHub", url: "https://github.com/PaddlePaddle/PaddleOCR" }]
  },
  "MinerU-2.5-pipeline": {
    label: "MinerU2.5-pipeline",
    links: [{ label: "GitHub", url: "https://github.com/opendatalab/MinerU" }]
  }
};
