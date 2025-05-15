<div align="center">
<h1> WebVoyager 
<img src="./assets/icon.png" width="45px">
<br> Building an End-to-End Web Agent with Large Multimodal Models </h1>
</div>

<div align="center">

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10.13-green.svg)
![Selenium](https://img.shields.io/badge/Selenium-4.15.2-red)

</div>

<div align="center">
<img src="./assets/overall_process_crop.png" width="90%">
</div>



<div align="center">
<h1> WebVoyager <img src="./assets/icon.png" width="45px"> <br> Web Agents with Memory Recall via RAG</h1>
</div>

<div align="center">

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10.13-green.svg)
![Selenium](https://img.shields.io/badge/Selenium-4.15.2-red)

</div>

<div align="center">
<img src="./assets/overall_process_crop.png" width="90%">
</div>

## Introduction

WebVoyager is an end-to-end web agent enhanced with Retrieval-Augmented Generation (RAG) and Large Multimodal Models (LMMs), designed to complete complex web-based tasks by browsing real-world websites. This repository includes:

* A browser agent capable of vision-language reasoning and web interaction.
* A robust PDF/manual retrieval and indexing pipeline (RAG) that enhances memory recall and instruction following.
* A WebDriver-based environment using Selenium to support real browser actions.
* Evaluation tools to assess both human and automated performance.

## Setup Environment

Make sure you have Google Chrome installed. We recommend using the latest version of Selenium, which does not require manual `chromedriver` installation.

```bash
conda create -n webvoyager python=3.10
conda activate webvoyager
pip install -r requirements.txt
pip install langchain sentence-transformers torch
pip install pdfplumber pymupdf pypdf pymupdf4llm
pip install langchain-community chromadb tiktoken
```

## Data Format

### Task File

Tasks are defined in JSONL format. Each task includes:

```json
{
  "id": 1,
  "web": "https://arxiv.org",
  "ques": "Search for papers on 'neural networks for image processing' in the Computer Science category on ArXiv and report how many were submitted in the last week."
}
```

### PDF Files

* PDFs can be stored in the `data/` directory.
* Enhanced Markdown with image descriptions and metadata are generated during processing.

## Running the Agent

### Step-by-step Execution

```bash
bash run.sh
```

This script runs `run.py`, which performs the following:

* Loads tasks from `data/arxiv_tasks1.jsonl`
* Indexes the specified PDF using the `PDFEnhancementPipeline`
* Retrieves relevant instructions using RAG
* Uses GPT-4o (or another model) to plan and act based on screenshots and retrieved manuals
* Interacts with the browser to complete tasks

You can change the default parameters in `run.sh`, e.g., to use text-only mode or change the OpenAI model.

### Example Run Script

```bash
python -u run.py \
    --test_file ./data/arxiv_tasks1.jsonl \
    --api_key "YOUR_OPENAI_KEY" \
    --api_model gpt-4o \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 0 \
    --fix_box_color \
    --seed 42 \
    --window_width 1920 \
    --window_height 1080 \
    --pdf_path data/arXiv.pdf
```

## PDF Retrieval Pipeline (RAG)

The agent uses a PDF processing pipeline that:

1. Converts PDF to Markdown and extracts images.
2. Uses GPT-4o to describe images.
3. Embeds text chunks with OpenAI or HuggingFace embeddings (e.g., `bge-m3`).
4. Indexes with ChromaDB.
5. Retrieves relevant steps based on user query.
6. Optionally enhances prompts with retrieved content.

## Prompt Design

You can modify prompts in `prompts.py` or `strict_guidance_prompts.py`. The agent distinguishes between system-level and task-specific prompts. For RAG-based planning, the agent will follow structured manuals step-by-step. Prompts can include step-tracking, contextual rules, and answer formatting.

## Evaluation

Evaluation can be done using logs or screenshots in `results/`. Auto-evaluation is supported with GPT-4V.

Run evaluation:

```bash
cd evaluation
bash run_eval.sh
```

### Evaluation Script

```bash
nohup python -u auto_eval.py \
    --api_key YOUR_API_KEY \
    --process_dir ../results/examples \
    --max_attached_imgs 15 > evaluation.log &
```

## Citation

If you use WebVoyager in your research:

```bibtex
@article{he2024webvoyager,
  title={WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models},
  author={He, Hongliang and Yao, Wenlin and Ma, Kaixin and Yu, Wenhao and Dai, Yong and Zhang, Hongming and Lan, Zhenzhong and Yu, Dong},
  journal={arXiv preprint arXiv:2401.13919},
  year={2024}
}
```

## Disclaimer

WebVoyager is a research project and not an official product. Its results may vary based on prompt engineering, OpenAI API behavior, and live website changes. Use at your own discretion.


## Student Information
```
Slametian Dewa Tegar Perkasa 石柏楷 - 113527602
International Graduate Program in AI
National Central University (NCU)
Taiwan

GitHub: https://github.com/dewa-ai/assignment3-agenticai
```