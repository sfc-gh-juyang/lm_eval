# Unified MMLU & MMMU Evaluation Framework

A comprehensive evaluation framework for testing both language models (MMLU) and vision-language models (MMMU) with a single unified interface.

## Features

- **MMLU Evaluation**: Test any Hugging Face language model on MMLU (Massive Multitask Language Understanding)
- **MMMU Evaluation**: Test any Hugging Face vision-language model on MMMU (Massive Multi-discipline Multimodal Understanding)
- **Unified Interface**: Single script handles both text-only and vision-language models
- Preset configurations for 40+ popular models
- Automatic model type detection
- Support for quantization (4bit/8bit)
- Quick and fast evaluation modes
- Comprehensive result analysis and reporting

## Quick Start

### List Available Models
```bash
# List all models (both MMLU and MMMU)
python quick_eval.py --list

# List only text-only models
python quick_eval.py --list --benchmark MMLU

# List only vision-language models  
python quick_eval.py --list --benchmark MMMU
```

### MMLU (Text-only Language Models)
```bash
# Evaluate a preset text model
python quick_eval.py mistral-7b --quick

# Evaluate custom text model (auto-detects MMLU)
python quick_eval.py --custom-model "microsoft/DialoGPT-medium" --fast

# Force MMLU benchmark
python quick_eval.py --custom-model "microsoft/phi-2" --benchmark MMLU --quick
```

### MMMU (Vision-Language Models)
```bash
# Evaluate a preset vision-language model
python quick_eval.py llava-1.5-7b --quick

# Evaluate custom vision-language model (auto-detects MMMU)
python quick_eval.py --custom-model "llava-hf/llava-1.5-13b-hf" --fast

# Force MMMU benchmark
python quick_eval.py --custom-model "Qwen/Qwen-VL-Chat" --benchmark MMMU --quick
```

## Installation

```bash
pip install -r requirements.txt
```

For more detailed instructions, see EVALUATION_GUIDE.md
