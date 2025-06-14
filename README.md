# Unified MMLU & MMMU Evaluation Framework

A comprehensive evaluation framework for testing language models on **MMLU** (Massive Multitask Language Understanding) and vision-language models on **MMMU** (Massive Multi-discipline Multimodal Understanding) benchmarks.

## Features

- **Unified Interface**: Single script handles both text-only and vision-language models
- **40+ Preset Models**: Pre-configured popular models including Llama, Mistral, Qwen, LLaVA, and more
- **Automatic Detection**: Automatically detects model type (text vs vision-language)
- **Flexible Evaluation**: Support for quick evaluation, custom subjects, and sample limits
- **Quantization Support**: 4-bit and 8-bit quantization for resource-constrained environments
- **Comprehensive Reporting**: Detailed results and analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd quan

# Install dependencies
pip install -r requirements.txt

# Optional: Run setup test
python test_setup.py
```

## Quick Start

### List Available Models

```bash
# List all preset models
python quick_eval.py --list

# List only MMLU models (text-only)
python quick_eval.py --list --benchmark MMLU

# List only MMMU models (vision-language)
python quick_eval.py --list --benchmark MMMU
```

### Evaluate Preset Models

```bash
# MMLU evaluation (text models)
python quick_eval.py mistral-7b --quick
python quick_eval.py llama3-8b-instruct --fast

# MMMU evaluation (vision-language models)  
python quick_eval.py llava-1.5-7b --quick
python quick_eval.py qwen-vl-chat --fast
```

### Evaluate Custom Models

```bash
# Custom model (auto-detects benchmark)
python quick_eval.py --custom-model "microsoft/DialoGPT-medium" --quick

# Force specific benchmark
python quick_eval.py --custom-model "microsoft/phi-2" --benchmark MMLU --fast
python quick_eval.py --custom-model "Qwen/Qwen-VL-Chat" --benchmark MMMU --quick

# With quantization
python quick_eval.py --custom-model "meta-llama/Llama-2-13b-hf" --quantization 4bit --quick
```

## Evaluation Modes

- **Full**: All subjects, all samples (most comprehensive)
- **Quick** (`--quick`): All subjects, limited samples (faster)
- **Fast** (`--fast`): Limited subjects and samples (fastest)

## Popular Preset Models

### MMLU Models (Text-Only)

| Family | Models |
|--------|---------|
| **Llama** | `llama2-7b`, `llama2-13b`, `llama3-8b-instruct`, `llama3-70b-instruct` |
| **Mistral** | `mistral-7b`, `mistral-7b-instruct`, `mixtral-8x7b-instruct` |
| **Qwen** | `qwen2-7b-instruct`, `qwen2-72b-instruct`, `qwen2.5-32b-instruct` |

### MMMU Models (Vision-Language)

| Family | Models |
|--------|---------|
| **LLaVA** | `llava-1.5-7b`, `llava-1.5-13b`, `llava-1.6-vicuna-7b` |
| **Qwen-VL** | `qwen-vl-chat`, `qwen-vl-plus` |
| **InstructBLIP** | `instructblip-vicuna-7b`, `instructblip-flan-t5-xl` |

## Advanced Usage

For detailed documentation including direct evaluation scripts, custom configurations, and advanced options, see [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md).

### Direct Evaluation Scripts

```bash
# Direct MMLU evaluation
python evaluate_mmlu.py --model "mistralai/Mistral-7B-v0.1" --output results.json

# Direct MMMU evaluation  
python evaluate_mmmu.py --model "llava-hf/llava-1.5-7b-hf" --output mmmu_results.json
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory (varies by model)

## Contributing

Contributions are welcome! Please see the [evaluation guide](EVALUATION_GUIDE.md) for detailed documentation.

## License

This project is licensed under the MIT License.
