# Unified MMLU & MMMU Evaluation Guide

This guide explains how to evaluate Hugging Face models on both MMLU and MMMU benchmarks using the unified evaluation framework.

## Available Scripts

### Unified Interface
- `quick_eval.py` - **Unified** quick evaluation for both MMLU and MMMU with preset configurations
  - Automatically detects model type (text-only vs vision-language)
  - Single interface for all models
  - 40+ preset model configurations

### Direct Evaluation Scripts
- `evaluate_mmlu.py` - Direct MMLU evaluation script (text-only models)
- `evaluate_mmmu.py` - Direct MMMU evaluation script (vision-language models)

## Quick Start

### List Available Models
```bash
# List all models (both MMLU and MMMU)
python quick_eval.py --list

# List only text-only models (MMLU)
python quick_eval.py --list --benchmark MMLU

# List only vision-language models (MMMU)
python quick_eval.py --list --benchmark MMMU
```

### Unified Evaluation Interface

#### Evaluate Preset Models
```bash
# Evaluate a preset text model on MMLU (auto-detected)
python quick_eval.py mistral-7b --quick

# Evaluate a preset vision-language model on MMMU (auto-detected)
python quick_eval.py llava-1.5-7b --quick

# Force specific benchmark
python quick_eval.py mistral-7b --benchmark MMLU --fast
python quick_eval.py llava-1.5-7b --benchmark MMMU --fast
```

#### Evaluate Custom Models
```bash
# Custom text model (auto-detects MMLU)
python quick_eval.py --custom-model "microsoft/DialoGPT-medium" --quick

# Custom vision-language model (auto-detects MMMU)
python quick_eval.py --custom-model "llava-hf/llava-1.5-13b-hf" --quick

# Force specific benchmark for custom models
python quick_eval.py --custom-model "microsoft/phi-2" --benchmark MMLU --fast
python quick_eval.py --custom-model "Qwen/Qwen-VL-Chat" --benchmark MMMU --fast
```

#### Evaluation with Quantization
```bash
# Text model with quantization
python quick_eval.py --custom-model "meta-llama/Llama-2-13b-hf" --quantization 4bit --quick

# Vision-language model with quantization
python quick_eval.py --custom-model "Qwen/Qwen-VL-Chat" --quantization 4bit --quick
```

### Direct Evaluation (Advanced)

#### MMLU Direct Evaluation
```bash
# Direct evaluation with full control
python evaluate_mmlu.py --model "mistralai/Mistral-7B-v0.1" --output results.json

# With specific subjects
python evaluate_mmlu.py --model "microsoft/phi-2" --subjects high_school_mathematics college_physics

# With quantization
python evaluate_mmlu.py --model "meta-llama/Llama-2-13b-hf" --quantization 4bit
```

#### MMMU Direct Evaluation
```bash
# Direct evaluation with full control
python evaluate_mmmu.py --model "llava-hf/llava-1.5-7b-hf" --output mmmu_results.json

# With specific subjects
python evaluate_mmmu.py --model "Qwen/Qwen-VL-Chat" --subjects Math Physics Chemistry

# With different dataset split
python evaluate_mmmu.py --model "llava-hf/llava-1.5-7b-hf" --split dev --num-samples 5
```

## Available Models

### MMLU Models (Language Models)

The framework includes presets for these model families:

#### Llama Family
- `llama2-7b` - Llama 2 7B base model
- `llama2-13b` - Llama 2 13B base model (with 4bit quantization)
- `llama2-7b-chat` - Llama 2 7B chat model
- `llama2-13b-chat` - Llama 2 13B chat model (with 4bit quantization)
- `codellama-7b` - Code Llama 7B base model

#### Mistral Family
- `mistral-7b` - Mistral 7B base model
- `mistral-7b-instruct` - Mistral 7B instruction-tuned
- `mixtral-8x7b` - Mixtral 8x7B MoE model (with 4bit quantization)
- `mixtral-8x7b-instruct` - Mixtral 8x7B instruction-tuned (with 4bit quantization)

#### Other Popular Models
- `qwen-7b`, `qwen-14b`, `qwen-7b-chat` - Qwen family
- `vicuna-7b`, `vicuna-13b` - Vicuna family
- `falcon-7b`, `falcon-7b-instruct` - Falcon family
- `mpt-7b`, `mpt-7b-chat` - MPT family
- `phi-2` - Microsoft Phi-2 (2.7B parameters)

### MMMU Models (Vision-Language Models)

The framework includes presets for these multimodal model families:

#### Llama 4 Family (NEW!)
- `llama4-scout-instruct` - Llama 4 Scout 17B (16 experts) - Multimodal with 10M context
- `llama4-scout` - Llama 4 Scout 17B (16 experts) - Base multimodal model
- `llama4-maverick-instruct` - Llama 4 Maverick 17B (128 experts) - Advanced multimodal
- `llama4-maverick` - Llama 4 Maverick 17B (128 experts) - Base multimodal model

#### LLaVA Family
- `llava-1.5-7b` - LLaVA 1.5 7B (multimodal)
- `llava-1.5-13b` - LLaVA 1.5 13B (multimodal, with 4bit quantization)
- `llava-1.6-vicuna-7b` - LLaVA 1.6 Vicuna 7B (multimodal)
- `llava-1.6-vicuna-13b` - LLaVA 1.6 Vicuna 13B (multimodal, with 4bit quantization)

#### Qwen-VL Family
- `qwen-vl-chat` - Qwen VL Chat (multimodal)
- `qwen-vl-plus` - Qwen VL Plus (multimodal, with 4bit quantization)

#### InstructBLIP Family
- `instructblip-vicuna-7b` - InstructBLIP Vicuna 7B (multimodal)
- `instructblip-vicuna-13b` - InstructBLIP Vicuna 13B (multimodal, with 4bit quantization)
- `instructblip-flan-t5-xl` - InstructBLIP FLAN-T5 XL (multimodal)
- `instructblip-flan-t5-xxl` - InstructBLIP FLAN-T5 XXL (multimodal, with 4bit quantization)

#### BLIP-2 Family
- `blip2-opt-2.7b` - BLIP-2 OPT 2.7B (multimodal)
- `blip2-opt-6.7b` - BLIP-2 OPT 6.7B (multimodal)
- `blip2-flan-t5-xl` - BLIP-2 FLAN-T5 XL (multimodal)
- `blip2-flan-t5-xxl` - BLIP-2 FLAN-T5 XXL (multimodal, with 4bit quantization)

#### Other Vision-Language Models
- `minigpt4-vicuna-7b`, `minigpt4-vicuna-13b` - MiniGPT-4 family
- `cogvlm-chat` - CogVLM Chat (multimodal, with 4bit quantization)
- `fuyu-8b` - Fuyu 8B (multimodal)
- `git-base`, `git-large` - GiT family (lightweight multimodal)

## Command Line Options

### MMLU Evaluation Options
```bash
python evaluate_mmlu.py \
  --model MODEL               # Required: Hugging Face model name
  --quantization {4bit,8bit}   # Optional: Quantization method
  --subjects SUBJECTS [...]    # Optional: Specific subjects to evaluate
  --num-samples N              # Optional: Number of samples per subject
  --output FILE                # Output file for results
  --device DEVICE              # Device to run on (default: auto)
  --no-chat-template           # Don't use chat template even if available
  --no-trust-remote-code       # Don't trust remote code
```

### MMMU Evaluation Options
```bash
python evaluate_mmmu.py \
  --model MODEL               # Required: Hugging Face vision-language model name
  --quantization {4bit,8bit}   # Optional: Quantization method
  --subjects SUBJECTS [...]    # Optional: Specific subjects to evaluate
  --num-samples N              # Optional: Number of samples per subject
  --split {dev,validation,test} # Dataset split to use
  --output FILE                # Output file for results
  --device DEVICE              # Device to run on (default: auto)
  --no-trust-remote-code       # Don't trust remote code
```

## Evaluation Modes

### 1. Full Evaluation
Evaluates all subjects with all samples:
```bash
# MMLU (57 subjects)
python quick_eval.py mistral-7b

# MMMU (30 subjects)
python quick_eval_mmmu.py llava-1.5-7b
```

### 2. Quick Evaluation
Evaluates all subjects but fewer samples (faster):
```bash
# MMLU: 5 samples per subject
python quick_eval.py mistral-7b --quick

# MMMU: 3 samples per subject
python quick_eval_mmmu.py llava-1.5-7b --quick
```

### 3. Fast Evaluation
Evaluates only key subjects:
```bash
# MMLU: 10 key subjects
python quick_eval.py mistral-7b --fast

# MMMU: 10 key subjects
python quick_eval_mmmu.py llava-1.5-7b --fast
```

### 4. Custom Evaluation
Specify exact subjects and sample counts:
```bash
# MMLU
python quick_eval.py mistral-7b --subjects high_school_mathematics college_physics --samples 10

# MMMU
python quick_eval_mmmu.py llava-1.5-7b --subjects Math Physics Chemistry --samples 5
```

## Model Requirements

### MMLU Models
- Compatible with `AutoModelForCausalLM`
- Can generate text responses
- Have a tokenizer available

### MMMU Models  
- Compatible with `AutoModelForCausalLM` or `AutoModelForVision2Seq`
- Support multimodal inputs (text + images)
- Have a processor or tokenizer available
- Can process PIL Images

### Memory Requirements
- **7B models**: ~14GB VRAM (FP16) or ~7GB with 4bit quantization
- **13B models**: ~26GB VRAM (FP16) or ~13GB with 4bit quantization
- **Larger models**: Use 4bit or 8bit quantization
- **Vision-Language models**: Additional ~2-4GB for vision components

## Dataset Information

### MMLU (Massive Multitask Language Understanding)
- **57 subjects** across 4 categories (STEM, Social Sciences, Humanities, Other)
- **Text-only** multiple choice questions
- Tests knowledge and reasoning in academic domains
- Average accuracy: GPT-4 achieves ~86%

### MMMU (Massive Multi-discipline Multimodal Understanding)  
- **30 subjects** across 6 disciplines (Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering)
- **Multimodal** questions requiring both text and image understanding
- 30 different image types (charts, diagrams, maps, tables, etc.)
- More challenging: GPT-4V achieves ~56%

## Results and Analysis

Results are saved as JSON files containing:
- Overall accuracy across all subjects
- Per-discipline/category breakdowns
- Individual subject scores
- Detailed predictions for error analysis

### MMLU Results Structure
```json
{
  "model": "mistralai/Mistral-7B-v0.1",
  "overall_accuracy": 0.627,
  "total_correct": 8734,
  "total_questions": 13928,
  "category_results": {
    "STEM": {"accuracy": 0.598, "correct": 4523, "total": 7564},
    "Social Sciences": {"accuracy": 0.672, "correct": 2145, "total": 3192},
    ...
  },
  "benchmark": "MMLU"
}
```

### MMMU Results Structure
```json
{
  "model": "llava-hf/llava-1.5-7b-hf",
  "overall_accuracy": 0.365,
  "total_correct": 328,
  "total_questions": 900,
  "discipline_results": {
    "Art & Design": {"accuracy": 0.412, "correct": 89, "total": 216},
    "Science": {"accuracy": 0.334, "correct": 67, "total": 201},
    ...
  },
  "benchmark": "MMMU"
}
```

## Examples

### Basic Evaluations
```bash
# Evaluate Mistral 7B on MMLU (auto-detected)
python quick_eval.py mistral-7b --quick

# Evaluate LLaVA 1.5 7B on MMMU (auto-detected)
python quick_eval.py llava-1.5-7b --quick
```

### Subject-Specific Evaluations
```bash
# MMLU math subjects only
python quick_eval.py mistral-7b --subjects high_school_mathematics college_mathematics

# MMMU science subjects only
python quick_eval.py llava-1.5-7b --subjects Math Physics Chemistry
```

### Custom Models with Quantization
```bash
# Custom text model (auto-detects MMLU)
python quick_eval.py --custom-model "facebook/opt-6.7b" --quantization 4bit --fast

# Custom vision-language model (auto-detects MMMU)
python quick_eval.py --custom-model "Qwen/Qwen-VL-Chat" --quantization 4bit --fast
```

### Benchmark Comparison
```bash
# Compare text models on MMLU
python quick_eval.py mistral-7b --fast
python quick_eval.py phi-2 --fast  
python quick_eval.py falcon-7b --fast

# Compare vision-language models on MMMU
python quick_eval.py llava-1.5-7b --fast
python quick_eval.py qwen-vl-chat --fast
python quick_eval.py instructblip-vicuna-7b --fast
```

### Force Specific Benchmark
```bash
# Force MMLU even for vision models (will likely fail/perform poorly)
python quick_eval.py --custom-model "llava-hf/llava-1.5-7b-hf" --benchmark MMLU --quick

# Force MMMU even for text models (will likely fail)
python quick_eval.py --custom-model "microsoft/phi-2" --benchmark MMMU --quick
```

## Troubleshooting

### Common Issues

1. **Model Access Denied**: Some models require approval on Hugging Face
2. **Out of Memory**: Use `--quantization 4bit` or `--quantization 8bit`
3. **Slow Evaluation**: Use `--quick` or `--fast` modes for testing
4. **Image Processing Errors (MMMU)**: Ensure PIL and requests are installed
5. **Processor Not Found (MMMU)**: Some models may need specific processor implementations

### Performance Tips

- Use quantization for large models
- Start with `--fast` mode to test setup
- Use `--quick` for faster full evaluations
- Monitor GPU memory usage during evaluation
- For MMMU: Smaller batch sizes may be needed due to image processing

## Adding New Models

### Adding Models to the Unified Interface

Edit the `POPULAR_MODELS` dictionary in `quick_eval.py`:

#### For Text-only Models (MMLU)
```python
'new-text-model': {
    'model': 'organization/text-model-name-hf',
    'quantization': '4bit',  # or None
    'description': 'Description of the text model',
    'type': 'text',
    'benchmark': 'MMLU'
},
```

#### For Vision-Language Models (MMMU)
```python
'new-vision-model': {
    'model': 'organization/vision-model-name-hf',
    'quantization': '4bit',  # or None
    'description': 'Description of the vision-language model',
    'type': 'vision',
    'benchmark': 'MMMU'
},
```

The unified interface automatically handles both types of models and routes them to the appropriate evaluation script based on the `benchmark` field.

This allows comprehensive benchmarking across different model families and both text-only and multimodal architectures.

## Llama-4 Evaluation Guide

Meta has released Llama-4, the first natively multimodal models in the Llama family! These models use Mixture-of-Experts (MoE) architecture and support both text and images.

### Llama-4 Models

The Llama-4 collection includes:

- **Llama 4 Scout**: 17B active parameters (109B total), 16 experts
  - Supports up to 10 million token context length
  - Optimized for efficiency, fits on single H100 with 4bit quantization
  
- **Llama 4 Maverick**: 17B active parameters (400B total), 128 experts
  - Advanced multimodal capabilities
  - 1 million token context length
  - State-of-the-art performance on vision-language tasks

### Quick Llama-4 Evaluation
```bash
# Evaluate Llama 4 Scout (instruction-tuned, multimodal)
python quick_eval.py llama4-scout-instruct --quick

# Evaluate Llama 4 Maverick (instruction-tuned, multimodal)  
python quick_eval.py llama4-maverick-instruct --quick

# Fast evaluation on key MMMU subjects
python quick_eval.py llama4-scout-instruct --fast
python quick_eval.py llama4-maverick-instruct --fast
```

### Custom Llama-4 Evaluation
```bash
# Direct model evaluation with custom settings
python quick_eval.py --custom-model "meta-llama/Llama-4-Scout-17B-16E-Instruct" --quantization 4bit --quick

python quick_eval.py --custom-model "meta-llama/Llama-4-Maverick-17B-128E-Instruct" --quantization 4bit --fast
```

### Llama-4 Performance Expectations

Based on Meta's benchmarks, Llama-4 models show strong performance on MMMU:

- **Llama 4 Scout**: Expected ~73% accuracy on MMMU
- **Llama 4 Maverick**: State-of-the-art performance, competing with GPT-4V

The models excel at:
- Visual reasoning and understanding
- Mathematical problem solving with diagrams
- Chart and graph interpretation
- Multi-image reasoning tasks
- Long context multimodal understanding

### Hardware Requirements for Llama-4

- **Scout models**: Single H100 (80GB) with 4bit quantization
- **Maverick models**: Single H100 DGX host or distributed setup
- **Memory**: ~40-80GB VRAM depending on model and quantization
- **Context**: Up to 1M-10M tokens (unprecedented for multimodal models)

These models represent a significant advancement in multimodal AI and are ideal for comprehensive MMMU evaluation. 