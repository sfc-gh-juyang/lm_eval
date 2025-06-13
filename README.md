# Llama 4 MMLU & MMMU Evaluation Framework

A comprehensive evaluation framework for testing Meta's Llama 4 models on the Massive Multitask Language Understanding (MMLU) and Massive Multi-discipline Multimodal Understanding (MMMU) benchmarks.

## Overview

This framework provides tools to evaluate the new Llama 4 family of models on MMLU, which consists of 57 tasks spanning mathematics, computer science, natural sciences, social sciences, and humanities. The evaluation measures the model's ability to answer multiple-choice questions across diverse academic domains.

### Supported Models

- **Llama 4 Scout**: 17B active parameters (109B total) with 16 experts, 10M context window
- **Llama 4 Maverick**: 17B active parameters (400B total) with 128 experts, 1M context window  
- **Llama 4 Behemoth**: 288B active parameters (2T total) with 16 experts *(when available)*

## Quick Start

### Installation

1. **Clone or download this evaluation framework**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to Hugging Face (required for Llama 4 access):**
   ```bash
   huggingface-cli login
   ```
   
   You'll need to:
   - Request access to Llama 4 models on Hugging Face
   - Accept the Llama 4 Community License Agreement
   - Use your HF token for authentication

### Quick Evaluation

For the fastest way to get started, use the quick evaluation script:

```bash
# Quick evaluation of Llama 4 Scout (5 samples per subject)
python quick_eval.py scout --quick

# Fast evaluation of Llama 4 Maverick (10 STEM subjects, full questions)
python quick_eval.py maverick --fast

# Full evaluation with 4-bit quantization
python quick_eval.py maverick --quantization 4bit
```

### Full Evaluation

For complete control, use the main evaluation script:

```bash
# Evaluate Llama 4 Scout on all MMLU subjects
python evaluate_llama4_mmlu.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct

# Evaluate with 4-bit quantization for memory efficiency
python evaluate_llama4_mmlu.py --model meta-llama/Llama-4-Maverick-17B-128E-Instruct --quantization 4bit

# Evaluate specific subjects only
python evaluate_llama4_mmlu.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct --subjects high_school_mathematics college_physics

# Evaluate with limited samples per subject (for testing)
python evaluate_llama4_mmlu.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct --num-samples 10
```

## Command Line Options

### Main Evaluation Script (`evaluate_llama4_mmlu.py`)

- `--model`: Model name (required)
  - `meta-llama/Llama-4-Scout-17B-16E-Instruct`
  - `meta-llama/Llama-4-Maverick-17B-128E-Instruct`
  - `meta-llama/Llama-4-Behemoth` (when available)

- `--quantization`: Memory optimization
  - `4bit`: 4-bit quantization (recommended for large models)
  - `8bit`: 8-bit quantization

- `--subjects`: Specific subjects to evaluate (space-separated)
  - Example: `--subjects high_school_mathematics college_physics`
  - Default: All 57 MMLU subjects

- `--num-samples`: Limit samples per subject
  - Useful for quick testing
  - Default: All samples (varies by subject, typically 100-300)

- `--output`: Output file for results
  - Default: `mmlu_results.json`

- `--device`: Device selection
  - `auto`: Automatic device selection (default)
  - `cuda`: Force CUDA GPU
  - `cpu`: Force CPU (not recommended)

### Quick Evaluation Script (`quick_eval.py`)

- `model`: Model variant (required)
  - `scout`: Llama 4 Scout
  - `maverick`: Llama 4 Maverick
  - `behemoth`: Llama 4 Behemoth (when available)

- `--quick`: Quick mode (5 samples per subject)
- `--fast`: Fast mode (10 STEM subjects only)
- `--quantization`: Same as main script
- `--samples`: Custom sample count per subject
- `--subjects`: Custom subject list

## MMLU Subjects

The evaluation covers 57 subjects organized into 4 categories:

### STEM (30 subjects)
- Mathematics: abstract_algebra, college_mathematics, elementary_mathematics, high_school_mathematics, high_school_statistics
- Physics: college_physics, conceptual_physics, high_school_physics
- Chemistry: college_chemistry, high_school_chemistry
- Biology: anatomy, college_biology, high_school_biology, medical_genetics, virology
- Computer Science: college_computer_science, computer_security, high_school_computer_science, machine_learning
- Engineering: electrical_engineering
- Medicine: clinical_knowledge, college_medicine, professional_medicine
- And more...

### Social Sciences (13 subjects)
- Economics: econometrics, high_school_macroeconomics, high_school_microeconomics
- Political Science: high_school_government_and_politics, us_foreign_policy
- Geography: high_school_geography
- Psychology: high_school_psychology, professional_psychology
- And more...

### Humanities (11 subjects)
- History: high_school_european_history, high_school_us_history, high_school_world_history, prehistory
- Philosophy: formal_logic, logical_fallacies, moral_disputes, moral_scenarios, philosophy
- Law: international_law, jurisprudence, professional_law
- Religion: world_religions

### Other (3 subjects)
- Business: business_ethics, management, marketing, professional_accounting
- Health: human_aging, nutrition
- General: global_facts, miscellaneous

## System Requirements

### Hardware Requirements

**Minimum for Llama 4 Scout:**
- GPU: 24GB VRAM (RTX 3090/4090, RTX A5000+)
- RAM: 32GB system memory
- Storage: 150GB free space

**Recommended for Llama 4 Maverick:**
- GPU: 80GB VRAM (H100, A100 80GB) or multi-GPU setup
- RAM: 64GB+ system memory
- Storage: 500GB free space

**For quantized models:**
- 4-bit quantization reduces VRAM requirements by ~75%
- 8-bit quantization reduces VRAM requirements by ~50%

### Software Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+
- Transformers 4.36+

## Output Format

Results are saved as JSON files containing:

```json
{
  "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "overall_accuracy": 0.847,
  "total_correct": 12405,
  "total_questions": 14642,
  "category_results": {
    "STEM": {"accuracy": 0.834, "correct": 7421, "total": 8897},
    "Social Sciences": {"accuracy": 0.863, "correct": 2847, "total": 3298},
    "Humanities": {"accuracy": 0.851, "correct": 1789, "total": 2103},
    "Other": {"accuracy": 0.873, "correct": 348, "total": 344}
  },
  "subject_results": [...],
  "evaluation_date": "2025-01-XX",
  "quantization": "4bit"
}
```

Each subject result includes:
- Individual accuracy scores
- Detailed predictions for each question
- Correct/incorrect classification
- Model responses

## Performance Expectations

Based on Meta's reported benchmarks:

| Model | Expected MMLU Score | Memory Usage | Inference Speed |
|-------|-------------------|--------------|-----------------|
| Llama 4 Scout | ~85% | 65GB (INT4) | Fast |
| Llama 4 Maverick | ~87% | 243GB (INT4) | Medium |
| Llama 4 Behemoth | ~90%+ | TBD | Slow |

*Actual results may vary based on your hardware setup and configuration.*

## Troubleshooting

### Common Issues

1. **Out of Memory Error:**
   - Use `--quantization 4bit` or `--quantization 8bit`
   - Reduce batch size (model loads one question at a time by default)
   - Try running on CPU (very slow but uses system RAM)

2. **Model Access Denied:**
   - Ensure you've requested access to Llama 4 on Hugging Face
   - Verify you're logged in: `huggingface-cli whoami`
   - Check you've accepted the license agreement

3. **Slow Inference:**
   - Ensure GPU acceleration is working: check for CUDA availability
   - Consider using quantization for faster inference
   - Monitor GPU utilization with `nvidia-smi`

4. **Dataset Loading Issues:**
   - Ensure stable internet connection for downloading MMLU
   - Clear Hugging Face cache: `~/.cache/huggingface/`

### Getting Help

- Check the evaluation logs: `mmlu_evaluation.log`
- Verify model loading with a simple test
- Monitor system resources during evaluation
- Enable debug logging for detailed output

## Advanced Usage

### Custom Evaluation Prompts

Modify the `_format_prompt` method in `LlamaMMLUEvaluator` to use different prompt formats:

```python
def _format_prompt(self, question: str, choices: List[str]) -> str:
    # Custom prompt formatting here
    return custom_prompt
```

### Batch Processing

The current implementation processes one question at a time for memory efficiency. For high-memory systems, you can modify the code to use batch processing.

### Integration with Other Benchmarks

The framework can be extended to support other multiple-choice benchmarks by:
1. Modifying the dataset loading code
2. Adjusting the prompt format
3. Updating the answer extraction logic

## License

This evaluation framework is provided under MIT License. Note that:
- Llama 4 models are subject to Meta's Llama 4 Community License
- MMLU dataset has its own academic usage terms
- Ensure compliance with all applicable licenses

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@misc{llama4-mmlu-eval,
  title={Llama 4 MMLU Evaluation Framework},
  year={2025},
  note={Evaluation framework for Meta's Llama 4 models on MMLU benchmark}
}
```

Also cite the original papers:
- Llama 4: [Meta's Llama 4 technical report](https://ai.meta.com/research/publications/)
- MMLU: Hendrycks et al. "Measuring Massive Multitask Language Understanding" (2021)

---

**Happy Evaluating! 🦙📊** 