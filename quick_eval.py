#!/usr/bin/env python3
"""
Unified quick evaluation script for both MMLU and MMMU benchmarks
Supports both language models (MMLU) and vision-language models (MMMU).
"""

import argparse
import subprocess
import sys
import os

# Unified model configurations - both text-only and vision-language models
POPULAR_MODELS = {
    # === MMLU Models (Text-only Language Models) ===
    
    # Llama Family
    'llama2-7b': {
        'model': 'meta-llama/Llama-2-7b-hf',
        'quantization': None,
        'description': 'Llama 2 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'llama2-13b': {
        'model': 'meta-llama/Llama-2-13b-hf',
        'quantization': None,
        'description': 'Llama 2 13B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'llama2-7b-chat': {
        'model': 'meta-llama/Llama-2-7b-chat-hf',
        'quantization': None,
        'description': 'Llama 2 7B chat model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'llama2-13b-chat': {
        'model': 'meta-llama/Llama-2-13b-chat-hf',
        'quantization': None,
        'description': 'Llama 2 13B chat model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'codellama-7b': {
        'model': 'codellama/CodeLlama-7b-hf',
        'quantization': None,
        'description': 'Code Llama 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Llama 3 Family
    'llama3-8b': {
        'model': 'meta-llama/Meta-Llama-3-8B',
        'quantization': None,
        'description': 'Llama 3 8B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'llama3-8b-instruct': {
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'quantization': None,
        'description': 'Llama 3.1 8B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'llama3-70b': {
        'model': 'meta-llama/Meta-Llama-3-70B',
        'quantization': None,
        'description': 'Llama 3 70B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'llama3-70b-instruct': {
        'model': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'quantization': None,
        'description': 'Llama 3.1 70B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Mistral Family
    'mistral-7b': {
        'model': 'mistralai/Mistral-7B-v0.1',
        'quantization': None,
        'description': 'Mistral 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'mistral-7b-instruct': {
        'model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'quantization': None,
        'description': 'Mistral 7B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'mixtral-8x7b': {
        'model': 'mistralai/Mixtral-8x7B-v0.1',
        'quantization': None,
        'description': 'Mixtral 8x7B MoE model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'mixtral-8x7b-instruct': {
        'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'quantization': None,
        'description': 'Mixtral 8x7B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Qwen Family
    'qwen-7b': {
        'model': 'Qwen/Qwen-7B',
        'quantization': None,
        'description': 'Qwen 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen-14b': {
        'model': 'Qwen/Qwen-14B',
        'quantization': None,
        'description': 'Qwen 14B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen-7b-chat': {
        'model': 'Qwen/Qwen-7B-Chat',
        'quantization': None,
        'description': 'Qwen 7B chat model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Qwen & Qwen2 Family
    'qwen2-7b': {
        'model': 'Qwen/Qwen2-7B',
        'quantization': None,
        'description': 'Qwen2 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen2-7b-instruct': {
        'model': 'Qwen/Qwen2-7B-Instruct',
        'quantization': None,
        'description': 'Qwen2 7B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen2-72b': {
        'model': 'Qwen/Qwen2-72B',
        'quantization': None,
        'description': 'Qwen2 72B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen2-72b-instruct': {
        'model': 'Qwen/Qwen2-72B-Instruct',
        'quantization': None,
        'description': 'Qwen2 72B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen2.5-32b-instruct': {
        'model': 'Qwen/Qwen2.5-32B-Instruct',
        'quantization': None,
        'description': 'Qwen2.5 32B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'qwen2.5-72b-instruct': {
        'model': 'Qwen/Qwen2.5-72B-Instruct',
        'quantization': None,
        'description': 'Qwen2.5 72B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Llama 4 Family - Native Multimodal (MoE Architecture)
    'llama4-scout-instruct': {
        'model': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'quantization': None,
        'description': 'Llama 4 Scout 17B (16 experts) - Multimodal with 10M context',
        'type': 'multimodal',
        'benchmark': 'MMMU'
    },
    'llama4-scout': {
        'model': 'meta-llama/Llama-4-Scout-17B-16E',
        'quantization': None,
        'description': 'Llama 4 Scout 17B (16 experts) - Base multimodal model',
        'type': 'multimodal',
        'benchmark': 'MMMU'
    },
    'llama4-maverick-instruct': {
        'model': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
        'quantization': None,
        'description': 'Llama 4 Maverick 17B (128 experts) - Advanced multimodal',
        'type': 'multimodal',
        'benchmark': 'MMMU'
    },
    'llama4-maverick': {
        'model': 'meta-llama/Llama-4-Maverick-17B-128E',
        'quantization': None,
        'description': 'Llama 4 Maverick 17B (128 experts) - Base multimodal model',
        'type': 'multimodal',
        'benchmark': 'MMMU'
    },
    
    # DeepSeek Family
    'deepseek-llm-7b': {
        'model': 'deepseek-ai/deepseek-llm-7b-base',
        'quantization': None,
        'description': 'DeepSeek LLM 7B base',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-llm-7b-chat': {
        'model': 'deepseek-ai/deepseek-llm-7b-chat',
        'quantization': None,
        'description': 'DeepSeek LLM 7B chat',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-coder-6.7b-instruct': {
        'model': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'quantization': None,
        'description': 'DeepSeek Coder 6.7B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-v2': {
        'model': 'deepseek-ai/DeepSeek-V2',
        'quantization': None,
        'description': 'DeepSeek V2 (236B MoE)',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-v3': {
        'model': 'deepseek-ai/DeepSeek-V3',
        'quantization': None,
        'description': 'DeepSeek V3 (671B MoE, 37B activated) - SOTA',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-v3-0324': {
        'model': 'deepseek-ai/DeepSeek-V3-0324',
        'quantization': None,
        'description': 'DeepSeek V3-0324 (671B MoE, enhanced reasoning)',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-r1': {
        'model': 'deepseek-ai/DeepSeek-R1',
        'quantization': None,
        'description': 'DeepSeek R1 (671B MoE, advanced reasoning)',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-r1-distill-32b': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'quantization': None,
        'description': 'DeepSeek R1 Distilled 32B (Reasoning)',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'deepseek-r1-distill-70b': {
        'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'quantization': None,
        'description': 'DeepSeek R1 Distilled 70B (Reasoning)',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Vicuna Family
    'vicuna-7b': {
        'model': 'lmsys/vicuna-7b-v1.5',
        'quantization': None,
        'description': 'Vicuna 7B v1.5',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'vicuna-13b': {
        'model': 'lmsys/vicuna-13b-v1.5',
        'quantization': None,
        'description': 'Vicuna 13B v1.5',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    
    # Other Popular Text Models
    'falcon-7b': {
        'model': 'tiiuae/falcon-7b',
        'quantization': None,
        'description': 'Falcon 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'falcon-7b-instruct': {
        'model': 'tiiuae/falcon-7b-instruct',
        'quantization': None,
        'description': 'Falcon 7B instruction-tuned',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'mpt-7b': {
        'model': 'mosaicml/mpt-7b',
        'quantization': None,
        'description': 'MPT 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'mpt-7b-chat': {
        'model': 'mosaicml/mpt-7b-chat',
        'quantization': None,
        'description': 'MPT 7B chat model',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'phi-2': {
        'model': 'microsoft/phi-2',
        'quantization': None,
        'description': 'Microsoft Phi-2 (2.7B parameters)',
        'type': 'text',
        'benchmark': 'MMLU'
    },
    'stablelm-7b': {
        'model': 'stabilityai/stablelm-base-alpha-7b',
        'quantization': None,
        'description': 'StableLM 7B base model',
        'type': 'text',
        'benchmark': 'MMLU'
    },

    # === MMMU Models (Vision-Language Models) ===
    
    # LLaVA Family
    'llava-1.5-7b': {
        'model': 'llava-hf/llava-1.5-7b-hf',
        'quantization': None,
        'description': 'LLaVA 1.5 7B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'llava-1.5-13b': {
        'model': 'llava-hf/llava-1.5-13b-hf',
        'quantization': None,
        'description': 'LLaVA 1.5 13B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'llava-1.6-vicuna-7b': {
        'model': 'llava-hf/llava-v1.6-vicuna-7b-hf',
        'quantization': None,
        'description': 'LLaVA 1.6 Vicuna 7B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'llava-1.6-vicuna-13b': {
        'model': 'llava-hf/llava-v1.6-vicuna-13b-hf',
        'quantization': None,
        'description': 'LLaVA 1.6 Vicuna 13B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    
    # Qwen-VL Family
    'qwen-vl-chat': {
        'model': 'Qwen/Qwen-VL-Chat',
        'quantization': None,
        'description': 'Qwen VL Chat (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'qwen-vl-plus': {
        'model': 'Qwen/Qwen-VL-Plus',
        'quantization': None,
        'description': 'Qwen VL Plus (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    
    # InstructBLIP Family  
    'instructblip-vicuna-7b': {
        'model': 'Salesforce/instructblip-vicuna-7b',
        'quantization': None,
        'description': 'InstructBLIP Vicuna 7B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'instructblip-vicuna-13b': {
        'model': 'Salesforce/instructblip-vicuna-13b',
        'quantization': None,
        'description': 'InstructBLIP Vicuna 13B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'instructblip-flan-t5-xl': {
        'model': 'Salesforce/instructblip-flan-t5-xl',
        'quantization': None,
        'description': 'InstructBLIP FLAN-T5 XL (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'instructblip-flan-t5-xxl': {
        'model': 'Salesforce/instructblip-flan-t5-xxl',
        'quantization': None,
        'description': 'InstructBLIP FLAN-T5 XXL (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    
    # BLIP-2 Family
    'blip2-opt-2.7b': {
        'model': 'Salesforce/blip2-opt-2.7b',
        'quantization': None,
        'description': 'BLIP-2 OPT 2.7B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'blip2-opt-6.7b': {
        'model': 'Salesforce/blip2-opt-6.7b',
        'quantization': None,
        'description': 'BLIP-2 OPT 6.7B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'blip2-flan-t5-xl': {
        'model': 'Salesforce/blip2-flan-t5-xl',
        'quantization': None,
        'description': 'BLIP-2 FLAN-T5 XL (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'blip2-flan-t5-xxl': {
        'model': 'Salesforce/blip2-flan-t5-xxl',
        'quantization': None,
        'description': 'BLIP-2 FLAN-T5 XXL (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    
    # Other Vision-Language Models
    'minigpt4-vicuna-7b': {
        'model': 'Vision-CAIR/MiniGPT-4',
        'quantization': None,
        'description': 'MiniGPT-4 Vicuna 7B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'minigpt4-vicuna-13b': {
        'model': 'Vision-CAIR/MiniGPT-4-13B',
        'quantization': None,
        'description': 'MiniGPT-4 Vicuna 13B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'cogvlm-chat': {
        'model': 'THUDM/cogvlm-chat-hf',
        'quantization': None,
        'description': 'CogVLM Chat (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'fuyu-8b': {
        'model': 'adept/fuyu-8b',
        'quantization': None,
        'description': 'Fuyu 8B (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'git-base': {
        'model': 'microsoft/git-base',
        'quantization': None,
        'description': 'GiT Base (multimodal, lightweight)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
    'git-large': {
        'model': 'microsoft/git-large',
        'quantization': None,
        'description': 'GiT Large (multimodal)',
        'type': 'vision',
        'benchmark': 'MMMU'
    },
}

def detect_model_type(model_name):
    """Detect if a model is text-only or vision-language based on its name."""
    vision_indicators = [
        'llava', 'qwen-vl', 'instructblip', 'blip', 'minigpt', 'cogvlm', 
        'fuyu', 'git-', 'flamingo', 'kosmos', 'gpt4v', 'gpt-4v'
    ]
    
    model_lower = model_name.lower()
    for indicator in vision_indicators:
        if indicator in model_lower:
            return 'vision'
    return 'text'

def list_models(benchmark_filter=None):
    """List all available preset models, optionally filtered by benchmark."""
    if benchmark_filter:
        print(f"Available preset models for {benchmark_filter}:")
    else:
        print("Available preset models (MMLU & MMMU):")
    print("=" * 70)
    
    # Separate models by benchmark
    mmlu_models = {k: v for k, v in POPULAR_MODELS.items() if v['benchmark'] == 'MMLU'}
    mmmu_models = {k: v for k, v in POPULAR_MODELS.items() if v['benchmark'] == 'MMMU'}
    
    if not benchmark_filter or benchmark_filter.upper() == 'MMLU':
        print(f"\n📝 MMLU Models (Text-only Language Models) - {len(mmlu_models)} models:")
        print("-" * 60)
        
        # Group MMLU models by family
        families = {}
        for key, config in mmlu_models.items():
            family = key.split('-')[0] if '-' in key else 'other'
            # Handle special cases for family names
            if 'llama3' in key: family = 'llama 3'
            if 'qwen2' in key or 'qwen' in key: family = 'qwen series'
            if 'deepseek' in key: family = 'deepseek'

            if family not in families:
                families[family] = []
            families[family].append((key, config))
        
        for family, models in families.items():
            print(f"\n{family.title()} Family:")
            for key, config in models:
                quant_info = f" (with {config['quantization']})" if config['quantization'] else ""
                print(f"  {key:25s} - {config['description']}{quant_info}")
    
    if not benchmark_filter or benchmark_filter.upper() == 'MMMU':
        print(f"\n🖼️  MMMU Models (Vision-Language Models) - {len(mmmu_models)} models:")
        print("-" * 60)
        
    # Group MMMU models by family
    families = {}
    for key, config in mmmu_models.items():
        family = key.split('-')[0] if '-' in key else 'other'
        if family not in families:
            families[family] = []
        families[family].append((key, config))
    
    for family, models in families.items():
        print(f"\n{family.title()} Family:")
        for key, config in models:
            quant_info = f" (with {config['quantization']})" if config['quantization'] else ""
            print(f"  {key:25s} - {config['description']}{quant_info}")

def get_fast_subjects(benchmark):
    """Get a subset of subjects for fast evaluation."""
    if benchmark.upper() == 'MMLU':
        return [
            'high_school_mathematics',
            'college_mathematics', 
            'elementary_mathematics',
            'high_school_physics',
            'college_physics',
            'high_school_chemistry',
            'high_school_biology',
            'computer_security',
            'machine_learning',
            'formal_logic'
        ]
    else:  # MMMU
        return [
            'Math',
            'Physics', 
            'Chemistry',
            'Biology',
            'Computer_Science',
            'Economics',
            'Art',
            'Design',
            'History',
            'Psychology'
        ]

def run_evaluation(model_key, benchmark=None, samples=None, subjects=None, quantization_override=None, 
                  no_chat_template=False, split="validation"):
    """Run evaluation for a preset model."""
    
    if model_key not in POPULAR_MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print("Use --list to see available models")
        return False
    
    config = POPULAR_MODELS[model_key]
    model_name = config['model']
    quantization = quantization_override or config['quantization']
    model_benchmark = benchmark or config['benchmark']
    
    # Determine output file and evaluation script
    if model_benchmark.upper() == 'MMLU':
        output_file = f"{model_key.replace('-', '_')}_mmlu_results.json"
        eval_script = "evaluate_mmlu.py"
    else:  # MMMU
        output_file = f"{model_key.replace('-', '_')}_mmmu_results.json"
        eval_script = "evaluate_mmmu.py"
    
    # Build command
    cmd = [sys.executable, eval_script, "--model", model_name, "--output", output_file]
    
    if model_benchmark.upper() == 'MMMU':
        cmd.extend(["--split", split])
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if samples:
        cmd.extend(["--num-samples", str(samples)])
    
    if subjects:
        cmd.extend(["--subjects"] + subjects)
    
    if no_chat_template and model_benchmark.upper() == 'MMLU':
        cmd.append("--no-chat-template")
    
    print(f"Running {model_benchmark} evaluation for {model_key}")
    print(f"Model: {model_name}")
    print(f"Benchmark: {model_benchmark}")
    print(f"Description: {config['description']}")
    if quantization:
        print(f"Quantization: {quantization}")
    if model_benchmark.upper() == 'MMMU':
        print(f"Split: {split}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{model_benchmark} evaluation completed! Results saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return False
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Unified quick evaluation for both MMLU (text) and MMMU (vision-language) benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python quick_eval.py --list
  
  # List only MMLU models
  python quick_eval.py --list --benchmark MMLU
  
  # Evaluate text model on MMLU
  python quick_eval.py mistral-7b --quick
  
  # Evaluate vision-language model on MMMU  
  python quick_eval.py llava-1.5-7b --quick
  
  # Custom model evaluation
  python quick_eval.py --custom-model "microsoft/phi-2" --benchmark MMLU --fast
  python quick_eval.py --custom-model "llava-hf/llava-1.5-7b-hf" --benchmark MMMU --fast
        """
    )
    
    parser.add_argument("model", nargs='?', 
                       help="Model preset name (use --list to see options)")
    parser.add_argument("--list", action="store_true",
                       help="List available preset models")
    parser.add_argument("--benchmark", choices=["MMLU", "MMMU"], 
                       help="Specify benchmark type (auto-detected if not provided)")
    parser.add_argument("--custom-model", type=str,
                       help="Use a custom Hugging Face model name")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation (5 samples for MMLU, 3 for MMMU)")
    parser.add_argument("--fast", action="store_true", 
                       help="Fast evaluation (10 selected subjects)")
    parser.add_argument("--quantization", choices=["4bit", "8bit"],
                       help="Override quantization setting")
    parser.add_argument("--samples", type=int,
                       help="Number of samples per subject")
    parser.add_argument("--subjects", nargs="+",
                       help="Specific subjects to evaluate")
    parser.add_argument("--split", choices=["dev", "validation", "test"], default="validation",
                       help="Dataset split to use (MMMU only, default: validation)")
    parser.add_argument("--no-chat-template", action="store_true",
                       help="Don't use chat template even if available (MMLU only)")
    
    args = parser.parse_args()
    
    if args.list:
        list_models(args.benchmark)
        return
    
    # Handle custom model
    if args.custom_model:
        # Detect model type if benchmark not specified
        model_type = detect_model_type(args.custom_model)
        benchmark = args.benchmark or ('MMMU' if model_type == 'vision' else 'MMLU')
        
        output_file = f"custom_{benchmark.lower()}_model_results.json"
        eval_script = f"evaluate_{benchmark.lower()}.py"
        
        cmd = [sys.executable, eval_script, "--model", args.custom_model, "--output", output_file]
        
        if benchmark.upper() == 'MMMU':
            cmd.extend(["--split", args.split])
        
        if args.quantization:
            cmd.extend(["--quantization", args.quantization])
        
        if args.quick:
            samples = 3 if benchmark.upper() == 'MMMU' else 5
            cmd.extend(["--num-samples", str(samples)])
        elif args.samples:
            cmd.extend(["--num-samples", str(args.samples)])
        
        if args.fast:
            cmd.extend(["--subjects"] + get_fast_subjects(benchmark))
        elif args.subjects:
            cmd.extend(["--subjects"] + args.subjects)
        
        if args.no_chat_template and benchmark.upper() == 'MMLU':
            cmd.append("--no-chat-template")
        
        print(f"Running custom {benchmark} model evaluation: {args.custom_model}")
        print(f"Detected model type: {model_type}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n{benchmark} evaluation completed! Results saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
            sys.exit(1)
        return
    
    if not args.model:
        parser.error("Please specify a model or use --list to see available options")
    
    # Quick mode samples
    if args.quick:
        # Determine benchmark first
        model_benchmark = args.benchmark or POPULAR_MODELS[args.model]['benchmark']
        samples = 3 if model_benchmark.upper() == 'MMMU' else 5
        print(f"Quick mode: Evaluating {samples} samples per subject")
    else:
        samples = args.samples
    
    # Fast mode: subset of subjects
    if args.fast:
        model_benchmark = args.benchmark or POPULAR_MODELS[args.model]['benchmark']
        subjects = get_fast_subjects(model_benchmark)
        print(f"Fast mode: Evaluating {len(subjects)} selected subjects")
    else:
        subjects = args.subjects
    
    success = run_evaluation(
        model_key=args.model,
        benchmark=args.benchmark,
        samples=samples,
        subjects=subjects,
        quantization_override=args.quantization,
        no_chat_template=args.no_chat_template,
        split=args.split
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 