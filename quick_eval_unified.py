#!/usr/bin/env python3
"""
Unified Quick Evaluation Script for Llama 4
This script provides easy preset configurations for MMLU and MMMU evaluation.
"""

import argparse
import subprocess
import sys
import os

def run_evaluation(model_variant, benchmark="mmlu", samples=None, subjects=None, quantization=None):
    """Run the evaluation with specified parameters."""
    
    # Model name mapping
    model_names = {
        'scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'maverick': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
        'behemoth': 'meta-llama/Llama-4-Behemoth'  # When available
    }
    
    if model_variant not in model_names:
        print(f"Error: Unknown model variant '{model_variant}'")
        print(f"Available variants: {list(model_names.keys())}")
        return False
    
    model_name = model_names[model_variant]
    output_file = f"llama4_{model_variant}_{benchmark}_results.json"
    
    # Build command based on benchmark
    if benchmark == "mmlu":
        script_name = "evaluate_llama4_mmlu.py"
    elif benchmark == "mmmu":
        script_name = "evaluate_llama4_mmmu.py"
    else:
        print(f"Error: Unknown benchmark '{benchmark}'")
        print("Available benchmarks: mmlu, mmmu")
        return False
    
    cmd = [
        sys.executable, script_name,
        "--model", model_name,
        "--output", output_file
    ]
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if samples:
        cmd.extend(["--num-samples", str(samples)])
    
    if subjects:
        cmd.extend(["--subjects"] + subjects)
    
    print(f"Running {benchmark.upper()} evaluation for Llama 4 {model_variant.title()}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nEvaluation completed! Results saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return False
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return False

def get_fast_subjects(benchmark):
    """Get a subset of subjects for fast evaluation."""
    if benchmark == "mmlu":
        return [
            'high_school_mathematics',
            'college_mathematics', 
            'elementary_mathematics',
            'high_school_physics',
            'college_physics',
            'high_school_chemistry',
            'college_chemistry',
            'high_school_biology',
            'college_biology',
            'machine_learning'
        ]
    elif benchmark == "mmmu":
        return [
            'Math',
            'Physics',
            'Chemistry',
            'Biology',
            'Computer_Science',
            'Economics',
            'Finance',
            'Accounting',
            'Art',
            'Music'
        ]
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Quick evaluation for Llama 4 on MMLU or MMMU")
    parser.add_argument("model", choices=["scout", "maverick", "behemoth"],
                       help="Llama 4 model variant to evaluate")
    parser.add_argument("--benchmark", choices=["mmlu", "mmmu"], default="mmlu",
                       help="Benchmark to evaluate on (default: mmlu)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation (5 samples per subject)")
    parser.add_argument("--fast", action="store_true", 
                       help="Fast evaluation (subset of subjects)")
    parser.add_argument("--quantization", choices=["4bit", "8bit"],
                       help="Use quantization for memory efficiency")
    parser.add_argument("--samples", type=int,
                       help="Number of samples per subject")
    parser.add_argument("--subjects", nargs="+",
                       help="Specific subjects to evaluate")
    
    args = parser.parse_args()
    
    # Quick mode: 5 samples per subject
    if args.quick:
        samples = 5
        print(f"Quick mode: Evaluating 5 samples per subject on {args.benchmark.upper()}")
    else:
        samples = args.samples
    
    # Fast mode: subset of subjects
    if args.fast:
        subjects = get_fast_subjects(args.benchmark)
        if subjects:
            print(f"Fast mode: Evaluating {len(subjects)} {args.benchmark.upper()} subjects")
        else:
            print(f"Error: Fast mode not supported for {args.benchmark}")
            sys.exit(1)
    else:
        subjects = args.subjects
    
    # Auto-select quantization for larger models if not specified
    quantization = args.quantization
    if not quantization and args.model in ['maverick', 'behemoth']:
        quantization = '4bit'
        print(f"Auto-selecting 4-bit quantization for {args.model}")
    
    success = run_evaluation(
        model_variant=args.model,
        benchmark=args.benchmark,
        samples=samples,
        subjects=subjects,
        quantization=quantization
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 