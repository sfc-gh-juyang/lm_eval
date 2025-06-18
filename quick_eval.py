#!/usr/bin/env python3
"""
Unified quick evaluation script for both MMLU and MMMU benchmarks
Supports both language models (MMLU) and vision-language models (MMMU).
"""

import argparse
import subprocess
import sys

# Import model configurations and utilities
from models import POPULAR_MODELS, detect_model_type, list_models


def get_fast_subjects(benchmark):
    """Get a subset of subjects for fast evaluation."""
    if benchmark.upper() == "MMLU":
        return [
            "high_school_mathematics",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_physics",
            "college_physics",
            "high_school_chemistry",
            "high_school_biology",
            "computer_security",
            "machine_learning",
            "formal_logic",
        ]
    else:  # MMMU
        return [
            "Math",
            "Physics",
            "Chemistry",
            "Biology",
            "Computer_Science",
            "Economics",
            "Art",
            "Design",
            "History",
            "Psychology",
        ]


def run_evaluation(
    model_key,
    benchmark=None,
    samples=None,
    subjects=None,
    quantization_override=None,
    no_chat_template=False,
    split="validation",
    batch_size=16,
):
    """Run evaluation for a preset model."""

    if model_key not in POPULAR_MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print("Use --list to see available models")
        return False

    config = POPULAR_MODELS[model_key]
    model_name = config["model"]
    quantization = quantization_override or config["quantization"]
    model_benchmark = benchmark or config["benchmark"]

    # Determine output file and evaluation script
    if model_benchmark.upper() == "MMLU":
        output_file = (
            f"{model_key.replace('-', '_')}_mmlu_results_{quantization_override}.json"
        )
        eval_script = "evaluate_mmlu.py"
    else:  # MMMU
        output_file = (
            f"{model_key.replace('-', '_')}_mmmu_results_{quantization_override}.json"
        )
        eval_script = "evaluate_mmmu.py"

    # Build command
    cmd = [sys.executable, eval_script, "--model", model_name, "--output", output_file]

    if model_benchmark.upper() == "MMMU":
        cmd.extend(["--split", split])

    if quantization:
        cmd.extend(["--quantization", quantization])

    if samples:
        cmd.extend(["--num-samples", str(samples)])

    if subjects:
        cmd.extend(["--subjects"] + subjects)

    if no_chat_template and model_benchmark.upper() == "MMLU":
        cmd.append("--no-chat-template")

    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])

    print(f"Running {model_benchmark} evaluation for {model_key}")
    print(f"Model: {model_name}")
    print(f"Benchmark: {model_benchmark}")
    print(f"Description: {config['description']}")
    if quantization:
        print(f"Quantization: {quantization}")
    if model_benchmark.upper() == "MMMU":
        print(f"Split: {split}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print(
            f"\n{model_benchmark} evaluation completed! Results saved to {output_file}"
        )
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
  python quick_eval.py mistral-7b --samples 5
  
  # Evaluate vision-language model on MMMU  
  python quick_eval.py llava-1.5-7b --samples 3
  
  # Custom model evaluation
  python quick_eval.py --custom-model "microsoft/phi-2" --benchmark MMLU --fast
  python quick_eval.py --custom-model "llava-hf/llava-1.5-7b-hf" --benchmark MMMU --fast
        """,
    )

    parser.add_argument(
        "model", nargs="?", help="Model preset name (use --list to see options)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available preset models"
    )
    parser.add_argument(
        "--benchmark",
        choices=["MMLU", "MMMU"],
        help="Specify benchmark type (auto-detected if not provided)",
    )
    parser.add_argument(
        "--custom-model", type=str, help="Use a custom Hugging Face model name"
    )

    parser.add_argument(
        "--fast", action="store_true", help="Fast evaluation (10 selected subjects)"
    )
    parser.add_argument(
        "--quantization", choices=["4bit", "8bit"], help="Override quantization setting"
    )
    parser.add_argument("--samples", type=int, help="Number of samples per subject")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to evaluate")
    parser.add_argument(
        "--split",
        choices=["dev", "validation", "test"],
        default="validation",
        help="Dataset split to use (MMMU only, default: validation)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Don't use chat template even if available (MMLU only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)",
    )

    args = parser.parse_args()

    if args.list:
        list_models(args.benchmark)
        return

    # Handle custom model
    if args.custom_model:
        # Detect model type if benchmark not specified
        model_type = detect_model_type(args.custom_model)
        benchmark = args.benchmark or ("MMMU" if model_type == "vision" else "MMLU")

        output_file = f"custom_{benchmark.lower()}_model_results.json"
        eval_script = f"evaluate_{benchmark.lower()}.py"

        cmd = [
            sys.executable,
            eval_script,
            "--model",
            args.custom_model,
            "--output",
            output_file,
        ]

        if benchmark.upper() == "MMMU":
            cmd.extend(["--split", args.split])

        if args.quantization:
            cmd.extend(["--quantization", args.quantization])

        if args.samples:
            cmd.extend(["--num-samples", str(args.samples)])

        if args.fast:
            cmd.extend(["--subjects"] + get_fast_subjects(benchmark))
        elif args.subjects:
            cmd.extend(["--subjects"] + args.subjects)

        if args.no_chat_template and benchmark.upper() == "MMLU":
            cmd.append("--no-chat-template")

        if args.batch_size:
            cmd.extend(["--batch-size", str(args.batch_size)])

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

    samples = args.samples

    # Fast mode: subset of subjects
    if args.fast:
        model_benchmark = args.benchmark or POPULAR_MODELS[args.model]["benchmark"]
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
        split=args.split,
        batch_size=args.batch_size,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
