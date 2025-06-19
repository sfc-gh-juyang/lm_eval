#!/usr/bin/env python3
"""
Standalone MMLU evaluator for Llama3-8B with 3-thread pipeline architecture.
Thread 1: Input preprocessing (tokenization)
Thread 2: GPU inference (model.generate)
Thread 3: Output processing (decoding, answer extraction)
"""

import os

import json
import torch
from threading import Thread, Lock
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from datasets import load_dataset
from datetime import datetime
import time

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# All 57 MMLU subjects
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

# Pipeline configuration
BATCH_SIZE = 64  # Batch size for processing
MAX_NEW_TOKENS = 5
MMLU_SUBJECTS = ["all"]
MODEL_IDX = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(MODEL_IDX)



# Shared data structures for thread communication
class ThreadSafeData:
    def __init__(self):
        self.tokenized_batches = deque()
        self.inference_results = deque()
        self.final_results = {}

        self.tokenized_lock = Lock()
        self.inference_lock = Lock()
        self.results_lock = Lock()

        self.preprocessing_done = False
        self.inference_done = False
        self.output_done = False

        self.stats = {"preprocess": 0, "gpu": 0, "output": 0}
        self.stats_lock = Lock()


def create_mmlu_prompts(questions, choices):
    """Create MMLU prompts in batch."""
    batch_prompts = []

    for question, choice_list in zip(questions, choices):
        prompt = "Answer the following multiple choice question with only the letter (A, B, C, or D).\n\n"
        prompt += f"Question: {question}\n\n"

        for i, choice in enumerate(choice_list):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}. {choice}\n"

        prompt += "\nAnswer:"
        batch_prompts.append(prompt)

    return batch_prompts


def format_chat_batch(prompts, tokenizer):
    """Format prompts using chat template for batch processing."""
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
        formatted_prompts.append(formatted)

    return formatted_prompts


def preprocessing_worker(model_name, shared_data):
    """Thread 1: Data loading, prompt formatting, and tokenization."""
    print(f"[PREPROCESS] Starting preprocessing thread for {model_name}")
    
    # Load tokenizer in this thread
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left", use_fast=True
    )
    
    # Fix pad_token setup - ensure it's properly set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without eos_token
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else "[PAD]"
    
    print(f"[PREPROCESS] Tokenizer loaded (pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token})")
    
    batch_id = 0
    
    # Load and process all subjects directly
    for subject in MMLU_SUBJECTS:
        try:
            print(f"[PREPROCESS] Loading dataset for {subject}...")
            dataset = load_dataset("cais/mmlu", subject, split="test")
            print(f"[PREPROCESS] Processing {subject} ({len(dataset)} samples)")
            
            # Process dataset in batches
            for batch_start in range(0, len(dataset), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(dataset))
                batch_data = dataset[batch_start:batch_end]
                
                # Extract batch data
                questions = batch_data["question"]
                choices = batch_data["choices"]
                correct_answers = [chr(65 + ans) for ans in batch_data["answer"]]
                
                # Create and format prompts
                prompts = create_mmlu_prompts(questions, choices)
                formatted_prompts = format_chat_batch(prompts, tokenizer)
                
                # Tokenize batch
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_attention_mask=True,
                )
                
                # Move inputs to GPU
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # Package for GPU thread
                batch_package = {
                    "batch_id": batch_id,
                    "subject": subject,
                    "inputs": inputs,
                    "correct_answers": correct_answers,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                }
                
                # Add to shared data structure
                with shared_data.tokenized_lock:
                    shared_data.tokenized_batches.append(batch_package)
                
                batch_id += 1
                
                with shared_data.stats_lock:
                    shared_data.stats["preprocess"] += 1
        
        except Exception as e:
            print(f"[PREPROCESS] Error processing {subject}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Signal completion
    shared_data.preprocessing_done = True
    print(f"[PREPROCESS] Preprocessing complete - processed {batch_id} batches")


def gpu_inference_worker(model_name, shared_data):
    """Thread 2: GPU model inference only."""
    print(f"[GPU] Starting GPU inference thread for {model_name}")
    
    # Load model in GPU thread
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    # Fix token IDs - ensure they are integers not lists
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]  # Take first element if it's a list
    
    pad_token_id = getattr(model.config, 'pad_token_id', None)
    if pad_token_id is None or isinstance(pad_token_id, list):
        pad_token_id = eos_token_id  # Use eos_token_id as fallback
    
    # Set generation config properly
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = eos_token_id
    
    print(f"[GPU] Model loaded successfully (pad_token_id: {pad_token_id}, eos_token_id: {eos_token_id})")
    
    while True:
        # Check if preprocessing is done and no more batches
        if shared_data.preprocessing_done:
            with shared_data.tokenized_lock:
                if len(shared_data.tokenized_batches) == 0:
                    break
        
        # Get next batch
        batch_package = None
        with shared_data.tokenized_lock:
            if shared_data.tokenized_batches:
                batch_package = shared_data.tokenized_batches.popleft()
        
        if batch_package is None:
            time.sleep(0.1)  # Wait for more batches
            continue
        
        try:
            # Get inputs (already on GPU from preprocessing)
            inputs = batch_package["inputs"]
            
            # Run inference with fixed token IDs
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
            
            inference_time = time.time() - start_time
            
            # Extract only new tokens (keep on GPU for output thread to transfer)
            new_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
            
            # Package for output thread (send GPU tensors directly)
            result_package = {
                "batch_id": batch_package["batch_id"],
                "subject": batch_package["subject"],
                "raw_tokens": new_tokens,  # Keep on GPU, output thread handles transfer
                "correct_answers": batch_package["correct_answers"],
                "batch_start": batch_package["batch_start"],
                "batch_end": batch_package["batch_end"],
                "inference_time": inference_time,
            }
            
            # Add to shared data structure
            with shared_data.inference_lock:
                shared_data.inference_results.append(result_package)
            
            print(
                f"[GPU] Processed batch {batch_package['batch_id']} ({inference_time:.2f}s)"
            )
            
            with shared_data.stats_lock:
                shared_data.stats["gpu"] += 1
                
        except Exception as e:
            print(f"[GPU] Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Signal completion
    shared_data.inference_done = True
    print(f"[GPU] GPU inference complete")


def extract_answers_batch(responses):
    """Extract answers from batch of responses."""
    answers = []
    for response in responses:
        response = response.strip().upper()
        for letter in ["A", "B", "C", "D"]:
            if letter in response:
                answers.append(letter)
                break
        else:
            answers.append("A")  # Default fallback
    return answers


def output_processing_worker(model_name, shared_data):
    """Thread 3: Output processing (CPU transfer, decoding and answer extraction)."""
    print(f"[OUTPUT] Starting output processing thread")

    # Load tokenizer for decoding in output thread
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )

    subject_results = {}
    total_inference_time = 0

    while True:
        # Check if inference is done and no more results
        if shared_data.inference_done:
            with shared_data.inference_lock:
                if len(shared_data.inference_results) == 0:
                    break

        # Get next result
        batch_package = None
        with shared_data.inference_lock:
            if shared_data.inference_results:
                batch_package = shared_data.inference_results.popleft()

        if batch_package is None:
            time.sleep(0.1)  # Wait for more results
            continue

        try:
            # Move raw tokens from GPU to CPU and decode to text
            raw_tokens = batch_package["raw_tokens"].cpu()
            responses = tokenizer.batch_decode(raw_tokens, skip_special_tokens=True)

            # Extract answers and calculate accuracy
            predicted_answers = extract_answers_batch(responses)
            correct_answers = batch_package["correct_answers"]

            correct_count = sum(
                1
                for pred, correct in zip(predicted_answers, correct_answers)
                if pred == correct
            )

            # Track subject results
            subject = batch_package["subject"]
            if subject not in subject_results:
                subject_results[subject] = {"correct": 0, "total": 0}

            subject_results[subject]["correct"] += correct_count
            subject_results[subject]["total"] += len(correct_answers)
            total_inference_time += batch_package["inference_time"]

            # Progress update
            batch_end = batch_package["batch_end"]
            total_samples = subject_results[subject]["total"]
            print(
                f"[OUTPUT] {subject}: Processed {total_samples} samples, "
                f"Batch accuracy: {correct_count}/{len(correct_answers)} "
                f"({correct_count / len(correct_answers):.3f})"
            )

            with shared_data.stats_lock:
                shared_data.stats["output"] += 1

        except Exception as e:
            print(f"[OUTPUT] Error processing batch: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Store final results
    with shared_data.results_lock:
        shared_data.final_results = {
            "subject_results": subject_results,
            "total_inference_time": total_inference_time,
        }

    shared_data.output_done = True
    print(f"[OUTPUT] Output processing complete")


def evaluate_model_threaded(model_name):
    """Evaluate model using 3-thread pipeline."""
    print(f"Starting threaded evaluation of {model_name}")
    print(f"Pipeline configuration: Batch size {BATCH_SIZE}")
    print("=" * 80)

    # Create shared data structure
    shared_data = ThreadSafeData()

    # Start pipeline threads
    threads = []

    # Thread 1: Preprocessing
    t1 = Thread(target=preprocessing_worker, args=(model_name, shared_data))
    t1.start()
    threads.append(t1)

    # Thread 2: GPU Inference
    t2 = Thread(target=gpu_inference_worker, args=(model_name, shared_data))
    t2.start()
    threads.append(t2)

    # Thread 3: Output Processing
    t3 = Thread(target=output_processing_worker, args=(model_name, shared_data))
    t3.start()
    threads.append(t3)

    # Monitor progress
    start_time = time.time()
    last_stats = (0, 0, 0)

    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with shared_data.stats_lock:
            current_stats = (
                shared_data.stats["preprocess"],
                shared_data.stats["gpu"],
                shared_data.stats["output"],
            )

        print(
            f"[MONITOR] Preprocess: {current_stats[0]}, "
            f"GPU: {current_stats[1]}, Output: {current_stats[2]} batches"
        )

        # Check if threads are stuck
        if current_stats == last_stats:
            print(f"[MONITOR] Warning: No progress detected")
        last_stats = current_stats

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Get final results
    with shared_data.results_lock:
        final_results = shared_data.final_results

    if not final_results:
        print("[ERROR] No results generated")
        return None

    total_time = time.time() - start_time

    # Calculate overall statistics
    subject_results = final_results["subject_results"]
    total_correct = sum(r["correct"] for r in subject_results.values())
    total_questions = sum(r["total"] for r in subject_results.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # Compile results
    results = {
        "model": model_name,
        "batch_size": BATCH_SIZE,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "num_subjects": len(subject_results),
        "subject_results": [
            {
                "subject": subject,
                "accuracy": data["correct"] / data["total"],
                "correct": data["correct"],
                "total": data["total"],
            }
            for subject, data in subject_results.items()
        ],
        "evaluation_date": datetime.now().isoformat(),
        "total_time": total_time,
        "inference_time": final_results["total_inference_time"],
        "pipeline_efficiency": final_results["total_inference_time"] / total_time,
    }

    # Save results
    os.makedirs("result", exist_ok=True)
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    output_file = f"result/{model_safe_name}_threaded_mmlu.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print(f"THREADED EVALUATION COMPLETE: {model_name}")
    print(
        f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})"
    )
    print(f"Total Time: {total_time:.1f}s")
    print(f"Inference Time: {final_results['total_inference_time']:.1f}s")
    print(f"Pipeline Efficiency: {results['pipeline_efficiency']:.1%}")
    print(f"Results saved: {output_file}")

    return results


def main():
    """Main evaluation function."""
    print("Llama3-8B MMLU Threaded Evaluator")
    print("=" * 80)
    print(f"Pipeline batch size: {BATCH_SIZE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Total subjects: {len(MMLU_SUBJECTS)}")
    print()
    
    # Evaluate first model
    model_name = MODELS[MODEL_IDX]
    result = evaluate_model_threaded(model_name)
    
    print("\nThreaded evaluation completed!")


if __name__ == "__main__":
    main()
