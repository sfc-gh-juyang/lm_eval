#!/usr/bin/env python3
"""
MMMU Evaluation Script for Vision-Language Models
Evaluates any compatible multimodal model on the MMMU benchmark.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"

import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmmu_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MMMU subject categories (30 subjects across 6 disciplines)
MMMU_DISCIPLINES = {
    'Art & Design': [
        'Art', 'Art_Theory', 'Design', 'Music'
    ],
    'Business': [
        'Accounting', 'Economics', 'Finance', 'Manage', 'Marketing'
    ],
    'Science': [
        'Biology', 'Chemistry', 'Geography', 'Math', 'Physics'
    ],
    'Health & Medicine': [
        'Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine',
        'Pharmacy', 'Public_Health'
    ],
    'Humanities & Social Science': [
        'History', 'Literature', 'Psychology', 'Sociology'
    ],
    'Tech & Engineering': [
        'Agriculture', 'Architecture_and_Engineering', 'Computer_Science',
        'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'
    ]
}

# Flatten subjects with their disciplines
MMMU_SUBJECTS = {}
for discipline, subjects in MMMU_DISCIPLINES.items():
    for subject in subjects:
        MMMU_SUBJECTS[subject] = discipline

class MMMUEvaluator:
    """Evaluator class for running vision-language models on MMMU benchmark."""
    
    def __init__(self, model_name: str, quantization: Optional[str] = None, device: str = "auto", trust_remote_code: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the Hugging Face vision-language model
            quantization: Optional quantization method ('4bit', '8bit', or None)
            device: Device to run on ('auto', 'cuda', 'cpu')
            trust_remote_code: Whether to trust remote code for model loading
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        logger.info(f"Initializing MMMU evaluator for {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the vision-language model and processors."""
        # Configure quantization if specified
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        logger.info("Loading processor/tokenizer...")
        
        # # Handle Llama 4 models specially
        # if "llama-4" in self.model_name.lower() or "llama4" in self.model_name.lower():
        #     # For Llama 4, use tokenizer only for now (processor not fully supported yet)
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         self.model_name,
        #         trust_remote_code=self.trust_remote_code,
        #         padding_side="left"
        #     )
        #     if self.tokenizer.pad_token is None:
        #         self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.processor = None  # Will be handled specially for Llama 4
        # else:
        try:
            # Try to load processor first (most VL models use this)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
        except Exception:
            # Fallback to tokenizer if processor not available
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading vision-language model...")
        
        # Build loading arguments
        loading_args = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.bfloat16 if quantization_config is None else None,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        
        # Only add quantization_config if it's not None
        if quantization_config is not None:
            loading_args["quantization_config"] = quantization_config
        

        # Check if this is a Llama 4 model and use compatible loading
        if "llama-4" in self.model_name.lower() or "llama4" in self.model_name.lower():
            loading_args["attn_implementation"] = "flex_attention"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **loading_args
        )
        
        logger.info("Vision-language model loaded successfully!")
    
    def _process_images(self, example: Dict) -> List[Image.Image]:
        """Extract and process images from the example."""
        images = []
        
        # MMMU dataset can have up to 7 images (image_1 to image_7)
        for i in range(1, 8):
            image_key = f'image_{i}'
            if image_key in example and example[image_key] is not None:
                image = example[image_key]
                
                # Handle different image formats
                if isinstance(image, str):
                    # If it's a URL or path, load it
                    if image.startswith('http'):
                        image = Image.open(requests.get(image, stream=True).raw)
                    else:
                        image = Image.open(image)
                elif hasattr(image, 'convert'):
                    # Already a PIL Image
                    image = image.convert('RGB')
                elif hasattr(image, '__array_interface__') or hasattr(image, 'shape'):
                    # Numpy array or similar
                    image = Image.fromarray(image).convert('RGB')
                elif isinstance(image, dict):
                    # Skip corrupted/invalid image data from batching
                    continue
                else:
                    # Try to convert other formats
                    try:
                        image = Image.fromarray(image).convert('RGB')
                    except:
                        # Skip if can't convert
                        continue
                
                images.append(image)

        return images
    
    def _format_prompt(self, question: str, choices: List[str], images: List[Image.Image]) -> str:
        """Format the MMMU question as a prompt for the vision-language model."""
        
        # Replace image placeholders in question with generic references
        formatted_question = question
        for i in range(1, 8):
            formatted_question = formatted_question.replace(f'<image {i}>', f'[Image {i}]')
        
        base_prompt = f"Answer the following multiple choice question. Look carefully at the image(s) and choose the correct answer.\n\n"
        base_prompt += f"Question: {formatted_question}\n\n"
        base_prompt += "Choices:\n"
        
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            base_prompt += f"{letter}. {choice}\n"
        
        base_prompt += "\nAnswer with only the letter (A, B, C, or D) of the correct choice:"
        
        return base_prompt
    
    def _generate_response(self, prompt: str, images: List[Image.Image]) -> str:
        """Generate response from the vision-language model."""
        # Check if this is a Llama 4 model
        # is_llama4 = "llama-4" in self.model_name.lower() or "llama4" in self.model_name.lower()
        
        if self.processor:
            # Use processor-based approach (most common for VL models)
            inputs = self.processor(
                text=prompt,
                images=images if images else None,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(self.model.device)
            
            # Filter inputs to only include what the model expects
            # Common inputs for vision-language models
            filtered_inputs = {}
            for key in ['input_ids', 'attention_mask', ]:
                if key in inputs:
                    filtered_inputs[key] = inputs[key]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **filtered_inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=1.0,  # Set explicit value to avoid warning
                    top_p=1.0,  # Set explicit value to avoid warning
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
            
            # Handle potential unpacking issues
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get the sequences if it's a tuple
            
            # Decode only the new tokens
            if 'input_ids' in inputs:
                new_tokens = outputs[0][len(inputs['input_ids'][0]):]
            else:
                new_tokens = outputs[0]
            
            response = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        # elif is_llama4:
        #     # Special handling for Llama 4 multimodal models
        #     # For now, convert image info to text description as workaround
        #     if images:
        #         image_text = f"\n[Note: This question includes {len(images)} image(s) that need to be analyzed.]\n"
        #         prompt = image_text + prompt
            
        #     inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        #     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
        #     with torch.no_grad():
        #         outputs = self.model.generate(
        #             **inputs,
        #             max_new_tokens=5,
        #             do_sample=False,
        #             temperature=1.0,  # Set explicit value to avoid warning
        #             top_p=1.0,  # Set explicit value to avoid warning
        #             pad_token_id=self.tokenizer.eos_token_id
        #         )
            
        #     # Handle potential unpacking issues with Llama 4 output
        #     if isinstance(outputs, tuple):
        #         outputs = outputs[0]  # Get the sequences if it's a tuple
            
        #     new_tokens = outputs[0][len(inputs['input_ids'][0]):]
        #     response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        else:
            # Fallback for models without processor
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=1.0,  # Set explicit value to avoid warning
                    top_p=1.0,  # Set explicit value to avoid warning
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Handle potential unpacking issues
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get the sequences if it's a tuple
                
            new_tokens = outputs[0][len(inputs['input_ids'][0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer letter from the model's response."""
        response = response.strip().upper()
        
        # Look for single letter answers
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter
        
        # Default to A if no answer found
        return 'A'
    
    def evaluate_subject(self, subject: str, num_samples: Optional[int] = None, split: str = 'test', batch_size: int = 4) -> Dict:
        """
        Evaluate the model on a specific MMMU subject.
        
        Args:
            subject: MMMU subject name
            num_samples: Number of samples to evaluate (all if None)
            split: Dataset split to use ('test', 'dev', or 'test')
            
        Returns:
            Dictionary with evaluation results
        """
        # Load the dataset for this subject
        dataset = load_dataset("MMMU/MMMU", subject)[split]
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = len(dataset)
        predictions = []
        
        # Process examples individually (batching corrupts image data)
        # But organize progress tracking by batch_size groups
        dataset_list = list(dataset)
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            
            for i in range(batch_start, batch_end):
                example = dataset_list[i]
                question = example['question']
                choices = example['options']
                correct_answer = example['answer']
                
                # Process images
                images = self._process_images(example)
                
                # Format prompt
                prompt = self._format_prompt(question, choices, images)
                
                # Generate response
                response = self._generate_response(prompt, images)
                predicted_answer = self._extract_answer(response)
                
                is_correct = predicted_answer == correct_answer
                if is_correct:
                    correct += 1
                
                predictions.append({
                    'question': question,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': response,
                    'num_images': len(images),
                    'image_types': example.get('img_type', [])
                })
        
        accuracy = correct / total if total > 0 else 0
        
        result = {
            'subject': subject,
            'discipline': MMMU_SUBJECTS.get(subject, 'Other'),
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'split': split,
            'predictions': predictions
        }
        
        logger.info(f"{subject}: {accuracy:.3f} ({correct}/{total})")
        return result
            
    def evaluate_all(self, subjects: Optional[List[str]] = None, num_samples: Optional[int] = None, split: str = 'test', batch_size: int = 16) -> Dict:
        """
        Evaluate the model on all or specified MMMU subjects.
        
        Args:
            subjects: List of subjects to evaluate (all if None)
            num_samples: Number of samples per subject
            split: Dataset split to use
            
        Returns:
            Dictionary with comprehensive results
        """
        if subjects is None:
            subjects = list(MMMU_SUBJECTS.keys())
        
        logger.info(f"Starting MMMU evaluation on {len(subjects)} subjects using {split} split")
        
        results = []
        discipline_results = {}
        
        for subject in subjects:
            result = self.evaluate_subject(subject, num_samples, split, batch_size)
            results.append(result)
            
            # Aggregate by discipline
            discipline = result['discipline']
            if discipline not in discipline_results:
                discipline_results[discipline] = {'correct': 0, 'total': 0, 'subjects': []}
            
            discipline_results[discipline]['correct'] += result['correct']
            discipline_results[discipline]['total'] += result['total']
            discipline_results[discipline]['subjects'].append(result)
        
        # Calculate overall statistics
        total_correct = sum(r['correct'] for r in results)
        total_questions = sum(r['total'] for r in results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        # Calculate discipline accuracies
        for discipline in discipline_results:
            disc_correct = discipline_results[discipline]['correct']
            disc_total = discipline_results[discipline]['total']
            discipline_results[discipline]['accuracy'] = disc_correct / disc_total if disc_total > 0 else 0
        
        final_results = {
            'model': self.model_name,
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'discipline_results': discipline_results,
            'subject_results': results,
            'evaluation_date': datetime.now().isoformat(),
            'num_subjects': len(subjects),
            'quantization': self.quantization,
            'split': split,
            'benchmark': 'MMMU'
        }
        
        logger.info(f"Overall MMMU Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_questions})")
        
        return final_results
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict, output_file: str = None):
        """Print a summary of the evaluation results and optionally save to file."""
        summary_lines = []
        
        # Build summary content
        summary_lines.append("=" * 60)
        summary_lines.append(f"MMMU Evaluation Results for {results['model']}")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Overall Accuracy: {results['overall_accuracy']:.3f} ({results['total_correct']}/{results['total_questions']})")
        summary_lines.append(f"Subjects Evaluated: {results['num_subjects']}")
        summary_lines.append(f"Dataset Split: {results['split']}")
        summary_lines.append(f"Quantization: {results.get('quantization', 'None')}")
        summary_lines.append(f"Evaluation Date: {results['evaluation_date']}")
        
        summary_lines.append("\nDiscipline Breakdown:")
        summary_lines.append("-" * 50)
        for discipline, disc_results in results['discipline_results'].items():
            accuracy = disc_results['accuracy']
            correct = disc_results['correct']
            total = disc_results['total']
            summary_lines.append(f"{discipline:30s}: {accuracy:.3f} ({correct:4d}/{total:4d})")
        
        summary_lines.append("\nTop 10 Subject Scores:")
        summary_lines.append("-" * 50)
        subject_scores = [(r['subject'], r['accuracy']) for r in results['subject_results']]
        subject_scores.sort(key=lambda x: x[1], reverse=True)
        
        for subject, accuracy in subject_scores[:10]:
            summary_lines.append(f"{subject:30s}: {accuracy:.3f}")
        
        if len(subject_scores) > 10:
            summary_lines.append("\nBottom 10 Subject Scores:")
            summary_lines.append("-" * 50)
            for subject, accuracy in subject_scores[-10:]:
                summary_lines.append(f"{subject:30s}: {accuracy:.3f}")
        
        # Print to console
        print("\n" + "\n".join(summary_lines))
        
        # Write to file if output_file is provided
        if output_file:
            summary_file = output_file.replace(".json", "_summary.txt")
            with open(summary_file, "w") as f:
                f.write("\n".join(summary_lines))
            logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate vision-language models on MMMU benchmark")
    parser.add_argument("--model", type=str, required=True, 
                       help="Hugging Face vision-language model name (e.g., 'llava-hf/llava-1.5-7b-hf')")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], 
                       help="Quantization method")
    parser.add_argument("--subjects", nargs="+", 
                       help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--num-samples", type=int, 
                       help="Number of samples per subject (default: all)")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "validation", "test"],
                       help="Dataset split to use")
    parser.add_argument("--output", type=str, default="mmmu_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on")
    parser.add_argument("--no-trust-remote-code", action="store_true",
                       help="Don't trust remote code")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation (default: 4, smaller due to image processing)")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MMMUEvaluator(
        model_name=args.model,
        quantization=args.quantization,
        device=args.device,
        trust_remote_code=not args.no_trust_remote_code
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(
        subjects=args.subjects,
        num_samples=args.num_samples,
        split=args.split,
        batch_size=args.batch_size
    )
    
    # Save and display results
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results, args.output)


if __name__ == "__main__":
    main()
 