#!/usr/bin/env python3
"""
MMMU Evaluation Script for Llama 4
Evaluates Llama 4 models on the Massive Multi-discipline Multimodal Understanding benchmark.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoProcessor, 
    Llama4ForConditionalGeneration,
    BitsAndBytesConfig
)
from datasets import load_dataset
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

# MMMU subject categories
MMMU_SUBJECTS = {
    'Accounting': 'Business',
    'Agriculture': 'Science',
    'Architecture_and_Engineering': 'Engineering',
    'Art': 'Humanities',
    'Art_Theory': 'Humanities',
    'Basic_Medical_Science': 'Medicine',
    'Biology': 'Science',
    'Chemistry': 'Science',
    'Clinical_Medicine': 'Medicine',
    'Computer_Science': 'STEM',
    'Design': 'Humanities',
    'Diagnostics_and_Laboratory_Medicine': 'Medicine',
    'Economics': 'Social Sciences',
    'Electronics': 'Engineering',
    'Energy_and_Power': 'Engineering',
    'Finance': 'Business',
    'Geography': 'Social Sciences',
    'History': 'Humanities',
    'Literature': 'Humanities',
    'Manage': 'Business',
    'Marketing': 'Business',
    'Materials': 'Engineering',
    'Math': 'STEM',
    'Mechanical_Engineering': 'Engineering',
    'Music': 'Humanities',
    'Pharmacy': 'Medicine',
    'Physics': 'Science',
    'Psychology': 'Social Sciences',
    'Public_Health': 'Medicine',
    'Sociology': 'Social Sciences'
}

class LlamaMMMUEvaluator:
    """Evaluator class for running Llama 4 on MMMU benchmark."""
    
    def __init__(self, model_name: str, quantization: Optional[str] = None, device: str = "auto"):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the Llama 4 model to evaluate
            quantization: Optional quantization method ('4bit', '8bit', or None)
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing MMMU evaluator for {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the Llama 4 model and processor."""
        try:
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
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
            
            logger.info("Loading model...")
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if quantization_config is None else None,
                attn_implementation="flex_attention"
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _format_prompt(self, question: str, choices: List[str], images: List = None) -> List[Dict]:
        """Format the MMMU question as a multimodal prompt for the model."""
        content = []
        
        # Add images if present
        if images:
            for img in images:
                if img is not None:
                    content.append({"type": "image", "image": img})
        
        # Add the question and choices
        prompt_text = f"The following is a multiple choice question. Answer with only the letter (A, B, C, D, or E) of the correct choice.\n\n"
        prompt_text += f"Question: {question}\n\n"
        
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D, E
            prompt_text += f"{letter}. {choice}\n"
        
        prompt_text += "\nAnswer:"
        content.append({"type": "text", "text": prompt_text})
        
        messages = [{"role": "user", "content": content}]
        return messages
    
    def _extract_answer(self, response: str, num_choices: int) -> str:
        """Extract the answer letter from the model's response."""
        response = response.strip().upper()
        
        # Valid choices based on number of options
        valid_choices = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D, E...
        
        # Look for single letter answers
        for letter in valid_choices:
            if letter in response:
                return letter
        
        # Default to A if no answer found
        return 'A'
    
    def evaluate_subject(self, subject: str, num_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the model on a specific MMMU subject.
        
        Args:
            subject: MMMU subject name
            num_samples: Number of samples to evaluate (all if None)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating MMMU {subject}...")
        
        try:
            # Load the dataset for this subject
            dataset = load_dataset("MMMU/MMMU", subject)['validation']
            
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = len(dataset)
            predictions = []
            
            for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {subject}")):
                question = example['question']
                choices = example['options']
                correct_answer = example['answer']
                
                # Handle images
                images = []
                for img_key in ['image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'image_7']:
                    if img_key in example and example[img_key] is not None:
                        images.append(example[img_key])
                
                try:
                    # Format prompt
                    messages = self._format_prompt(question, choices, images)
                    
                    # Process inputs
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(self.model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=10,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                        )
                    
                    # Decode response
                    response = self.processor.batch_decode(
                        outputs[:, inputs["input_ids"].shape[-1]:], 
                        skip_special_tokens=True
                    )[0]
                    
                    predicted_answer = self._extract_answer(response, len(choices))
                    
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
                        'num_images': len(images)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing question {i} in {subject}: {e}")
                    predictions.append({
                        'question': question,
                        'choices': choices,
                        'correct_answer': correct_answer,
                        'predicted_answer': 'A',  # Default
                        'is_correct': False,
                        'response': f"Error: {e}",
                        'num_images': len(images)
                    })
            
            accuracy = correct / total if total > 0 else 0
            
            result = {
                'subject': subject,
                'category': MMMU_SUBJECTS.get(subject, 'Other'),
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'predictions': predictions
            }
            
            logger.info(f"{subject}: {accuracy:.3f} ({correct}/{total})")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {subject}: {e}")
            return {
                'subject': subject,
                'category': MMMU_SUBJECTS.get(subject, 'Other'),
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'predictions': [],
                'error': str(e)
            }
    
    def evaluate_all(self, subjects: Optional[List[str]] = None, num_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the model on all or specified MMMU subjects.
        
        Args:
            subjects: List of subjects to evaluate (all if None)
            num_samples: Number of samples per subject
            
        Returns:
            Dictionary with comprehensive results
        """
        if subjects is None:
            subjects = list(MMMU_SUBJECTS.keys())
        
        logger.info(f"Starting MMMU evaluation on {len(subjects)} subjects")
        
        results = []
        category_results = {}
        
        for subject in subjects:
            result = self.evaluate_subject(subject, num_samples)
            results.append(result)
            
            # Aggregate by category
            category = result['category']
            if category not in category_results:
                category_results[category] = {'correct': 0, 'total': 0, 'subjects': []}
            
            category_results[category]['correct'] += result['correct']
            category_results[category]['total'] += result['total']
            category_results[category]['subjects'].append(result)
        
        # Calculate overall statistics
        total_correct = sum(r['correct'] for r in results)
        total_questions = sum(r['total'] for r in results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        # Calculate category accuracies
        for category in category_results:
            cat_correct = category_results[category]['correct']
            cat_total = category_results[category]['total']
            category_results[category]['accuracy'] = cat_correct / cat_total if cat_total > 0 else 0
        
        final_results = {
            'model': self.model_name,
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'category_results': category_results,
            'subject_results': results,
            'evaluation_date': datetime.now().isoformat(),
            'num_subjects': len(subjects),
            'quantization': self.quantization,
            'benchmark': 'MMMU'
        }
        
        logger.info(f"Overall MMMU Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_questions})")
        
        return final_results
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """Print a summary of the evaluation results."""
        print("\n" + "="*60)
        print(f"MMMU Evaluation Results for {results['model']}")
        print("="*60)
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f} ({results['total_correct']}/{results['total_questions']})")
        print(f"Subjects Evaluated: {results['num_subjects']}")
        print(f"Quantization: {results.get('quantization', 'None')}")
        print(f"Evaluation Date: {results['evaluation_date']}")
        
        print("\nCategory Breakdown:")
        print("-" * 40)
        for category, cat_results in results['category_results'].items():
            accuracy = cat_results['accuracy']
            correct = cat_results['correct']
            total = cat_results['total']
            print(f"{category:20s}: {accuracy:.3f} ({correct:4d}/{total:4d})")
        
        print("\nTop 10 Subject Scores:")
        print("-" * 40)
        subject_scores = [(r['subject'], r['accuracy']) for r in results['subject_results']]
        subject_scores.sort(key=lambda x: x[1], reverse=True)
        
        for subject, accuracy in subject_scores[:10]:
            print(f"{subject:30s}: {accuracy:.3f}")
        
        print("\nBottom 10 Subject Scores:")
        print("-" * 40)
        for subject, accuracy in subject_scores[-10:]:
            print(f"{subject:30s}: {accuracy:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama 4 on MMMU benchmark")
    parser.add_argument("--model", type=str, required=True, 
                       help="Llama 4 model name (e.g., 'meta-llama/Llama-4-Scout-17B-16E-Instruct')")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], 
                       help="Quantization method")
    parser.add_argument("--subjects", nargs="+", 
                       help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--num-samples", type=int, 
                       help="Number of samples per subject (default: all)")
    parser.add_argument("--output", type=str, default="mmmu_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LlamaMMMUEvaluator(
        model_name=args.model,
        quantization=args.quantization,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_all(
        subjects=args.subjects,
        num_samples=args.num_samples
    )
    
    # Save and display results
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)


if __name__ == "__main__":
    main() 