#!/usr/bin/env python3
"""
MMLU Evaluation Script for Llama 4
Evaluates Llama 4 models on the Massive Multitask Language Understanding benchmark.
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
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmlu_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MMLU subject categories
MMLU_SUBJECTS = {
    'abstract_algebra': 'STEM',
    'anatomy': 'STEM',
    'astronomy': 'STEM',
    'business_ethics': 'Social Sciences',
    'clinical_knowledge': 'Other',
    'college_biology': 'STEM',
    'college_chemistry': 'STEM',
    'college_computer_science': 'STEM',
    'college_mathematics': 'STEM',
    'college_medicine': 'STEM',
    'college_physics': 'STEM',
    'computer_security': 'STEM',
    'conceptual_physics': 'STEM',
    'econometrics': 'Social Sciences',
    'electrical_engineering': 'STEM',
    'elementary_mathematics': 'STEM',
    'formal_logic': 'Humanities',
    'global_facts': 'Other',
    'high_school_biology': 'STEM',
    'high_school_chemistry': 'STEM',
    'high_school_computer_science': 'STEM',
    'high_school_european_history': 'Humanities',
    'high_school_geography': 'Social Sciences',
    'high_school_government_and_politics': 'Social Sciences',
    'high_school_macroeconomics': 'Social Sciences',
    'high_school_mathematics': 'STEM',
    'high_school_microeconomics': 'Social Sciences',
    'high_school_physics': 'STEM',
    'high_school_psychology': 'Social Sciences',
    'high_school_statistics': 'STEM',
    'high_school_us_history': 'Humanities',
    'high_school_world_history': 'Humanities',
    'human_aging': 'Other',
    'human_sexuality': 'Social Sciences',
    'international_law': 'Humanities',
    'jurisprudence': 'Humanities',
    'logical_fallacies': 'Humanities',
    'machine_learning': 'STEM',
    'management': 'Other',
    'marketing': 'Other',
    'medical_genetics': 'STEM',
    'miscellaneous': 'Other',
    'moral_disputes': 'Humanities',
    'moral_scenarios': 'Humanities',
    'nutrition': 'Other',
    'philosophy': 'Humanities',
    'prehistory': 'Humanities',
    'professional_accounting': 'Other',
    'professional_law': 'Humanities',
    'professional_medicine': 'STEM',
    'professional_psychology': 'Social Sciences',
    'public_relations': 'Social Sciences',
    'security_studies': 'Social Sciences',
    'sociology': 'Social Sciences',
    'us_foreign_policy': 'Social Sciences',
    'virology': 'STEM',
    'world_religions': 'Humanities'
}

class LlamaMMLUEvaluator:
    """Evaluator class for running Llama 4 on MMLU benchmark."""
    
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
        self.tokenizer = None
        self.pipe = None
        
        logger.info(f"Initializing evaluator for {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the Llama 4 model and tokenizer."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if quantization_config is None else None,
            )
            
            # Create pipeline for easier inference
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _format_prompt(self, question: str, choices: List[str]) -> str:
        """Format the MMLU question as a prompt for the model."""
        prompt = f"The following is a multiple choice question. Answer with only the letter (A, B, C, or D) of the correct choice.\n\n"
        prompt += f"Question: {question}\n\n"
        
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}. {choice}\n"
        
        prompt += "\nAnswer:"
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer letter from the model's response."""
        response = response.strip().upper()
        
        # Look for single letter answers
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                return letter
        
        # If no clear answer found, return the first letter in the response
        for char in response:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        # Default to A if no answer found
        return 'A'
    
    def evaluate_subject(self, subject: str, num_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the model on a specific MMLU subject.
        
        Args:
            subject: MMLU subject name
            num_samples: Number of samples to evaluate (all if None)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {subject}...")
        
        try:
            # Load the dataset for this subject
            dataset = load_dataset("cais/mmlu", subject)['test']
            
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            correct = 0
            total = len(dataset)
            predictions = []
            
            for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {subject}")):
                question = example['question']
                choices = example['choices']
                correct_answer = chr(65 + example['answer'])  # Convert 0,1,2,3 to A,B,C,D
                
                # Format prompt
                prompt = self._format_prompt(question, choices)
                
                try:
                    # Generate response
                    outputs = self.pipe(
                        prompt,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_full_text=False
                    )
                    
                    response = outputs[0]['generated_text']
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
                        'response': response
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing question {i} in {subject}: {e}")
                    predictions.append({
                        'question': question,
                        'choices': choices,
                        'correct_answer': correct_answer,
                        'predicted_answer': 'A',  # Default
                        'is_correct': False,
                        'response': f"Error: {e}"
                    })
            
            accuracy = correct / total if total > 0 else 0
            
            result = {
                'subject': subject,
                'category': MMLU_SUBJECTS.get(subject, 'Other'),
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
                'category': MMLU_SUBJECTS.get(subject, 'Other'),
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'predictions': [],
                'error': str(e)
            }
    
    def evaluate_all(self, subjects: Optional[List[str]] = None, num_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the model on all or specified MMLU subjects.
        
        Args:
            subjects: List of subjects to evaluate (all if None)
            num_samples: Number of samples per subject
            
        Returns:
            Dictionary with comprehensive results
        """
        if subjects is None:
            subjects = list(MMLU_SUBJECTS.keys())
        
        logger.info(f"Starting MMLU evaluation on {len(subjects)} subjects")
        
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
            'quantization': self.quantization
        }
        
        logger.info(f"Overall MMLU Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_questions})")
        
        return final_results
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """Print a summary of the evaluation results."""
        print("\n" + "="*60)
        print(f"MMLU Evaluation Results for {results['model']}")
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
    parser = argparse.ArgumentParser(description="Evaluate Llama 4 on MMLU benchmark")
    parser.add_argument("--model", type=str, required=True, 
                       help="Llama 4 model name (e.g., 'meta-llama/Llama-4-Scout-17B-16E-Instruct', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct')")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], 
                       help="Quantization method")
    parser.add_argument("--subjects", nargs="+", 
                       help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--num-samples", type=int, 
                       help="Number of samples per subject (default: all)")
    parser.add_argument("--output", type=str, default="mmlu_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LlamaMMLUEvaluator(
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