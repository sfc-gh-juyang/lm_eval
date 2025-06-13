#!/usr/bin/env python3
"""
Test script to verify Llama 4 MMLU evaluation setup
Run this before attempting full evaluation to catch common issues early.
"""

import sys
import torch
import warnings
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

def test_dependencies():
    """Test that all required dependencies are installed."""
    print("Testing dependencies...")
    
    required_packages = [
        ('torch', torch.__version__),
        ('transformers', None),
        ('datasets', None),
        ('accelerate', None),
        ('bitsandbytes', None),
        ('tqdm', None),
        ('numpy', None),
        ('pandas', None)
    ]
    
    missing_packages = []
    
    for package, version in required_packages:
        try:
            if package == 'torch':
                print(f"✓ {package}: {version}")
            else:
                imported = __import__(package)
                version = getattr(imported, '__version__', 'unknown')
                print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed")
    return True

def test_gpu():
    """Test GPU availability and CUDA setup."""
    print("\nTesting GPU setup...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✓ GPU memory allocation test passed")
        except Exception as e:
            print(f"✗ GPU memory allocation failed: {e}")
            return False
            
        return True
    else:
        print("✗ CUDA not available - will use CPU (very slow)")
        return True  # Not a failure, just a warning

def test_huggingface_auth():
    """Test Hugging Face authentication."""
    print("\nTesting Hugging Face authentication...")
    
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info.get('name', 'Unknown')
        print(f"✓ Logged in as: {username}")
        return True
    except Exception as e:
        print(f"✗ Hugging Face authentication failed: {e}")
        print("Run: huggingface-cli login")
        return False

def test_model_access():
    """Test access to Llama 4 models."""
    print("\nTesting Llama 4 model access...")
    
    model_names = [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    ]
    
    accessible_models = []
    
    for model_name in model_names:
        try:
            # Try to load just the tokenizer to test access
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            print(f"✓ Access granted to {model_name}")
            accessible_models.append(model_name)
        except Exception as e:
            print(f"✗ Cannot access {model_name}: {e}")
    
    if not accessible_models:
        print("✗ No Llama 4 models accessible")
        print("Ensure you have:")
        print("  1. Requested access on Hugging Face")
        print("  2. Accepted the Llama 4 Community License")
        print("  3. Logged in with your HF token")
        return False
    
    print(f"✓ {len(accessible_models)} model(s) accessible")
    return True

def test_dataset_access():
    """Test MMLU dataset access."""
    print("\nTesting MMLU dataset access...")
    
    try:
        # Try loading a small subject to test access
        dataset = load_dataset("cais/mmlu", "abstract_algebra", split="test")
        sample_count = len(dataset)
        print(f"✓ MMLU dataset accessible ({sample_count} samples in abstract_algebra)")
        return True
    except Exception as e:
        print(f"✗ Cannot access MMLU dataset: {e}")
        return False

def test_memory_requirements():
    """Estimate memory requirements and provide recommendations."""
    print("\nChecking memory requirements...")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Available GPU memory: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 80:
            print("✓ Sufficient memory for Llama 4 Maverick (full precision)")
        elif gpu_memory >= 40:
            print("✓ Sufficient memory for Llama 4 Maverick (4-bit quantization)")
            print("  Recommendation: Use --quantization 4bit")
        elif gpu_memory >= 24:
            print("✓ Sufficient memory for Llama 4 Scout")
            print("  Recommendation: Use Scout model or Maverick with 4-bit quantization")
        else:
            print("⚠ Limited GPU memory - evaluation may be slow or fail")
            print("  Recommendations:")
            print("  - Use --quantization 4bit")
            print("  - Try Llama 4 Scout")
            print("  - Consider CPU evaluation (very slow)")
    
    # Check system RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"System RAM: {ram_gb:.1f}GB")
        if ram_gb < 32:
            print("⚠ Low system RAM - consider upgrading for better performance")
    except ImportError:
        print("Cannot check system RAM (psutil not installed)")
    
    return True

def run_mini_evaluation():
    """Run a tiny evaluation test to verify everything works."""
    print("\nRunning mini evaluation test...")
    
    try:
        from evaluate_llama4_mmlu import LlamaMMLUEvaluator
        
        # Try to initialize evaluator with Scout (smallest model)
        print("Initializing evaluator...")
        evaluator = LlamaMMLUEvaluator(
            model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            quantization="4bit",  # Use quantization for safety
            device="auto"
        )
        
        # Run evaluation on just 1 sample from 1 subject
        print("Running test evaluation (1 question)...")
        result = evaluator.evaluate_subject("abstract_algebra", num_samples=1)
        
        if result['total'] > 0:
            accuracy = result['accuracy']
            print(f"✓ Mini evaluation successful! Accuracy: {accuracy:.1%}")
            print("Full evaluation should work correctly.")
            return True
        else:
            print("✗ Mini evaluation failed - no questions processed")
            return False
            
    except Exception as e:
        print(f"✗ Mini evaluation failed: {e}")
        print("Check the error above and resolve before running full evaluation")
        return False

def main():
    """Run all tests and provide summary."""
    print("Llama 4 MMLU Evaluation Setup Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU Setup", test_gpu),
        ("Hugging Face Auth", test_huggingface_auth),
        ("Model Access", test_model_access),
        ("Dataset Access", test_dataset_access),
        ("Memory Requirements", test_memory_requirements),
    ]
    
    passed_tests = 0
    critical_failures = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                if test_name in ["Dependencies", "Hugging Face Auth", "Model Access", "Dataset Access"]:
                    critical_failures.append(test_name)
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            critical_failures.append(test_name)
    
    print("\n" + "=" * 50)
    print(f"Setup Test Results: {passed_tests}/{len(tests)} tests passed")
    
    if critical_failures:
        print(f"\n❌ Critical failures: {', '.join(critical_failures)}")
        print("Please resolve these issues before running evaluation.")
        return False
    
    print("\n✅ Basic setup looks good!")
    
    # Offer to run mini evaluation
    try:
        response = input("\nRun mini evaluation test? (y/n): ").lower().strip()
        if response == 'y':
            if run_mini_evaluation():
                print("\n🎉 All tests passed! Ready for full evaluation.")
            else:
                print("\n⚠ Mini evaluation failed - check configuration.")
                return False
    except KeyboardInterrupt:
        print("\nSkipping mini evaluation test.")
    
    print("\nNext steps:")
    print("1. Run quick evaluation: python quick_eval.py scout --quick")
    print("2. Run full evaluation: python evaluate_llama4_mmlu.py --model meta-llama/Llama-4-Scout")
    print("3. Check README.md for more options")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 