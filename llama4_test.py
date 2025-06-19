#!/usr/bin/env python3
"""
Test script for loading and testing Llama-4-Scout model.
Includes error handling for unsupported model configurations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
import warnings

warnings.filterwarnings("ignore")


def test_llama4_scout_basic():
    """Test basic text-only functionality with Llama-4-Scout."""
    model_id = "meta-llama/Llama-4-Scout-17B-16E"

    try:
        print(f"Testing basic text functionality with {model_id}")

        # Try to load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✓ Tokenizer loaded successfully")

        # Try to load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,  # May be needed for newer models
        )
        print("✓ Model loaded successfully")

        # Test basic text generation
        messages = [
            [{"role": "user", "content": "Hello, who are you?"}],
            [{"role": "user", "content": "Hello, who are you?"}],
            [{"role": "user", "content": "Hello, who are you?"}],
            [{"role": "user", "content": "Hello, who are you?"}],
        ]

        # Use tokenizer instead of processor for text-only
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        responses = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs[:, inputs["input_ids"].shape[-1] :]
        ]
        print("response length: ", len(responses))
        # responses2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print("responses2 length: ", len(responses2))

        for i, response in enumerate(responses):
            print(f"✓ Generated response {i}: {response[:80]}...")
        return True

    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False


def test_llama4_scout_multimodal():
    """Test multimodal functionality with Llama-4-Scout (if supported)."""
    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    print(f"Testing multimodal functionality with {model_id}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Processor loaded successfully")

    # Try to load the model - note: Llama4ForConditionalGeneration doesn't exist
    # We'll use AutoModelForCausalLM or AutoModel instead
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✓ Model loaded with AutoModelForCausalLM")
    except Exception:
        model = AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✓ Model loaded with AutoModel")

    # Test text-only generation
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe what you can do."}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    print(type(inputs), "inputs: ", inputs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])
    print(f"✓ Generated response: {response[0][:100]}...")
    return True


def test_fallback_model():
    """Test with a known working model as fallback."""
    model_id = "microsoft/phi-2"

    try:
        print(f"Testing fallback model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "Hello, I am a language model and I can"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Fallback model works: {response}")
        return True

    except Exception as e:
        print(f"✗ Fallback test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Llama-4-Scout Model Tests")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Basic text functionality
    if test_llama4_scout_basic():
        tests_passed += 1

    print("-" * 40)

    # Test 2: Multimodal functionality
    if test_llama4_scout_multimodal():
        tests_passed += 1

    print("-" * 40)

    # Test 3: Fallback model
    if test_fallback_model():
        tests_passed += 1

    print("=" * 60)
    print(f"Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == 0:
        print("❌ All tests failed - there may be environment or model access issues")
    elif tests_passed < total_tests:
        print("⚠️  Some tests failed - Llama-4-Scout may not be fully supported")
    else:
        print("✅ All tests passed!")

    print("=" * 60)


if __name__ == "__main__":
    main()
