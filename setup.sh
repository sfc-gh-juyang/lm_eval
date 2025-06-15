#!/bin/bash

# Llama 4 MMLU Evaluation Setup Script
echo "Setting up Llama 4 MMLU Evaluation Framework..."
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Install dependencies
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x evaluate_mmlu.py
chmod +x evaluate_mmmu.py
chmod +x quick_eval.py
chmod +x test_setup.py

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face: huggingface-cli login"  
echo "2. Request access to Llama 4 models on HuggingFace Hub"
echo "3. Test your setup: python3 test_setup.py"
echo "4. Run evaluation: python3 quick_eval.py llama4-scout-instruct --quick"
echo ""
echo "For more information, see README.md" 