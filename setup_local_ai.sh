#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure the script is running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root. Please run it with 'sudo' or as the root user."
    exit 1
fi

# Update and upgrade system packages
echo "Updating system packages..."
apt update && apt upgrade -y
if [ $? -ne 0 ]; then
    echo "Error: Failed to update packages."
    exit 1
fi

# Ensure Python 3 and pip are installed
echo "Checking for Python 3..."
if ! command_exists python3; then
    echo "Python 3 not found. Installing Python 3..."
    apt install python3 -y
fi

echo "Checking for pip..."
if ! command_exists pip3; then
    echo "pip not found. Installing pip..."
    apt install python3-pip -y
fi

# Create project directory in the root's home directory
echo "Creating project directory..."
project_dir="/home/$USER/local-ai-cli"
mkdir -p "$project_dir"
cd "$project_dir" || exit

# Set up a Python virtual environment
echo "Setting up a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required Python libraries
echo "Installing required Python libraries (transformers and torch)..."
pip install --upgrade pip
pip install transformers torch

if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python libraries."
    exit 1
fi

# Create the interactive_ai.py script
echo "Creating the interactive AI Python script..."
cat <<EOF >interactive_ai.py
#!/usr/bin/env python3
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo-125M model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
print(f"Loading model {model_name}...")
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a pad token if not available
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if GPU is available and move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Interactive loop
print("Welcome to the interactive AI CLI! Type 'exit' to quit.")
while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Tokenize input and generate AI response, with attention mask
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=1000, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    
    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"AI: {response}")
EOF

# Ensure the Python script is executable
chmod +x interactive_ai.py

# Inform the user and run the Python script
echo "Running the interactive AI Python script..."
./interactive_ai.py

# Deactivate the virtual environment
deactivate

# Script complete
echo "Setup and execution complete. You are now interacting with GPT-Neo-125M!"
