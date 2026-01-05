#!/bin/bash
# Setup script for MLOps Assignment Project

set -e

echo "Setting up MLOps Assignment Project..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p models
mkdir -p notebooks
mkdir -p screenshots
mkdir -p logs

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download data
echo "Downloading dataset..."
python src/data/download_data.py

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"

