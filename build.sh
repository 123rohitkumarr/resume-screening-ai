#!/bin/bash
# Render build script — runs during deploy
set -e

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Generating seed data..."
python data/seed_data.py

echo "Training classifier model..."
python ml/train_classifier.py

echo "Building FAISS embeddings index..."
python ml/build_embeddings.py

echo "Build complete!"
