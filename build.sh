#!/usr/bin/env bash
set -e

echo "=== Installing CPU-only PyTorch first ==="
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing remaining dependencies ==="
pip install -r requirements.txt

echo "=== Pre-caching embedding model ==="
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model cached.')"

echo "=== Build complete ==="
