#!/bin/bash
# ============================================================
#  START MODEL SERVER — doc-qwen3.5-122b-a10b
#  A4000: 16GB VRAM + 86GB shared = 102GB effective
#  122B loads across VRAM + shared RAM via device_map=auto
#  First download: ~65GB — takes time
# ============================================================

# Install only if transformers serve command not found
if ! python -c "import transformers; from transformers.commands.transformers_cli import main" &>/dev/null; then
  echo "Installing transformers serving support..."
  pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main" \
      torchvision pillow accelerate huggingface_hub --quiet
else
  echo "✓ Transformers already installed, skipping."
fi

echo ""
echo "Starting Qwen3.5-122B-A10B model server on port 8000..."
echo "Model will download from HuggingFace on first run (~65GB)"
echo ""

transformers serve \
    --force-model Qwen/Qwen3.5-122B-A10B \
    --port 8000 \
    --continuous-batching

# ── After model server is running ──────────────────────────
# 1. In config.py set: USE_MOCK = False
# 2. Start backend: uvicorn main:app --port 8002 --reload
# 3. Start frontend: npm run dev
