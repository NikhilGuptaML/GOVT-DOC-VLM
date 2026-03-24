#!/bin/bash
# ============================================================
#  SETUP & RUN — doc-qwen3.5-27b (GGUF via llama.cpp)
#  A4000: 16GB VRAM + 64GB RAM
#  Q4_K_M quant (~16GB) — fits in VRAM cleanly
# ============================================================
#
# What this script does:
#   1. Installs llama-cpp-python with CUDA support
#   2. Downloads GGUF model files from HuggingFace (first run only)
#   3. Installs backend dependencies
#   4. Starts the FastAPI backend (model loads in-process)
#
# Usage:
#   bash start_model.sh
#
# ============================================================

set -e  # exit on any error

MODELS_DIR="./models"
MODEL_FILE="Qwen3.5-VL-27B-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-BF16.gguf"
HF_REPO="unsloth/Qwen3.5-VL-27B-GGUF"

echo "============================================================"
echo "  Qwen3.5-VL-27B GGUF Setup (llama.cpp)"
echo "============================================================"
echo ""

# ── Step 1: Install llama-cpp-python with CUDA ────────────
if python -c "import llama_cpp" &>/dev/null; then
    echo "✓ llama-cpp-python already installed"
else
    echo "📦 Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --quiet
    echo "✓ llama-cpp-python installed"
fi

# ── Step 2: Install backend dependencies ──────────────────
echo ""
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt --quiet
echo "✓ Backend dependencies installed"

# ── Step 3: Download GGUF model files ─────────────────────
echo ""
mkdir -p "$MODELS_DIR"

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "✓ Model file already downloaded: $MODEL_FILE"
else
    echo "⬇️  Downloading $MODEL_FILE (~16GB) — this takes a while on first run..."
    huggingface-cli download "$HF_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR"
    echo "✓ Model downloaded"
fi

if [ -f "$MODELS_DIR/$MMPROJ_FILE" ]; then
    echo "✓ Vision projector already downloaded: $MMPROJ_FILE"
else
    echo "⬇️  Downloading $MMPROJ_FILE (~1.2GB)..."
    huggingface-cli download "$HF_REPO" "$MMPROJ_FILE" --local-dir "$MODELS_DIR"
    echo "✓ Vision projector downloaded"
fi

# ── Step 4: Set USE_MOCK = False ──────────────────────────
echo ""
echo "⚠️  IMPORTANT: Before running, set USE_MOCK = False in backend/config.py"
echo "   (USE_MOCK = True is the default for laptop testing without the model)"
echo ""

# ── Step 5: Start backend ────────────────────────────────
echo "============================================================"
echo "  Starting backend server on port 8001..."
echo "  Model will load in-process (no separate model server needed)"
echo "============================================================"
echo ""

cd backend
python -m uvicorn main:app --port 8001 --reload

# ── After running ─────────────────────────────────────────
# 1. Set USE_MOCK = False in backend/config.py
# 2. Start frontend: cd frontend && npm install && npm run dev
# 3. Open http://localhost:5173
