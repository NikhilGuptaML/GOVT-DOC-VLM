#!/bin/bash
# ============================================================
#  SETUP & RUN — doc-qwen3.5-27b (HF Inference API)
# ============================================================
#
# What this script does:
#   1. Installs backend dependencies
#   2. Prepares .env from .env.example if missing
#   3. Starts FastAPI backend
#
# Usage:
#   bash start_model.sh
#
# ============================================================

set -e  # exit on any error

echo "============================================================"
echo "  Qwen3.5-27B Hugging Face Setup"
echo "============================================================"
echo ""

# ── Step 1: Install backend dependencies ──────────────────
echo ""
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt --quiet
echo "✓ Backend dependencies installed"

# ── Step 2: Prepare .env ───────────────────────────────────
echo ""
if [ -f ".env" ]; then
    echo "✓ .env already present"
else
    cp .env.example .env
    echo "✓ Created .env from .env.example"
fi

# ── Step 3: Remind token setup ─────────────────────────────
if grep -q "HF_TOKEN=hf_x" .env; then
    echo ""
    echo "⚠️  .env still has placeholder HF_TOKEN."
    echo "   Set HF_TOKEN in .env to use real Hugging Face inference."
    echo "   Without token, backend falls back to mock mode."
fi

echo ""
echo "ℹ️  Runtime mode selection:"
echo "   - HF token present -> real HF model"
echo "   - token missing or USE_MOCK/FORCE_MOCK -> mock mode"
echo ""

# ── Step 4: Start backend ─────────────────────────────────
echo "============================================================"
echo "  Starting backend server on port 8001..."
echo "  Use frontend stream endpoint for page-by-page live updates"
echo "============================================================"
echo ""

cd backend
python -m uvicorn main:app --port 8001 --reload

# ── After running ─────────────────────────────────────────
# 1. Start frontend: cd frontend && npm install && npm run dev
# 2. Open http://localhost:5173
