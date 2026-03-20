# Govt Doc VLM — Qwen 3.5 Document Extractor

PDF scanned document → page images → structured text using Qwen 3.5 VLM models.

## Folder Structure

```
govt-doc-vlm/
├── doc-qwen3.5-27b/          ← 27B model version (fits in 16GB VRAM cleanly)
│   ├── backend/
│   │   ├── config.py         ← ONLY file that differs between folders
│   │   ├── main.py           ← FastAPI app
│   │   ├── pdf_processor.py  ← PDF → images
│   │   ├── model_client.py   ← calls local Qwen model via plain HTTP (no cloud)
│   │   ├── mock_client.py    ← fake output for laptop testing
│   │   └── requirements.txt
│   ├── frontend/             ← React + Vite app
│   └── start_model.sh        ← run this on GPU machine first
│
└── doc-qwen3.5-122b-a10b/    ← 122B model version (uses VRAM + shared RAM)
    ├── backend/              ← same files, config.py has different model name
    ├── frontend/             ← same frontend, port 8002
    └── start_model.sh
```

---

## What is different between the two folders?

| File | 27B | 122B |
|---|---|---|
| `config.py` MODEL_NAME | `Qwen/Qwen3.5-27B-FP8` | `Qwen/Qwen3.5-122B-A10B` |
| `config.py` BACKEND_PORT | `8001` (reference only) | `8002` (reference only) |
| `start_model.sh` | loads 27B | loads 122B |
| Everything else | identical | identical |

---

## Workstation Hardware (A4000)

- 16GB VRAM (internal) + 86GB shared RAM = **102GB effective**
- 27B FP8 model = ~14GB → fits in VRAM cleanly
- 122B model = ~65GB → loads across VRAM + shared RAM via device_map=auto

---

## Step 1 — Laptop Setup (no model, full UI testing)

```bash
# Backend
cd doc-qwen3.5-27b/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --port 8001 --reload
# ⚠️  Always use 'python -m uvicorn' NOT 'uvicorn' directly
# 'uvicorn' alone uses system Python and can't see venv packages

# Frontend (new terminal)
cd doc-qwen3.5-27b/frontend
npm install
npm run dev
```

Open http://localhost:5173 — upload a PDF, real page images show + mock text below. Full flow works.

---

## Step 2 — GPU Workstation Setup (real model)

```bash
# 1. Install GPU libraries (on GPU machine only)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
pip install torchvision accelerate huggingface_hub
# Note: pillow, httpx already installed via requirements.txt — no openai needed

# 2. Start model server (downloads model on first run)
cd doc-qwen3.5-27b
bash start_model.sh

# 3. In config.py — set USE_MOCK = False

# 4. Start backend
cd backend
python -m uvicorn main:app --port 8001 --reload
# ⚠️  Always use 'python -m uvicorn' NOT 'uvicorn' directly

# 5. Start frontend
cd ../frontend
npm install && npm run dev
```

---

## Where are model files stored?

```
~/.cache/huggingface/hub/
├── models--Qwen--Qwen3.5-27B-FP8/     (~14GB)
└── models--Qwen--Qwen3.5-122B-A10B/   (~65GB)
```

Downloaded once. Loaded from cache on every run after.

---

## Pre-download models (optional, before running)

```bash
huggingface-cli download Qwen/Qwen3.5-27B-FP8
huggingface-cli download Qwen/Qwen3.5-122B-A10B
```

---

## No cloud dependencies

This project calls **no external APIs**. Everything runs locally:

```
React UI → FastAPI backend → httpx POST → localhost:8000 → Qwen model on GPU
```

`model_client.py` uses plain `httpx` (comes with FastAPI) to call the local
model server. No `openai` package, no API keys, no internet required at runtime.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `No module named 'pymupdf'` | Running `uvicorn` instead of `python -m uvicorn` | Use `python -m uvicorn main:app --port 8001 --reload` |
| `No module named 'fitz'` | Same as above | Same fix |
| Images not showing | PyMuPDF import failing silently | Same fix — always `python -m uvicorn` |
| `ModuleNotFoundError` on any package | Wrong Python / venv not active | Run `source venv/bin/activate` first |
| `[CONNECTION ERROR]` in extracted text | Model server not running | Run `bash start_model.sh` first, then start backend |
| `[TIMEOUT]` in extracted text | Page image too large | Lower `PDF_DPI` in `config.py` from 200 to 150 |
