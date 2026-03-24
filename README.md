# Govt Doc VLM — Qwen 3.5 Document Extractor

PDF scanned document → page images → structured text using Qwen 3.5 VLM (GGUF via llama.cpp).

## Folder Structure

```
govt-doc-vlm/
└── doc-qwen3.5-27b/
    ├── backend/
    │   ├── config.py         ← GGUF model paths + llama.cpp settings
    │   ├── main.py           ← FastAPI app
    │   ├── pdf_processor.py  ← PDF → images
    │   ├── model_client.py   ← llama-cpp-python in-process inference
    │   ├── mock_client.py    ← fake output for laptop testing
    │   └── requirements.txt
    ├── frontend/             ← React + Vite app
    ├── models/               ← GGUF files (downloaded by start_model.sh)
    │   ├── Qwen3.5-VL-27B-Q4_K_M.gguf   (~16GB)
    │   └── mmproj-BF16.gguf              (~1.2GB)
    └── start_model.sh        ← downloads model + starts backend
```

---

## Model: unsloth/Qwen3.5-VL-27B-GGUF (Q4_K_M)

| Detail | Value |
|--------|-------|
| Source | `unsloth/Qwen3.5-VL-27B-GGUF` on HuggingFace |
| Quant | Q4_K_M — best quality/memory balance for 16GB VRAM |
| Runtime | `llama-cpp-python` with CUDA (in-process, no separate server) |
| VRAM | ~16GB (fits RTX A4000 cleanly) |
| RAM fallback | Automatic — overflow layers spill to system RAM |
| Context | 2048 tokens (configurable in `config.py`, max 128K) |
| Thinking | Disabled (`/no_think` in system prompt) |

---

## Workstation Hardware (A4000)

- **GPU**: RTX A4000 — 16GB VRAM
- **RAM**: 64GB system
- Q4_K_M quant (~16GB) → fits in VRAM cleanly
- `n_gpu_layers=-1` → all layers on GPU, auto-fallback to RAM if needed

---

## Quick Start — Laptop (no GPU, UI testing only)

```bash
cd doc-qwen3.5-27b/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --port 8001 --reload
# ⚠️  USE_MOCK = True in config.py (default) — no model needed

# Frontend (new terminal)
cd doc-qwen3.5-27b/frontend
npm install
npm run dev
```

Open http://localhost:5173 — upload a PDF, page images + mock text. Full flow works without GPU.

---

## Quick Start — GPU Workstation (real model)

### Option A: One command

```bash
cd doc-qwen3.5-27b
bash start_model.sh
```

This will:
1. Install `llama-cpp-python` with CUDA
2. Install backend dependencies
3. Download GGUF model files (~17GB, first run only)
4. Start the FastAPI backend on port 8001

Then set `USE_MOCK = False` in `backend/config.py` and restart.

### Option B: Step by step

```bash
# 1. Install llama-cpp-python with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 2. Download GGUF model files (~17GB total, first run only)
cd doc-qwen3.5-27b
mkdir -p models
huggingface-cli download unsloth/Qwen3.5-VL-27B-GGUF \
    Qwen3.5-VL-27B-Q4_K_M.gguf --local-dir ./models
huggingface-cli download unsloth/Qwen3.5-VL-27B-GGUF \
    mmproj-BF16.gguf --local-dir ./models

# 3. Set USE_MOCK = False in backend/config.py

# 4. Start backend (model loads in-process — no separate model server)
cd backend
python -m uvicorn main:app --port 8001 --reload

# 5. Start frontend (new terminal)
cd ../frontend
npm install && npm run dev
```

> ⚠️ Always use `python -m uvicorn`, never bare `uvicorn` — it won't see venv packages.

---

## Architecture

```
React UI  →  FastAPI backend  →  llama-cpp-python (in-process)  →  Qwen GGUF on GPU
   :5173        :8001              no separate server                 16GB VRAM
```

- **No cloud. No API keys. No internet at runtime.**
- Model loads once at backend startup, stays in GPU memory.
- Each PDF page is processed sequentially — image → base64 → GGUF inference → extracted text.

---

## Config Reference (`backend/config.py`)

| Setting | Default | What it does |
|---------|---------|-------------|
| `MODEL_GGUF_PATH` | `../models/Qwen3.5-VL-27B-Q4_K_M.gguf` | Path to main GGUF model |
| `MMPROJ_GGUF_PATH` | `../models/mmproj-BF16.gguf` | Path to vision projector |
| `N_GPU_LAYERS` | `-1` | Layers on GPU (-1 = all, auto-fallback to RAM) |
| `N_CTX` | `2048` | Context window (tokens). Increase for multi-page docs |
| `USE_MOCK` | `True` | `True` = laptop mock, `False` = real GPU inference |
| `PDF_DPI` | `200` | Page image quality. Lower = faster, less VRAM |
| `MAX_IMAGE_WIDTH` | `2048` | Resize images wider than this |
| `MAX_NEW_TOKENS` | `4096` | Max tokens model can generate per page |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `No module named 'llama_cpp'` | llama-cpp-python not installed | `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` |
| `Model file not found` | GGUF files not downloaded | Run `bash start_model.sh` or download manually (see above) |
| Slow inference (minutes/page) | CPU-only llama-cpp-python build | Reinstall with CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall` |
| `No module named 'pymupdf'` | Running bare `uvicorn` | Use `python -m uvicorn main:app --port 8001 --reload` |
| Images not showing | PyMuPDF import failing silently | Same fix — always `python -m uvicorn` |
| `ModuleNotFoundError` | Wrong Python or venv not active | `source venv/bin/activate` first |
| Empty model output | Vision projector missing | Download `mmproj-BF16.gguf` to `models/` folder |
