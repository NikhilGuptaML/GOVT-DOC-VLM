# ============================================================
#  CONFIG — doc-qwen3.5-27b (GGUF via llama.cpp)
#  A4000 workstation: 16GB VRAM + 64GB RAM
#  Q4_K_M quant (~16GB) fits cleanly in VRAM
# ============================================================

# ── GGUF model files ──────────────────────────────────────
# Download from: huggingface.co/unsloth/Qwen3.5-VL-27B-GGUF
# Place both files in a /models directory next to this folder.
MODEL_GGUF_PATH = "../models/Qwen3.5-VL-27B-Q4_K_M.gguf"
MMPROJ_GGUF_PATH = "../models/mmproj-BF16.gguf"

# ── llama.cpp settings ────────────────────────────────────
# -1 = offload all layers to GPU. llama.cpp falls back to
# CPU/RAM automatically if VRAM is insufficient.
N_GPU_LAYERS = -1
# Short context for testing — Qwen3.5 supports up to 128K
# but 2048 saves VRAM and is enough for single-page extraction.
N_CTX = 2048

# ── Backend settings ──────────────────────────────────────
BACKEND_PORT = 8001

# Toggle: True = use mock (laptop, no model), False = use real model (GPU machine)
USE_MOCK = True

# ── Image conversion settings ─────────────────────────────
PDF_DPI = 200          # higher = better quality, more VRAM per image
MAX_IMAGE_WIDTH = 2048 # resize if wider than this

# ── Model inference settings ──────────────────────────────
MAX_NEW_TOKENS = 4096
