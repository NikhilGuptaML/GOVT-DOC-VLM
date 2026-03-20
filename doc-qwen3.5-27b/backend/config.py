# ============================================================
#  CONFIG — doc-qwen3.5-27b
#  A4000 workstation: 16GB VRAM + 86GB shared = 102GB effective
# ============================================================

# Model to load from Hugging Face
# FP8 variant = same quality as full, fits in 16GB VRAM cleanly
MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"

# Server settings
MODEL_SERVER_URL = "http://localhost:8000/v1"
BACKEND_PORT = 8001

# Toggle: True = use mock (laptop, no model), False = use real model (GPU machine)
USE_MOCK = True

# Image conversion settings
PDF_DPI = 200          # higher = better quality, more VRAM per image
MAX_IMAGE_WIDTH = 2048 # resize if wider than this

# Model inference settings
MAX_NEW_TOKENS = 4096
CONTEXT_LENGTH = 32768  # keep lower to save VRAM, 128K min for full thinking
