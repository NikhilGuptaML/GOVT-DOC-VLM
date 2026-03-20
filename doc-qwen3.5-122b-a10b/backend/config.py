# ============================================================
#  CONFIG — doc-qwen3.5-122b-a10b
#  A4000 workstation: 16GB VRAM + 86GB shared = 102GB effective
#  122B loads across VRAM + shared RAM via device_map=auto
# ============================================================

# Model to load from Hugging Face
MODEL_NAME = "Qwen/Qwen3.5-122B-A10B"

# Server settings
MODEL_SERVER_URL = "http://localhost:8000/v1"
BACKEND_PORT = 8002   # different port so both can run simultaneously

# Toggle: True = use mock (laptop, no model), False = use real model (GPU machine)
USE_MOCK = True

# Image conversion settings
PDF_DPI = 200
MAX_IMAGE_WIDTH = 2048

# Model inference settings
MAX_NEW_TOKENS = 4096
CONTEXT_LENGTH = 32768
