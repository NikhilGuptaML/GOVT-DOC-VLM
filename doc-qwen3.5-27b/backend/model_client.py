"""
model_client.py — Local GGUF inference via llama-cpp-python
===========================================================
Loads Qwen3.5-VL-27B GGUF model in-process (no separate model server).
Replaces the old HTTP-to-transformers-serve approach.

Install (CUDA required for GPU acceleration):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
"""

import base64
import os
import sys
import time
from pathlib import Path

from config import (
    MODEL_GGUF_PATH,
    MMPROJ_GGUF_PATH,
    N_GPU_LAYERS,
    N_CTX,
    MAX_NEW_TOKENS,
)

# ════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ════════════════════════════════════════════════════════════
# /no_think disables Qwen3.5's internal thinking mode — keeps
# output clean (no <think>...</think> reasoning traces) and
# saves tokens during extraction.
SYSTEM_PROMPT = (
    "You are a document reconstruction expert. "
    "Analyze this scanned government document page image and extract ALL "
    "content with exact structure preserved.\n\n"
    "Output requirements:\n"
    "- Reproduce all text exactly as it appears, preserving reading order\n"
    "- Reconstruct all tables in markdown table format with all rows and columns\n"
    "- Preserve section headings, sub-headings, numbering\n"
    "- For handwritten content: transcribe exactly, mark unclear parts as [UNCLEAR]\n"
    "- For Hindi/mixed text: transcribe in the original script, do not translate\n"
    "- Do NOT add any commentary, explanations or summaries\n"
    "- Output ONLY the reconstructed document content\n\n"
    "Begin reconstruction: /no_think"
)

# ════════════════════════════════════════════════════════════
#  MODEL LOADING — runs once at import time
# ════════════════════════════════════════════════════════════
# The model is loaded as a module-level singleton so main.py
# only pays the load cost once at startup, not per-request.

_model = None  # lazy-loaded on first call or at import


def _load_model():
    """Load GGUF model with vision support via llama-cpp-python."""
    global _model

    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
    except ImportError:
        print(
            "❌ llama-cpp-python is not installed.\n"
            "   Install with CUDA support:\n"
            '   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python'
        )
        sys.exit(1)

    # Resolve paths relative to this file's directory
    base_dir = Path(__file__).parent
    model_path = str((base_dir / MODEL_GGUF_PATH).resolve())
    mmproj_path = str((base_dir / MMPROJ_GGUF_PATH).resolve())

    # ── Validate model files exist ────────────────────────
    if not os.path.isfile(model_path):
        print(
            f"❌ Model file not found: {model_path}\n"
            f"   Download with:\n"
            f"   huggingface-cli download unsloth/Qwen3.5-VL-27B-GGUF "
            f"Qwen3.5-VL-27B-Q4_K_M.gguf --local-dir ../models"
        )
        sys.exit(1)

    # ── Set up vision handler ─────────────────────────────
    # mmproj (multimodal projector) encodes images into embeddings
    # that the LLM can understand.  Without it, text-only mode.
    chat_handler = None
    if os.path.isfile(mmproj_path):
        print(f"📷 Loading vision projector: {mmproj_path}")
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
    else:
        print(
            f"⚠️  Vision projector not found: {mmproj_path}\n"
            f"   Running in TEXT-ONLY mode (images will be ignored).\n"
            f"   Download with:\n"
            f"   huggingface-cli download unsloth/Qwen3.5-VL-27B-GGUF "
            f"mmproj-BF16.gguf --local-dir ../models"
        )

    # ── Load GGUF model ───────────────────────────────────
    model_size_gb = os.path.getsize(model_path) / (1024 ** 3)
    print(
        f"🔄 Loading GGUF model: {Path(model_path).name} "
        f"({model_size_gb:.1f} GB)\n"
        f"   n_gpu_layers={N_GPU_LAYERS}  n_ctx={N_CTX}"
    )

    start = time.time()
    _model = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        # verbose=True shows per-layer GPU/CPU offloading stats
        verbose=True,
        chat_handler=chat_handler,
    )
    elapsed = time.time() - start
    print(f"✅ Model loaded in {elapsed:.1f}s")

    return _model


def _get_model():
    """Get or lazily load the model singleton."""
    global _model
    if _model is None:
        _load_model()
    return _model


# ════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ════════════════════════════════════════════════════════════

def _encode_image(image_path: str) -> str:
    """Read image file → base64 data URI for llama.cpp multimodal input."""
    with open(image_path, "rb") as f:
        raw = f.read()

    # Detect MIME type from extension
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(ext, "image/png")

    b64_data = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


# ════════════════════════════════════════════════════════════
#  INFERENCE — called by main.py for each page
# ════════════════════════════════════════════════════════════

def process_image(image_path: str, page_number: int) -> str:
    """
    Run GGUF inference on a document page image.
    Called by main.py — same signature as the old HTTP-based client.
    No external server needed — model runs in-process.
    """
    model = _get_model()

    # Warn if image very large — slows inference
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb > 3:
        print(
            f"⚠️  Page {page_number} image is {size_mb:.1f}MB "
            f"— consider lowering PDF_DPI in config.py"
        )

    # ── Build multimodal messages ─────────────────────────
    data_uri = _encode_image(image_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all content from this government document page. "
                        "Preserve tables in markdown format, transcribe handwritten "
                        "amounts exactly, mark unclear parts as [UNCLEAR]."
                    ),
                },
            ],
        },
    ]

    # ── Run inference ─────────────────────────────────────
    try:
        start = time.time()
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
        )
        elapsed = time.time() - start

        content = response["choices"][0]["message"]["content"]

        # Log inference stats
        usage = response.get("usage", {})
        print(
            f"   Page {page_number}: {elapsed:.1f}s "
            f"(prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')} tokens)"
        )

        if not content or not content.strip():
            return (
                f"[Page {page_number}: Model returned empty response — try again]"
            )

        return content

    except Exception as e:
        return f"[Page {page_number} ERROR]: {str(e)}"
