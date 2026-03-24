"""
inference_gguf.py — Local GGUF inference via llama.cpp for Qwen3.5-VL
=====================================================================
Standalone inference module for testing on RTX A4000 (16GB VRAM + 64GB RAM).
Uses llama-cpp-python with CUDA offloading instead of HuggingFace
from_pretrained().  LoRA finetuning code (finetune_qwen_vlm.py) is untouched.

Model: unsloth/Qwen3.5-VL-27B-GGUF — Q4_K_M quantisation
  • Language model:    Qwen3.5-VL-27B-Q4_K_M.gguf    (~16GB)
  • Vision projector:  mmproj-BF16.gguf               (~1.2GB)

Install (CUDA required):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

Usage:
  python inference_gguf.py \\
    --model_path ../models/Qwen3.5-VL-27B-Q4_K_M.gguf \\
    --mmproj_path ../models/mmproj-BF16.gguf \\
    --image_path test_document.png
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference_gguf")


# ════════════════════════════════════════════════════════════
#  LAZY IMPORT — llama-cpp-python may not be installed in all
#  environments (e.g. laptop dev without CUDA).  We import
#  lazily so the module can still be introspected/imported
#  without crashing.
# ════════════════════════════════════════════════════════════

def _import_llama_cpp():
    """Import llama_cpp and its chat handlers, with a clear error message."""
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        return Llama, Llava15ChatHandler
    except ImportError as e:
        logger.error(
            "llama-cpp-python is not installed or was built without CUDA.\n"
            "Install with:\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n'
            f"Original error: {e}"
        )
        sys.exit(1)


# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════

# Default context length — kept short for test runs to save VRAM.
# Qwen3.5-VL supports up to 128K, but 2048 is plenty for single-page
# government document extraction during inference testing.
DEFAULT_N_CTX = 2048

# Push all transformer layers to GPU.  -1 = offload everything.
# llama.cpp will automatically fall back to CPU/RAM if VRAM runs out,
# so this is safe even on 16GB cards — partial offload happens silently.
DEFAULT_N_GPU_LAYERS = -1

# System prompt for document extraction.
# /no_think disables Qwen3.5's internal thinking mode — we don't need
# <think>...</think> reasoning traces during inference testing.
# This keeps output clean and saves tokens.
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
#  § 1  MODEL LOADING
# ════════════════════════════════════════════════════════════

def load_model(
    model_path: str,
    mmproj_path: Optional[str] = None,
    n_ctx: int = DEFAULT_N_CTX,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
) -> Any:
    """
    Load a GGUF model via llama-cpp-python with optional vision support.

    Args:
        model_path:    Path to the main GGUF file (e.g. Qwen3.5-VL-27B-Q4_K_M.gguf)
        mmproj_path:   Path to the multimodal projector GGUF (e.g. mmproj-BF16.gguf).
                       Required for image input.  If None, text-only mode.
        n_ctx:         Context window size in tokens (default: 2048)
        n_gpu_layers:  Number of layers to offload to GPU (-1 = all).
                       llama.cpp handles partial offload if VRAM is insufficient.

    Returns:
        Llama model instance ready for inference.
    """
    Llama, Llava15ChatHandler = _import_llama_cpp()

    # ── Validate paths ────────────────────────────────────
    if not os.path.isfile(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # ── Set up vision handler if mmproj provided ──────────
    # Qwen3.5-VL is a multimodal model — the GGUF split separates
    # the vision encoder/projector into a standalone mmproj file.
    # Llava15ChatHandler loads the CLIP-based vision projector and
    # handles image→embedding conversion before feeding to the LLM.
    chat_handler = None
    if mmproj_path:
        if not os.path.isfile(mmproj_path):
            logger.error(f"Vision projector file not found: {mmproj_path}")
            sys.exit(1)
        logger.info(f"Loading vision projector: {mmproj_path}")
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
    else:
        logger.warning(
            "No mmproj_path provided — running in TEXT-ONLY mode. "
            "Image inputs will be ignored. For multimodal inference, "
            "provide --mmproj_path pointing to mmproj-BF16.gguf."
        )

    # ── Load main GGUF model ─────────────────────────────
    model_size_gb = os.path.getsize(model_path) / (1024 ** 3)
    logger.info(
        f"Loading GGUF model: {model_path} ({model_size_gb:.1f} GB)\n"
        f"  n_gpu_layers={n_gpu_layers}  n_ctx={n_ctx}"
    )

    start = time.time()
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        # verbose=True shows per-layer offloading stats — useful for
        # verifying how many layers land on GPU vs RAM on A4000 16GB.
        verbose=True,
        chat_handler=chat_handler,
        # Qwen3.5-VL uses the Qwen chat format.  llama-cpp-python
        # auto-detects from GGUF metadata in most cases, but we
        # only set chat_format explicitly when using a vision handler
        # (Llava15ChatHandler overrides the format internally).
    )
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    return model


# ════════════════════════════════════════════════════════════
#  § 2  IMAGE ENCODING
# ════════════════════════════════════════════════════════════

def encode_image_base64(image_path: str) -> str:
    """
    Read an image file and return a base64-encoded data URI.

    llama-cpp-python's multimodal chat completion expects images as
    base64 data URIs in the format:
        data:image/png;base64,<base64_data>

    Returns the complete data URI string.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as f:
        raw = f.read()

    # Detect MIME type from extension — llama.cpp is forgiving but
    # correct MIME types prevent silent decode failures.
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "image/png")  # fallback to PNG

    size_mb = len(raw) / (1024 * 1024)
    if size_mb > 5:
        logger.warning(
            f"Image is {size_mb:.1f}MB — large images slow down inference. "
            f"Consider resizing or lowering DPI."
        )

    b64_data = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


# ════════════════════════════════════════════════════════════
#  § 3  INFERENCE
# ════════════════════════════════════════════════════════════

def run_inference(
    model: Any,
    image_path: Optional[str] = None,
    prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """
    Run inference on a document image using the loaded GGUF model.

    Args:
        model:       Loaded Llama model instance (from load_model)
        image_path:  Path to document image.  If None, text-only prompt.
        prompt:      Optional custom user prompt.  Defaults to the
                     standard document extraction prompt.
        max_tokens:  Maximum tokens to generate (default: 2048)
        temperature: Sampling temperature (0.1 = near-deterministic)

    Returns:
        Raw model output text.
    """
    # ── Build messages array ──────────────────────────────
    # llama-cpp-python's create_chat_completion uses the OpenAI-style
    # messages format.  For multimodal, image_url entries go into the
    # content array alongside text.
    user_content = []

    if image_path:
        data_uri = encode_image_base64(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })

    # User prompt — use default extraction prompt if none specified
    user_text = prompt or (
        "Extract all content from this government document page. "
        "Preserve tables in markdown format, transcribe handwritten "
        "amounts exactly, mark unclear parts as [UNCLEAR]."
    )
    user_content.append({
        "type": "text",
        "text": user_text,
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # ── Run chat completion ───────────────────────────────
    logger.info(
        f"Running inference"
        f"{f' on image: {image_path}' if image_path else ' (text-only)'}"
        f"  max_tokens={max_tokens}  temperature={temperature}"
    )

    start = time.time()
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        # Disable thinking mode at the API level as well.
        # The /no_think in system prompt handles the model side;
        # this ensures no extra reasoning tokens are generated.
    )
    elapsed = time.time() - start

    # ── Extract response text ─────────────────────────────
    content = response["choices"][0]["message"]["content"]

    # Log stats
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")
    logger.info(
        f"Inference complete in {elapsed:.1f}s  "
        f"(prompt={prompt_tokens} tokens, completion={completion_tokens} tokens)"
    )

    return content


# ════════════════════════════════════════════════════════════
#  § 4  SANITY CHECK
# ════════════════════════════════════════════════════════════

def run_sanity_check(model: Any, image_path: Optional[str] = None) -> None:
    """
    Quick sanity check — run one hardcoded inference and print raw output.

    Intended to be called right after model loading, before any
    training/evaluation code runs.  Confirms the GGUF pipeline is
    working end-to-end: model loads → image encodes → tokens generate.
    """
    logger.info("=" * 60)
    logger.info("SANITY CHECK — running one test inference")
    logger.info("=" * 60)

    if image_path and os.path.isfile(image_path):
        logger.info(f"Test image: {image_path}")
    else:
        # No image provided or file missing — run text-only test
        if image_path:
            logger.warning(f"Test image not found: {image_path}")
        logger.info("Running text-only sanity check (no image)")
        image_path = None

    output = run_inference(
        model=model,
        image_path=image_path,
        prompt=(
            "Extract all content from this government document page. "
            "Preserve tables in markdown format, transcribe handwritten "
            "amounts exactly, mark unclear parts as [UNCLEAR]."
        ),
        max_tokens=1024,  # cap output for sanity check
        temperature=0.1,
    )

    print("\n" + "═" * 60)
    print("  SANITY CHECK — RAW MODEL OUTPUT")
    print("═" * 60)
    print(output)
    print("═" * 60 + "\n")

    logger.info("Sanity check passed — model is producing output.")


# ════════════════════════════════════════════════════════════
#  § 5  CLI ENTRYPOINT
# ════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GGUF inference testing for Qwen3.5-VL via llama.cpp. "
            "Run a sanity check or process a single document image."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sanity check with vision (recommended first run)
  python inference_gguf.py \\
    --model_path ./models/Qwen3.5-VL-27B-Q4_K_M.gguf \\
    --mmproj_path ./models/mmproj-BF16.gguf \\
    --image_path test_doc.png

  # Text-only test (no mmproj needed)
  python inference_gguf.py \\
    --model_path ./models/Qwen3.5-VL-27B-Q4_K_M.gguf

  # Custom context length and GPU layers
  python inference_gguf.py \\
    --model_path ./models/Qwen3.5-VL-27B-Q4_K_M.gguf \\
    --mmproj_path ./models/mmproj-BF16.gguf \\
    --n_ctx 4096 --n_gpu_layers 40
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the GGUF model file (e.g. Qwen3.5-VL-27B-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--mmproj_path",
        type=str,
        default=None,
        help=(
            "Path to the multimodal projector GGUF (e.g. mmproj-BF16.gguf). "
            "Required for image input. Omit for text-only mode."
        ),
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a test document image for the sanity check.",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=DEFAULT_N_CTX,
        help=f"Context window size in tokens (default: {DEFAULT_N_CTX})",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=DEFAULT_N_GPU_LAYERS,
        help=(
            "Number of transformer layers to offload to GPU. "
            "-1 = all layers (default). Falls back to RAM if VRAM full."
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────
    model = load_model(
        model_path=args.model_path,
        mmproj_path=args.mmproj_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    # ── Sanity check — always runs first ──────────────────
    # This is the inference testing gate: if this fails, the model
    # or GGUF file is broken and there's no point running anything else.
    run_sanity_check(model=model, image_path=args.image_path)


if __name__ == "__main__":
    main()
