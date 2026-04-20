"""Hugging Face inference client for page-level document extraction."""

import base64
import os
import re
import time
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

from config import (
        HF_BASE_URL,
        HF_ENABLE_THINKING,
        HF_MAX_TOKENS,
        HF_MODEL_ID,
        HF_TEMPERATURE,
        HF_TIMEOUT_SECONDS,
        get_hf_token,
)

# ════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a document reconstruction expert. Analyze this scanned government "
    "document page image and extract all content with exact structure preserved.\n\n"
    "Output requirements:\n"
    "- Preserve original reading order\n"
    "- Reconstruct tables in markdown format with all rows and columns\n"
    "- Preserve headings, labels, numbering, and formatting signals\n"
    "- For handwritten text, transcribe exactly and mark uncertain spans as [UNCLEAR]\n"
    "- For Hindi or mixed-language text, keep original script (no translation)\n"
    "- Keep final answer focused on reconstructed content only\n\n"
    "You may reason in <think>...</think> before the final answer."
)

_client = None
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    _client = OpenAI(api_key=token, base_url=HF_BASE_URL)
    return _client


# ════════════════════════════════════════════════════════════
#  IMAGE PROCESSING
# ════════════════════════════════════════════════════════════

def _encode_image(image_path: str) -> str:
    """Read image file and convert it to a base64 data URI."""
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


def _normalize_text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content)


def _extract_reasoning_and_answer(content_text: str, reasoning_field: str | None = None) -> tuple[str, str]:
    if reasoning_field and reasoning_field.strip():
        return reasoning_field.strip(), content_text.strip()

    matches = _THINK_RE.findall(content_text)
    if not matches:
        return "", content_text.strip()

    reasoning = "\n\n".join(chunk.strip() for chunk in matches if chunk.strip())
    answer = _THINK_RE.sub("", content_text).strip()
    return reasoning, answer


def _format_error(page_number: int, message: str) -> dict:
    clean = message.strip() if message else "Unknown inference error"
    return {
        "extracted_text": f"[Page {page_number} ERROR]: {clean}",
        "reasoning_text": "",
        "error": clean,
    }


def _chat_completion(client: OpenAI, messages: list[dict]):
    request_kwargs = {
        "model": HF_MODEL_ID,
        "messages": messages,
        "max_tokens": HF_MAX_TOKENS,
        "temperature": HF_TEMPERATURE,
    }

    if HF_ENABLE_THINKING:
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    try:
        return client.with_options(timeout=HF_TIMEOUT_SECONDS).chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        # Some providers reject chat_template_kwargs; retry once without it.
        if "extra_body" in request_kwargs:
            request_kwargs.pop("extra_body")
            return client.with_options(timeout=HF_TIMEOUT_SECONDS).chat.completions.create(**request_kwargs)
        raise exc


# ════════════════════════════════════════════════════════════
#  INFERENCE — called by main.py for each page
# ════════════════════════════════════════════════════════════

def process_image(image_path: str, page_number: int) -> dict:
    """
    Run HF inference on a single document page.
    Returns extracted text and optional reasoning text.
    """
    try:
        client = _get_client()
    except Exception as exc:
        return _format_error(page_number, str(exc))

    # Large images can trigger provider payload limits.
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb > 3:
        print(
            f"⚠️  Page {page_number} image is {size_mb:.1f}MB "
            "— consider lowering PDF_DPI in config.py"
        )

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
                    "text": "Extract and reconstruct this page faithfully.",
                },
            ],
        },
    ]

    try:
        start = time.time()
        response = _chat_completion(client, messages)
        elapsed = time.time() - start

        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)
        content_text = _normalize_text_content(getattr(message, "content", ""))
        reasoning_field = _normalize_text_content(getattr(message, "reasoning_content", ""))
        reasoning_text, extracted_text = _extract_reasoning_and_answer(content_text, reasoning_field)

        usage = getattr(response, "usage", None)
        print(
            f"   Page {page_number}: {elapsed:.1f}s "
            f"(prompt={getattr(usage, 'prompt_tokens', '?')}, "
            f"completion={getattr(usage, 'completion_tokens', '?')} tokens)"
        )

        if not extracted_text:
            return _format_error(page_number, "Model returned empty response")

        return {
            "extracted_text": extracted_text,
            "reasoning_text": reasoning_text,
            "error": None,
        }

    except APITimeoutError:
        return _format_error(page_number, "Request timed out while waiting for Hugging Face")
    except RateLimitError:
        return _format_error(page_number, "Rate limit reached on Hugging Face provider")
    except APIConnectionError:
        return _format_error(page_number, "Failed to connect to Hugging Face API")
    except BadRequestError as exc:
        return _format_error(page_number, f"Bad request to Hugging Face: {str(exc)}")
    except Exception as exc:
        return _format_error(page_number, str(exc))
