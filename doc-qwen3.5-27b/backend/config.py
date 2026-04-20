import os

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
	raw = os.getenv(name)
	if raw is None:
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Backend settings ──────────────────────────────────────
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))

# Runtime mode controls
# FORCE_MOCK=True always uses mock mode.
# USE_MOCK=True also forces mock mode for local testing.
FORCE_MOCK = _bool_env("FORCE_MOCK", False)
USE_MOCK = _bool_env("USE_MOCK", False)

# ── HF Inference settings ─────────────────────────────────
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen3.5-27B")
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "4096"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))
HF_TIMEOUT_SECONDS = float(os.getenv("HF_TIMEOUT_SECONDS", "180"))
HF_ENABLE_THINKING = _bool_env("HF_ENABLE_THINKING", True)

# ── Image conversion settings ─────────────────────────────
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "2048"))


def get_hf_token() -> str:
	return os.getenv("HF_TOKEN", "").strip()


def active_runtime_mode() -> str:
	if FORCE_MOCK or USE_MOCK:
		return "mock"
	if get_hf_token():
		return "hf"
	return "mock"


def runtime_mode_reason() -> str:
	if FORCE_MOCK:
		return "FORCE_MOCK is enabled"
	if USE_MOCK:
		return "USE_MOCK is enabled"
	if get_hf_token():
		return "HF_TOKEN found"
	return "HF_TOKEN missing; falling back to mock"
