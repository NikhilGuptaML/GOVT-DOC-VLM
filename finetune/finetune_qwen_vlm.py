"""
finetune_qwen_vlm.py — LoRA / QLoRA finetuning for Qwen3.5 VLM
================================================================
Single-file pipeline for government document understanding.
Supports:
  • Qwen3.5-VL-27B   (dense, bf16 LoRA)       on A100 80GB
  • Qwen3.5-VL-122B-A10B (sparse MoE, QLoRA)  on A100 80GB

Usage:
  python finetune_qwen_vlm.py --model_variant 27b --data_path data.jsonl --output_dir ./ckpt
  python finetune_qwen_vlm.py --model_variant 122b --data_path data.jsonl --output_dir ./ckpt

Existing inference pipeline (doc-qwen3.5-*/backend/) is NOT touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import editdistance
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finetune")


# ════════════════════════════════════════════════════════════
#  CONSTANTS — model registry
# ════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "27b": {
        "model_id": "Qwen/Qwen3.5-VL-27B",
        "is_moe": False,
        "load_in_4bit": False,
        # bf16 LoRA — no quantisation needed on A100 80GB
        "torch_dtype": torch.bfloat16,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
    },
    "122b": {
        "model_id": "Qwen/Qwen3.5-VL-122B-A10B",
        "is_moe": True,
        "load_in_4bit": True,
        # MoE: only ~10B active per token but full 122B lives in memory
        "torch_dtype": torch.bfloat16,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
}

# Modules that MUST NOT be quantised or LoRA-adapted in MoE models.
# Router/gate weights control expert selection — perturbing them under
# 4-bit quant causes training instability (loss spikes, expert collapse).
MOE_PROTECTED_MODULES = ["gate", "router"]

# ────────────────────────────────────────────────────────────
#  LoRA target modules — deliberate selection for early-fusion VLM
# ────────────────────────────────────────────────────────────
# Qwen3.5-VL is early-fusion: vision + language share the same
# transformer backbone.  We target:
#   ✅ All self-attention projections   → cross-modal spatial reasoning
#   ✅ Vision encoder layers            → adapt to govt doc image features
#   ✅ Top-K language layers (last 25%) → shift output vocabulary to domain
#   ❌ MLP / FFN                        → 60% of params, VRAM hog, minimal
#                                         gain for extraction tasks
#   ❌ MoE gate / router                → routing stability on 4-bit model
LORA_TARGET_MODULES = [
    # Attention projections — every layer, both vision and language
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # Vision-specific projection (Qwen3.5-VL merges vision via this)
    "visual",
]


# ════════════════════════════════════════════════════════════
#  § 1  DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════

class DocumentAugmentor:
    """
    Augmentations tailored for government document images.
    Simulates real-world scan quality variance:
      - perspective warp   → skewed/tilted scans
      - gaussian blur      → out-of-focus pages
      - brightness jitter  → over/under-exposed scans
      - contrast jitter    → faded ink, especially handwritten portions

    Applied probabilistically so the model sees both clean and degraded
    inputs during training.
    """

    def __init__(
        self,
        perspective_prob: float = 0.3,
        blur_prob: float = 0.2,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.3),
    ):
        self.perspective_prob = perspective_prob
        self.blur_prob = blur_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, image: Image.Image) -> Image.Image:
        # Perspective warp — simulates tilted scan placement
        if random.random() < self.perspective_prob:
            image = self._perspective_warp(image)

        # Gaussian blur — simulates slight defocus
        if random.random() < self.blur_prob:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Brightness jitter — over/under-exposed scans
        factor = random.uniform(*self.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast jitter — faded handwritten ink
        factor = random.uniform(*self.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(factor)

        return image

    @staticmethod
    def _perspective_warp(image: Image.Image) -> Image.Image:
        """Minor perspective transform — max 2% shift per corner."""
        w, h = image.size
        # Max pixel displacement = 2% of dimension
        dx = int(w * 0.02)
        dy = int(h * 0.02)

        # Four source corners with small random shifts
        coeffs = [
            random.randint(-dx, dx), random.randint(-dy, dy),  # top-left
            random.randint(-dx, dx), random.randint(-dy, dy),  # top-right
            random.randint(-dx, dx), random.randint(-dy, dy),  # bottom-right
            random.randint(-dx, dx), random.randint(-dy, dy),  # bottom-left
        ]

        # PIL perspective transform needs 8 coefficients mapping
        # destination → source.  We use PERSPECTIVE mode.
        return image.transform(
            image.size, Image.PERSPECTIVE,
            _find_perspective_coeffs(
                [(0, 0), (w, 0), (w, h), (0, h)],  # destination
                [  # source (shifted corners)
                    (coeffs[0], coeffs[1]),
                    (w + coeffs[2], coeffs[3]),
                    (w + coeffs[4], h + coeffs[5]),
                    (coeffs[6], h + coeffs[7]),
                ],
            ),
            Image.BICUBIC,
        )


def _find_perspective_coeffs(
    dst: list[tuple[int, int]],
    src: list[tuple[int, int]],
) -> list[float]:
    """
    Compute 8 perspective transform coefficients from 4 point pairs.
    Uses the standard 8-equation linear system for projective mapping.
    """
    import numpy as np

    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array([c for pair in src for c in pair], dtype=float)
    res = np.linalg.solve(A, B)
    return list(res.flat)


class GovtDocDataset(Dataset):
    """
    Multimodal dataset for government document finetuning.

    JSONL format per line:
    {
        "image_path": "path/to/page.png",
        "conversations": [
            {"role": "user",      "content": "<prompt>"},
            {"role": "assistant", "content": "<ground_truth_extraction>"}
        ]
    }

    The processor handles tokenization + image preprocessing internally
    (Qwen3.5-VL uses its own image processor — we don't manually normalise).
    """

    def __init__(
        self,
        data_path: str,
        processor: Any,
        max_length: int = 4096,
        augmentor: Optional[DocumentAugmentor] = None,
        is_train: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.augmentor = augmentor if is_train else None
        self.is_train = is_train

        # Load JSONL
        self.samples: List[Dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

        logger.info(
            f"Loaded {len(self.samples)} samples from {data_path} "
            f"({'train' if is_train else 'eval'} split)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        conversations = sample["conversations"]

        # ── Load & augment image ──────────────────────────
        image = Image.open(image_path).convert("RGB")
        if self.augmentor is not None:
            image = self.augmentor(image)

        # ── Build chat messages for the processor ─────────
        # Qwen3.5-VL processor expects messages with image content
        user_msg = conversations[0]["content"]
        assistant_msg = conversations[1]["content"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_msg},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_msg},
                ],
            },
        ]

        # ── Tokenize via processor (handles vision + text) ─
        # apply_chat_template converts to model's expected format
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze batch dim (collator will re-batch)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # ── Mask prompt tokens in labels ──────────────────
        # Only the assistant's response should contribute to loss.
        # We find the assistant token boundary and set everything
        # before it to -100 (ignored by CrossEntropyLoss).
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Tokenize just the user portion to find boundary
        user_text = self.processor.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        user_tokens = self.processor(
            text=[user_text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = user_tokens["input_ids"].shape[-1]

        # Mask: prompt tokens → -100, completion tokens → keep
        labels[:prompt_len] = -100
        inputs["labels"] = labels

        return inputs


@dataclass
class MultimodalDataCollator:
    """
    Collates multimodal samples into a batch.

    Handles variable-length text tokens via left-padding (causal LM
    convention) and stacks pixel_values tensors.  Attention masks are
    constructed so padded positions are ignored.

    Why left-padding for training?  Qwen3.5-VL's architecture expects
    the most recent tokens at the right edge — left-padding keeps the
    generation position consistent across batch elements.
    """

    processor: Any
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, Any] = {}

        # ── Pad input_ids and labels (left-pad for causal LM) ─
        max_len = max(f["input_ids"].shape[0] for f in features)

        padded_ids, padded_labels, attention_masks = [], [], []
        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Left-pad: [PAD PAD PAD ... actual_tokens]
            padded_ids.append(
                torch.cat([
                    torch.full((pad_len,), self.pad_token_id, dtype=f["input_ids"].dtype),
                    f["input_ids"],
                ])
            )
            padded_labels.append(
                torch.cat([
                    torch.full((pad_len,), -100, dtype=f["labels"].dtype),
                    f["labels"],
                ])
            )
            attention_masks.append(
                torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(seq_len, dtype=torch.long),
                ])
            )

        batch["input_ids"] = torch.stack(padded_ids)
        batch["labels"] = torch.stack(padded_labels)
        batch["attention_mask"] = torch.stack(attention_masks)

        # ── Stack pixel values ────────────────────────────
        # pixel_values may have varying shapes if images differ —
        # the processor should have resized them, but we handle
        # the edge case by padding to max spatial dims.
        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            # If all same shape, simple stack; otherwise pad
            shapes = [pv.shape for pv in pixel_values]
            if len(set(shapes)) == 1:
                batch["pixel_values"] = torch.stack(pixel_values)
            else:
                # Pad to max dims (C, H, W) — zero-pad spatially
                max_h = max(s[-2] for s in shapes)
                max_w = max(s[-1] for s in shapes)
                padded_pv = []
                for pv in pixel_values:
                    pad_h = max_h - pv.shape[-2]
                    pad_w = max_w - pv.shape[-1]
                    # F.pad format: (left, right, top, bottom)
                    padded_pv.append(
                        torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h))
                    )
                batch["pixel_values"] = torch.stack(padded_pv)

        # ── Pass through any remaining vision keys ────────
        # (image_grid_thw, pixel_values_videos, etc.)
        for key in features[0]:
            if key not in batch:
                try:
                    batch[key] = torch.stack([f[key] for f in features])
                except (TypeError, RuntimeError):
                    # Non-stackable (e.g. strings) — skip
                    pass

        return batch


# ════════════════════════════════════════════════════════════
#  § 2  MODEL LOADING & QUANTIZATION
# ════════════════════════════════════════════════════════════

def load_model_and_processor(
    variant: str,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
) -> tuple[Any, Any, Any]:
    """
    Load Qwen3.5-VL model + processor with appropriate precision/quantization.

    Returns: (model, processor, peft_config)

    Key decisions:
      - 27B: bf16 full precision — fits A100 80GB with LoRA
      - 122B: NF4 4-bit via bitsandbytes — MoE gate weights excluded from quant
      - LoRA rank=64 (not default 16) — vision-heavy tasks need extra capacity
        to learn spatial features like table boundaries and handwriting strokes
    """
    cfg = MODEL_REGISTRY[variant]
    model_id = cfg["model_id"]
    logger.info(f"Loading model: {model_id} (variant={variant})")

    # ── Quantization config (122B MoE only) ───────────────
    quant_config = None
    if cfg["load_in_4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # NF4 > FP4: better preserves the normal distribution of
            # pretrained weights, reducing quantisation error
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Double quantisation: quantises the quantisation constants
            # themselves — saves ~3GB on 122B model
            bnb_4bit_use_double_quant=True,
            # CRITICAL for MoE: skip quantisation on gate/router modules.
            # 4-bit quant distorts the small router MLPs that control
            # expert selection, causing expert collapse during training.
            llm_int8_skip_modules=MOE_PROTECTED_MODULES,
        )
        logger.info(
            "QLoRA: NF4 4-bit with double quant enabled. "
            f"Protected modules (full precision): {MOE_PROTECTED_MODULES}"
        )

    # ── Load processor ────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # ── Load model ────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=cfg["torch_dtype"],
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        # Low CPU memory usage: stream weights directly to GPU instead
        # of loading full model into CPU RAM first (critical for 122B)
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )

    # ── Prepare for k-bit training (QLoRA path) ──────────
    if cfg["load_in_4bit"]:
        # Prepares the quantized model for training: enables gradient
        # computation on quantized weights, casts layer norms to fp32
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            # Keep output embeddings in full precision for stable loss
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # ── Freeze MoE gate/router weights ───────────────────
    # Even though we skip quantising these, we also freeze them entirely.
    # Rationale: on a 4-bit base model, gradients flowing through the
    # router are noisy. Finetuning the router risks destabilising which
    # experts handle which tokens — catastrophic for a pretrained MoE.
    if cfg["is_moe"]:
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(m in name.lower() for m in MOE_PROTECTED_MODULES):
                param.requires_grad = False
                frozen_count += 1
        logger.info(
            f"MoE: froze {frozen_count} gate/router parameters "
            f"to prevent expert routing instability on 4-bit model"
        )

    # ── Freeze MLP / FFN modules ─────────────────────────
    # MLPs are ~60% of parameters.  For extraction tasks, attention
    # layers drive spatial reasoning over document layout — MLPs add
    # VRAM cost with minimal accuracy gain.
    mlp_frozen = 0
    for name, param in model.named_parameters():
        # Match common MLP naming patterns in Qwen architecture
        if any(p in name for p in ["mlp.", "feed_forward.", "ffn."]):
            # Skip gate modules — already handled above
            if not any(m in name.lower() for m in MOE_PROTECTED_MODULES):
                param.requires_grad = False
                mlp_frozen += 1
    logger.info(f"Froze {mlp_frozen} MLP/FFN parameters (VRAM budget + minimal gain)")

    # ── Freeze bottom 75% of language decoder layers ─────
    # General language understanding is already learned.  Only the top
    # layers need adaptation to shift output distribution toward
    # government document terminology (field names, Hindi script, etc.)
    _freeze_bottom_layers(model, keep_top_fraction=0.25)

    # ── LoRA configuration ────────────────────────────────
    # rank=64 rationale: typical NLP LoRA uses r=8-16, but vision-heavy
    # tasks need more capacity to encode spatial features like:
    #   - table cell boundaries and column alignment
    #   - handwriting stroke patterns
    #   - mixed print/handwritten region boundaries
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",  # bias training adds complexity, no measurable gain
        # For MoE: ensure LoRA is NOT applied to gate/router modules
        modules_to_save=None,
    )

    # ── Apply LoRA adapters ───────────────────────────────
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Enable gradient checkpointing ─────────────────────
    # Saves ~40% activation VRAM at ~15% speed cost.
    # Non-negotiable for 122B on A100 80GB.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return model, processor, peft_config


def _freeze_bottom_layers(model: nn.Module, keep_top_fraction: float = 0.25):
    """
    Freeze the bottom (1 - keep_top_fraction) of decoder layers.

    Only the top layers are trainable — they need to shift output
    distribution toward domain-specific vocabulary.  Bottom layers
    encode general language/vision features that transfer well.
    """
    # Find all numbered layer modules (e.g., model.layers.0, model.layers.1, ...)
    layer_names = []
    for name, _ in model.named_modules():
        # Match patterns like "model.layers.0", "model.model.layers.23"
        match = re.search(r"layers\.(\d+)$", name)
        if match:
            layer_names.append((int(match.group(1)), name))

    if not layer_names:
        logger.warning("Could not identify decoder layers — skipping layer freezing")
        return

    layer_names.sort(key=lambda x: x[0])
    total_layers = len(layer_names)
    freeze_count = int(total_layers * (1 - keep_top_fraction))

    frozen_layers = {name for _, name in layer_names[:freeze_count]}
    param_frozen = 0
    for name, param in model.named_parameters():
        for frozen_layer in frozen_layers:
            if frozen_layer in name:
                param.requires_grad = False
                param_frozen += 1
                break

    logger.info(
        f"Layer freezing: {freeze_count}/{total_layers} bottom layers frozen, "
        f"top {total_layers - freeze_count} layers trainable "
        f"({param_frozen} parameters frozen)"
    )


# ════════════════════════════════════════════════════════════
#  § 3  VRAM PROFILING CALLBACK
# ════════════════════════════════════════════════════════════

class VRAMProfilerCallback(TrainerCallback):
    """
    Logs GPU memory usage every N steps.
    Raises early warning if utilization exceeds 95% to prevent
    silent OOM crashes mid-training.
    """

    def __init__(self, log_every_n_steps: int = 25, warn_threshold: float = 0.95):
        self.log_every_n_steps = log_every_n_steps
        self.warn_threshold = warn_threshold

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return

        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        utilization = max_alloc / total

        logger.info(
            f"[VRAM step {state.global_step}] "
            f"allocated={allocated:.1f}GB  reserved={reserved:.1f}GB  "
            f"peak={max_alloc:.1f}GB  total={total:.1f}GB  "
            f"util={utilization:.1%}"
        )

        if utilization > self.warn_threshold:
            logger.warning(
                f"⚠️  VRAM utilization {utilization:.1%} > {self.warn_threshold:.0%} threshold! "
                f"Risk of OOM. Mitigations: reduce batch_size, lower image resolution, "
                f"or enable CPU offloading."
            )


class GradientNormCallback(TrainerCallback):
    """
    Monitors gradient norm for MoE training stability.

    MoE models under QLoRA can exhibit gradient spikes when expert
    routing becomes unstable.  This callback tracks a running average
    and warns when current norm exceeds 5x the average — a signal that
    the gate weights may be leaking gradients or that learning rate
    is too high for the quantized precision.
    """

    def __init__(self, spike_multiplier: float = 5.0):
        self.spike_multiplier = spike_multiplier
        self.running_norms: list[float] = []  # type: ignore

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        self.running_norms.append(total_norm)

        # Need at least 10 steps to establish baseline
        if len(self.running_norms) > 10:
            avg = sum(self.running_norms[-50:]) / len(self.running_norms[-50:])
            if total_norm > avg * self.spike_multiplier:
                logger.warning(
                    f"⚠️  Gradient spike at step {state.global_step}: "
                    f"norm={total_norm:.2f} vs running_avg={avg:.2f} "
                    f"({total_norm/avg:.1f}x). Possible MoE routing instability. "
                    f"Consider lowering learning rate."
                )


# ════════════════════════════════════════════════════════════
#  § 4  EVALUATION & METRICS
# ════════════════════════════════════════════════════════════

def compute_cer(prediction: str, reference: str) -> float:
    """
    Character Error Rate: edit_distance(pred, ref) / len(ref).

    Why CER over WER?  Government documents contain numeric amounts
    (Rs. 4,51,800), dates (15-Jan-2024), and codes (PWD/BCD/2024/0042)
    where single-character errors change meaning.  WER would treat
    "4,51,800" vs "4,51,900" as 1 word error; CER correctly flags
    the 1-char substitution.
    """
    if not reference:
        return 0.0 if not prediction else 1.0
    return editdistance.eval(prediction, reference) / len(reference)


def extract_fields(text: str) -> Dict[str, str]:
    """
    Extract key: value pairs from structured text.

    Handles two common formats in government documents:
      1. "Key: Value" lines
      2. Markdown table rows "| Key | Value |"

    Returns dict of normalised_key → value for F1 computation.
    """
    fields: Dict[str, str] = {}

    # Pattern 1: "Key: Value" or "Key — Value"
    for match in re.finditer(r"^([^|:\n]{2,50})[:—]\s*(.+)$", text, re.MULTILINE):
        key = match.group(1).strip().lower()
        val = match.group(2).strip()
        if key and val:
            fields[key] = val

    # Pattern 2: markdown table rows "| Key | Value |"
    for match in re.finditer(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", text):
        key = match.group(1).strip().lower()
        val = match.group(2).strip()
        # Skip header separators like "---|---"
        if key and val and not key.startswith("-"):
            fields[key] = val

    return fields


def compute_field_f1(prediction: str, reference: str) -> Dict[str, float]:
    """
    Field-level extraction F1.

    Computes precision/recall/F1 on the SET of (key, value) pairs
    extracted from prediction vs reference.  This catches:
      - Missing fields (recall drop)
      - Hallucinated fields (precision drop)
      - Correct field detection but wrong value (both drop)
    """
    pred_fields = extract_fields(prediction)
    ref_fields = extract_fields(reference)

    if not ref_fields:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # A predicted field is "correct" if both key exists and value matches
    correct = 0
    for key, val in pred_fields.items():
        if key in ref_fields and ref_fields[key].strip() == val.strip():
            correct += 1

    precision = correct / len(pred_fields) if pred_fields else 0.0
    recall = correct / len(ref_fields) if ref_fields else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_table_cell_accuracy(prediction: str, reference: str) -> float:
    """
    Per-cell exact match accuracy for markdown tables.

    Government docs are table-heavy.  Overall text accuracy can be 95%
    while individual table cells have 80% accuracy (errors cluster in
    numeric columns).  This metric surfaces that.
    """
    def extract_cells(text: str) -> List[str]:
        cells = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("|"):
                continue
            # Skip separator rows (|---|---|)
            if re.match(r"^\|[\s\-:|]+\|$", line):
                continue
            parts = [c.strip() for c in line.split("|")[1:-1]]
            cells.extend(parts)
        return cells

    pred_cells = extract_cells(prediction)
    ref_cells = extract_cells(reference)

    if not ref_cells:
        return 1.0 if not pred_cells else 0.0

    # Align by position (tables should have same structure)
    correct = sum(
        1 for p, r in zip(pred_cells, ref_cells) if p.strip() == r.strip()
    )
    total = max(len(ref_cells), len(pred_cells))

    return correct / total if total > 0 else 1.0


def run_evaluation(
    model: Any,
    processor: Any,
    eval_dataset: GovtDocDataset,
    max_eval_samples: int = 50,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run full evaluation suite on the eval split.

    Generates text via model.generate() (not just forward pass loss)
    and computes CER, field F1, and table cell accuracy.
    """
    model.eval()
    device = next(model.parameters()).device

    all_cer, all_f1, all_table_acc = [], [], []
    n_samples = min(len(eval_dataset), max_eval_samples)

    logger.info(f"Running evaluation on {n_samples} samples...")

    for i in tqdm(range(n_samples), desc="Evaluating"):
        sample = eval_dataset.samples[i]
        image = Image.open(sample["image_path"]).convert("RGB")
        user_msg = sample["conversations"][0]["content"]
        reference = sample["conversations"][1]["content"]

        # Build generation input (user message only, no assistant)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_msg},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=False,
            )

        # Decode only the generated tokens (strip input)
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        prediction = processor.decode(gen_ids, skip_special_tokens=True)

        # ── Compute metrics ───────────────────────
        cer = compute_cer(prediction, reference)
        field_metrics = compute_field_f1(prediction, reference)
        table_acc = compute_table_cell_accuracy(prediction, reference)

        all_cer.append(cer)
        all_f1.append(field_metrics["f1"])
        all_table_acc.append(table_acc)

    results = {
        "avg_cer": sum(all_cer) / len(all_cer),
        "char_accuracy": 1.0 - sum(all_cer) / len(all_cer),
        "avg_field_f1": sum(all_f1) / len(all_f1),
        "avg_table_cell_accuracy": sum(all_table_acc) / len(all_table_acc),
        "num_samples": n_samples,
    }

    logger.info(
        f"Eval results: "
        f"CER={results['avg_cer']:.4f}  "
        f"Char_Acc={results['char_accuracy']:.4f}  "
        f"Field_F1={results['avg_field_f1']:.4f}  "
        f"Table_Cell_Acc={results['avg_table_cell_accuracy']:.4f}"
    )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Eval results saved to {output_path}")

    model.train()
    return results


# ════════════════════════════════════════════════════════════
#  § 5  TRAINING ORCHESTRATION
# ════════════════════════════════════════════════════════════

def build_training_args(variant: str, output_dir: str, **overrides) -> TrainingArguments:
    """
    Construct TrainingArguments tuned for the specific model variant.

    VRAM-critical decisions are annotated inline.
    """
    cfg = MODEL_REGISTRY[variant]

    defaults = dict(
        output_dir=output_dir,
        # ── Batch sizing ─────────────────────────────────
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        # Effective batch = per_device * accumulation = 16 for both variants
        per_device_eval_batch_size=1,

        # ── Precision ────────────────────────────────────
        # A100 natively supports bf16 — no loss scaling needed (unlike fp16)
        bf16=True,
        fp16=False,
        # tf32 for matmuls — free 8x speedup on A100 Ampere cores
        tf32=True,

        # ── Optimizer ────────────────────────────────────
        # paged_adamw_8bit: 8-bit optimizer states via bitsandbytes
        # saves ~8GB for 122B model. "paged" variant handles memory
        # fragmentation from dynamic MoE expert activation.
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ── Schedule ─────────────────────────────────────
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        num_train_epochs=3,

        # ── Checkpointing ────────────────────────────────
        # Gradient checkpointing: ~40% VRAM saved, ~15% speed cost.
        # Non-negotiable for 122B on A100 80GB.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,

        # ── Logging ──────────────────────────────────────
        logging_steps=10,
        logging_first_step=True,
        report_to="none",  # set to "wandb" if you have wandb configured

        # ── Eval ─────────────────────────────────────────
        eval_strategy="steps",
        eval_steps=200,

        # ── Misc ─────────────────────────────────────────
        remove_unused_columns=False,  # multimodal — keys vary
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        seed=42,
    )

    # Apply any CLI overrides
    defaults.update(overrides)

    return TrainingArguments(**defaults)


def train(
    model_variant: str,
    data_path: str,
    output_dir: str,
    eval_data_path: Optional[str] = None,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    max_steps: int = -1,
    **training_overrides,
):
    """
    Main training entrypoint.

    Orchestrates: load model → prepare data → train → evaluate → save.
    """
    start_time = time.time()

    # ── Load model & processor ────────────────────────────
    model, processor, peft_config = load_model_and_processor(
        variant=model_variant,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # ── Prepare datasets ──────────────────────────────────
    augmentor = DocumentAugmentor()
    train_dataset = GovtDocDataset(
        data_path=data_path,
        processor=processor,
        augmentor=augmentor,
        is_train=True,
    )

    eval_dataset = None
    if eval_data_path:
        eval_dataset = GovtDocDataset(
            data_path=eval_data_path,
            processor=processor,
            is_train=False,
        )

    # ── Data collator ─────────────────────────────────────
    pad_token_id = (
        processor.tokenizer.pad_token_id
        if hasattr(processor, "tokenizer")
        else 0
    )
    collator = MultimodalDataCollator(processor=processor, pad_token_id=pad_token_id)

    # ── Training arguments ────────────────────────────────
    training_args = build_training_args(
        variant=model_variant,
        output_dir=output_dir,
        **({"max_steps": max_steps} if max_steps > 0 else {}),
        **training_overrides,
    )

    # ── Callbacks ─────────────────────────────────────────
    callbacks = [VRAMProfilerCallback(log_every_n_steps=25)]
    if MODEL_REGISTRY[model_variant]["is_moe"]:
        callbacks.append(GradientNormCallback(spike_multiplier=5.0))
        logger.info("MoE gradient norm monitoring enabled (spike threshold: 5x)")

    # ── Trainer ───────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Starting training: variant={model_variant}")
    logger.info(f"  LoRA rank={lora_rank}, alpha={lora_alpha}")
    logger.info(f"  Effective batch size: "
                f"{training_args.per_device_train_batch_size} × "
                f"{training_args.gradient_accumulation_steps} = "
                f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"  Precision: bf16={training_args.bf16}")
    logger.info("=" * 60)

    trainer.train()

    # ── Save ──────────────────────────────────────────────
    # Save only the LoRA adapter weights (not full model)
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    logger.info(f"LoRA adapter saved to {adapter_dir}")

    # ── Post-training evaluation ──────────────────────────
    if eval_dataset:
        eval_results = run_evaluation(
            model=model,
            processor=processor,
            eval_dataset=eval_dataset,
            output_path=os.path.join(output_dir, "eval_results.json"),
        )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed/3600:.1f}h")

    return model, processor


# ════════════════════════════════════════════════════════════
#  § 6  CLI ENTRYPOINT
# ════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA finetuning for Qwen3.5-VL on government documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 27B dense model (bf16 LoRA)
  python finetune_qwen_vlm.py --model_variant 27b --data_path train.jsonl --output_dir ./ckpt_27b

  # 122B MoE model (4-bit QLoRA)
  python finetune_qwen_vlm.py --model_variant 122b --data_path train.jsonl --output_dir ./ckpt_122b

  # Quick test run (10 steps)
  python finetune_qwen_vlm.py --model_variant 27b --data_path train.jsonl --output_dir ./test --max_steps 10
        """,
    )

    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["27b", "122b"],
        required=True,
        help="Model variant: '27b' (dense bf16 LoRA) or '122b' (MoE 4-bit QLoRA)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to evaluation data JSONL file (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetune_output",
        help="Directory for checkpoints and results",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (default: 64 — higher for vision-heavy tasks)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha scaling factor (default: 128, ratio alpha/rank=2)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 = use num_epochs instead)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    # VRAM check — warn early if not on CUDA
    if not torch.cuda.is_available():
        logger.warning(
            "⚠️  No CUDA GPU detected. Training requires an A100 80GB (or equivalent). "
            "Exiting to prevent CPU fallback (would take weeks)."
        )
        sys.exit(1)

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.0f} GB)")

    if args.model_variant == "27b" and gpu_mem_gb < 70:
        logger.warning(
            f"⚠️  27B bf16 needs ~74GB VRAM, you have {gpu_mem_gb:.0f}GB. "
            f"May OOM. Mitigations: reduce batch_size, lower image res, or use 122b QLoRA."
        )
    elif args.model_variant == "122b" and gpu_mem_gb < 45:
        logger.warning(
            f"⚠️  122B QLoRA needs ~49GB VRAM, you have {gpu_mem_gb:.0f}GB. "
            f"May OOM."
        )

    train(
        model_variant=args.model_variant,
        data_path=args.data_path,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
