"""Model loading for Apple Silicon (MPS) and CUDA.

The adapters were trained with QLoRA on 4-bit quantized bases
(unsloth/*-bnb-4bit). bitsandbytes is CUDA-only, so we attach each adapter
to the stock full-precision base in bfloat16 instead; adapter weights are
identical, only base precision differs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from . import config


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(method: str, size: str, merge: bool = False):
    base_id = config.SIZES[size]
    adapter_dir = config.adapter_path(method, size)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    device = pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    model.to(device)
    model.eval()

    # Tokenizer from the adapter checkpoint (carries the training-time config);
    # left padding so batched generation aligns completions at the end.
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device
