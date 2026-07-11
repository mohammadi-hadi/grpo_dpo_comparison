# LoRA adapters

The eight fine-tuned LoRA adapters (PEFT format) are hosted on the Hugging Face Hub:

**https://huggingface.co/hadimh93/qwen2.5-gsm8k-grpo-dpo-adapters**

| Subfolder | Base model | LoRA r/α | Checkpoint |
|-----------|------------|----------|------------|
| `grpo-1.5b` | Qwen/Qwen2.5-1.5B-Instruct | 16/32 | 57 |
| `grpo-3b`   | Qwen/Qwen2.5-3B-Instruct   | 16/32 | 114 |
| `grpo-7b`   | Qwen/Qwen2.5-7B-Instruct   | 16/32 | 114 |
| `grpo-14b`  | Qwen/Qwen2.5-14B-Instruct  | 16/32 | 114 |
| `dpo-1.5b`  | Qwen/Qwen2.5-1.5B-Instruct | 32/64 | 408 |
| `dpo-3b`    | Qwen/Qwen2.5-3B-Instruct   | 32/64 | 408 |
| `dpo-7b`    | Qwen/Qwen2.5-7B-Instruct   | 32/64 | 411 |
| `dpo-14b`   | Qwen/Qwen2.5-14B-Instruct  | 32/64 | 274 |

Usage:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "hadimh93/qwen2.5-gsm8k-grpo-dpo-adapters", subfolder="grpo-14b")
```

Note: adapters were trained with QLoRA on 4-bit quantized bases (`unsloth/qwen2.5-*-instruct-unsloth-bnb-4bit`). Applying them to full-precision bases is standard practice and is what the evaluation pipeline in `src/eval/` does; minor numerical differences relative to 4-bit inference are possible.

For local evaluation, place the extracted adapter folders here (e.g. `adapters/grpo/grpo-14b-checkpoint-114/`); the directory contents are not tracked by git.
