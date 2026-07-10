"""Prompt formats and answer extraction, ported verbatim from the original
evaluation notebooks (Qwen2.5-Accuracy-Faithfulness-Eval-{GRPO,DPO}.ipynb).

The originals build raw pseudo-tag prompts (<|system|>/<|user|>/<|assistant|>)
rather than using the tokenizer chat template; we reproduce that exactly so
the fine-tuned adapters see the same input distribution they were evaluated
with originally.
"""

import re

# --- GRPO format ---------------------------------------------------------

GRPO_REASONING_START = "<start_working_out>"
GRPO_REASONING_END = "<end_working_out>"
GRPO_SOLUTION_START = "<SOLUTION>"
GRPO_SOLUTION_END = "</SOLUTION>"

GRPO_SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {GRPO_REASONING_START} and {GRPO_REASONING_END}.
Then, provide your solution between {GRPO_SOLUTION_START}{GRPO_SOLUTION_END}"""


def format_prompt_grpo(question: str) -> str:
    return (
        f"<|system|>\n{GRPO_SYSTEM_PROMPT}\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\nLet me work through this problem.\n{GRPO_REASONING_START}\n"
    )


def extract_reasoning_grpo(text: str) -> str:
    try:
        blocks = re.findall(
            r"<start_working_out>([\s\S]*?)<end_working_out>", text, re.IGNORECASE
        )
        return blocks[-1].strip() if blocks else ""
    except Exception:
        return ""


def extract_answer_grpo(text: str):
    try:
        solution_blocks = re.findall(
            r"<SOLUTION>([\s\S]*?)</SOLUTION>", text, re.IGNORECASE
        )
        if solution_blocks:
            content = solution_blocks[-1].strip()
            match = re.search(r"-?\d+(?:\.\d+)?", content.replace(",", ""))
            if match:
                return match.group(0)
        return None
    except Exception:
        return None


# --- DPO format ----------------------------------------------------------

DPO_SYSTEM_PROMPT = """You are a precise arithmetic assistant.
For each question, output exactly:

<reasoning>
…your step-by-step operations…
</reasoning>
<answer>
…your final numeric answer…
</answer>"""


def format_prompt_dpo(question: str) -> str:
    return (
        f"<|system|>\n{DPO_SYSTEM_PROMPT}\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\nLet's think step by step.\n<reasoning>\n"
    )


def extract_reasoning_dpo(text: str) -> str:
    try:
        blocks = re.findall(r"<reasoning>([\s\S]*?)</reasoning>", text, re.IGNORECASE)
        return blocks[-1].strip() if blocks else ""
    except Exception:
        return ""


def extract_answer_dpo(text: str):
    try:
        xml_blocks = re.findall(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if xml_blocks:
            content = xml_blocks[-1].strip()
            match = re.search(r"-?\d+(?:\.\d+)?", content.replace(",", ""))
            if match:
                return match.group(0)
        if "####" in text:
            hash_content = text.split("####")[1].strip()
            match = re.search(r"-?\d+(?:\.\d+)?", hash_content.replace(",", ""))
            if match:
                return match.group(0)
        return None
    except Exception:
        return None


# --- Shared --------------------------------------------------------------

def extract_hash_answer(text: str):
    """Extract answer after '#### X' marker in the GSM8K ground truth."""
    if not isinstance(text, str) or "####" not in text:
        return None
    parts = text.split("####")
    if len(parts) < 2:
        return None
    answer_text = parts[1].strip().replace("$", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", answer_text)
    return match.group(0) if match else None


FORMATS = {
    "grpo": {
        "format_prompt": format_prompt_grpo,
        "extract_reasoning": extract_reasoning_grpo,
        "extract_answer": extract_answer_grpo,
    },
    "dpo": {
        "format_prompt": format_prompt_dpo,
        "extract_reasoning": extract_reasoning_dpo,
        "extract_answer": extract_answer_dpo,
    },
}
