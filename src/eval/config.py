"""Central configuration for the portable evaluation pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = REPO_ROOT / "adapters"
PER_EXAMPLE_DIR = REPO_ROOT / "results" / "per_example"
AGGREGATE_DIR = REPO_ROOT / "results" / "aggregate"

# Evaluation protocol — mirrors the original notebooks.
N_QUESTIONS = 200
DATASET_SEED = 42          # df.sample(n=200, random_state=42) on the GSM8K test split
NUM_SAMPLES = 5            # sampled generations per question (original used 3)
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 256
MASTER_SEED = 1234         # seeds per-sample generation deterministically

# One extra deterministic pass (do_sample=False); stored as sample_idx = -1.
GREEDY_PASS = True

JUDGE_MODEL = "gpt-5-mini"

SIZES = {
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
}

ADAPTER_CHECKPOINTS = {
    ("grpo", "1.5b"): "grpo/grpo-1.5b-checkpoint-57",
    ("grpo", "3b"): "grpo/grpo-3b-checkpoint-114",
    ("grpo", "7b"): "grpo/grpo-7b-checkpoint-114",
    ("grpo", "14b"): "grpo/grpo-14b-checkpoint-114",
    ("dpo", "1.5b"): "dpo/dpo-1.5b-checkpoint-408",
    ("dpo", "3b"): "dpo/dpo-3b-checkpoint-408",
    ("dpo", "7b"): "dpo/dpo-7b-checkpoint-411",
    ("dpo", "14b"): "dpo/dpo-14b-checkpoint-274",
}

# Question batch size during generation, per model size (rows = questions × samples).
BATCH_SIZES = {"1.5b": 16, "3b": 12, "7b": 8, "14b": 4}


def adapter_path(method: str, size: str) -> Path:
    return ADAPTERS_DIR / ADAPTER_CHECKPOINTS[(method, size)]


def run_name(method: str, size: str) -> str:
    return f"{method}_{size}"


def generations_path(method: str, size: str) -> Path:
    return PER_EXAMPLE_DIR / f"{run_name(method, size)}_generations.jsonl"


def scores_path(method: str, size: str) -> Path:
    return PER_EXAMPLE_DIR / f"{run_name(method, size)}_per_question.csv"
