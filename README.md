# Chain-of-Thought Faithfulness Under GRPO and DPO Fine-Tuning

Code, adapters, and evaluation results for a controlled comparison of **Group Relative Policy Optimization (GRPO)** and **Direct Preference Optimization (DPO)** as fine-tuning strategies for improving the faithfulness of chain-of-thought (CoT) reasoning in large language models.

We fine-tune Qwen2.5-Instruct models at four scales (1.5B, 3B, 7B, 14B) with each method on GSM8K-derived data and evaluate them on five metrics covering accuracy, reliability, and reasoning faithfulness.

## Paper status

The paper ("Chain-of-Thought Faithfulness Under GRPO and DPO Fine-Tuning") is under review as a declared dual submission at **INLG 2026** (archival long paper) and **BlackboxNLP 2026** (EMNLP 2026 workshop, archival track), both submitted July 18, 2026; it will be published at exactly one of the two (INLG notifies Aug 15, BlackboxNLP Aug 27/Sep 8). This repository stays private during review; at camera-ready it goes public and the eight LoRA adapters are uploaded to the Hugging Face Hub. Submission packaging and timelines are tracked in the private submissions hub.

## Results

Primary results are from the bf16 re-evaluation reported in the paper: 200 GSM8K test questions (fixed seed), five samples per question at temperature 0.7 plus a true-greedy pass, NLI entailment, and an LLM judge (GPT-5-mini). Values are means over 200 paired questions; 95% bootstrap CIs and paired significance tests are in the paper. Bold marks the better method within a scale where the paired test is significant (Holm-corrected, p < .05).

| Size | Method | Greedy | pass@1 | Self-Cons. | Cons. Ratio | Entailment | Judge |
|------|--------|-------:|-------:|-----------:|------------:|-----------:|------:|
| 1.5B | DPO  | .425 | .345 | .615 | **.664** | .319 | .379 |
| 1.5B | GRPO | .470 | .350 | .555 | .528 | **.539** | .375 |
| 3B   | DPO  | .135 | .200 | .510 | .659 | .176 | .228 |
| 3B   | GRPO | **.265** | .205 | .530 | .689 | **.427** | .206 |
| 7B   | DPO  | .430 | .390 | .610 | .651 | .199 | .427 |
| 7B   | GRPO | **.750** | **.650** | **.875** | **.834** | **.507** | **.655** |
| 14B  | DPO  | .780 | .735 | .870 | .874 | .353 | .740 |
| 14B  | GRPO | .830 | .775 | **.935** | .891 | **.491** | .711 |

GRPO's reasoning entails its answer significantly more often at every scale (entailment margins +.14 to +.31, all p < 1e-7) and sweeps every metric at 7B, while DPO matches GRPO's accuracy at 14B at a fraction of the training cost.

### Original evaluation (for comparison)

The earlier evaluation (3 samples per question, 4-bit inference, Gemma 3 27B judge) is kept for comparison; per-example records for both are under `results/`.

| Method | Size (B) | Greedy Acc. | Self-Cons. | Cons. Ratio | NLI | LLM-Judge |
|--------|---------:|------------:|-----------:|------------:|------:|----------:|
| DPO    | 1.5      | 0.315       | 0.445      | 0.641       | 0.313 | 0.134     |
| DPO    | 3        | 0.165       | 0.315      | 0.588       | 0.150 | 0.074     |
| DPO    | 7        | 0.435       | 0.580      | 0.654       | 0.209 | 0.250     |
| DPO    | 14       | 0.740       | 0.885      | 0.882       | 0.314 | 0.576     |
| GRPO   | 1.5      | 0.250       | 0.310      | 0.512       | 0.476 | 0.120     |
| GRPO   | 3        | 0.055       | 0.120      | 0.197       | 0.079 | 0.235     |
| GRPO   | 7        | 0.720       | 0.810      | 0.813       | 0.470 | 0.374     |
| GRPO   | 14       | 0.755       | 0.885      | 0.902       | 0.491 | 0.748     |

## Repository structure

```
notebooks/            Original training and evaluation notebooks (CUDA / Unsloth QLoRA)
src/judge_original/   Original LLM-as-a-Judge scripts (Gemma 3 27B, CUDA)
src/eval/             Portable re-evaluation pipeline (Apple Silicon / MPS, CUDA optional)
results/aggregate/    Aggregate metrics (results.xlsx, table1.csv)
results/per_example/  Per-example generations and scores from the re-evaluation
results/training_logs/  GRPO trainer states (per-step reward/KL curves)
figures/              Plotting code and figures
adapters/             LoRA adapter download instructions (weights on Hugging Face Hub)
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fine-tuned adapters

Eight LoRA adapters (PEFT format) are available on the Hugging Face Hub — see `adapters/README.md`. Bases are Qwen2.5-Instruct models:

| Method | LoRA rank / alpha | Epochs | Checkpoints |
|--------|-------------------|--------|-------------|
| GRPO   | 16 / 32           | 0.25–0.5 (early-stopped) | 57 (1.5B), 114 (3B/7B/14B) |
| DPO    | 32 / 64           | 3      | 408 (1.5B/3B), 411 (7B), 274 (14B) |

Training was done with Unsloth QLoRA (4-bit) on CUDA; the training notebooks in `notebooks/` are kept as the record of that process and are not runnable on Apple Silicon.

## Reproducing the evaluation

The portable pipeline in `src/eval/` loads the stock bf16 base model plus a LoRA adapter and evaluates on the same 200 GSM8K test questions (seed 42):

```bash
# Generate k samples + one greedy pass per question
python -m src.eval.run_eval --method grpo --size 14b --stage generate

# Score: NLI entailment, then LLM judge (needs OPENAI_API_KEY)
python -m src.eval.run_eval --method grpo --size 14b --stage nli
python -m src.eval.run_eval --method grpo --size 14b --stage judge

# Aggregate + statistics
python -m src.eval.run_eval --stage aggregate
```

Generation parameters follow the original protocol (temperature 0.7, top-p 0.95, max 256 new tokens); per-example outputs are persisted as JSONL under `results/per_example/`.

## Data

GSM8K (Cobbe et al., 2021) is loaded via `datasets`. DPO training used the GSM8K preference dataset with chosen/rejected reasoning pairs; GRPO training used a ~15% subset of the GSM8K train split with format and correctness rewards.

## License

MIT — see `LICENSE`.
