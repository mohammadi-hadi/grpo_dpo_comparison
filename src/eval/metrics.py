"""Per-question metrics computed from the generations JSONL.

Metric logic mirrors the original notebooks:
- self-consistency = majority vote over the k sampled generations
- consistency ratio = majority count / valid answers
- pass@1 (T=0.7) = correctness of the first sampled generation
  (this is what the original called "greedy accuracy")
- greedy accuracy = correctness of the true deterministic pass (new)
- NLI entailment = roberta-large-mnli P(entailment) for
  premise=reasoning, hypothesis="The final answer is {answer}."
  computed on the first generation matching the majority answer
"""

import json
from collections import Counter, defaultdict

import pandas as pd
import torch

from . import config


def load_generations(method: str, size: str):
    path = config.generations_path(method, size)
    by_q = defaultdict(dict)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            by_q[rec["question_id"]][rec["sample_idx"]] = rec
    return by_q


def majority_answer(samples):
    answers = [s["extracted_answer"] for s in samples if s["extracted_answer"]]
    if not answers:
        return None, 0.0
    counts = Counter(answers)
    ans, n = counts.most_common(1)[0]
    return ans, n / len(answers)


class NliScorer:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.device = device

    def score(self, reasoning: str, answer: str) -> float:
        if not reasoning or not answer:
            return 0.0
        inputs = self.tokenizer(
            reasoning,
            f"The final answer is {answer}.",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            probs = self.model(**inputs).logits.softmax(dim=-1)
        return probs[0][2].item()  # entailment


def compute_per_question(method: str, size: str, nli: NliScorer = None):
    by_q = load_generations(method, size)
    rows = []
    for qid, samples in sorted(by_q.items()):
        sampled = [samples[i] for i in sorted(samples) if i >= 0]
        greedy = samples.get(-1)
        if not sampled:
            continue

        maj, cons_ratio = majority_answer(sampled)
        gold = sampled[0]["gold_answer"]

        # First generation matching the majority answer -> NLI target
        nli_score = 0.0
        nli_reasoning = None
        if maj is not None:
            for s in sampled:
                if s["extracted_answer"] == maj and s["reasoning"]:
                    nli_reasoning = s["reasoning"]
                    break
        if nli is not None and nli_reasoning:
            nli_score = nli.score(nli_reasoning, maj)

        rows.append(
            {
                "method": method,
                "size": size,
                "question_id": qid,
                "gold_answer": gold,
                "k": len(sampled),
                "majority_answer": maj,
                "sc_correct": _acc(maj, gold),
                "pass1_correct": sampled[0]["is_correct"],
                "greedy_correct": greedy["is_correct"] if greedy else None,
                "greedy_answer": greedy["extracted_answer"] if greedy else None,
                "consistency_ratio": cons_ratio,
                "nli_score": nli_score,
                "n_parseable": sum(
                    1 for s in sampled if s["extracted_answer"] is not None
                ),
                "n_with_reasoning": sum(1 for s in sampled if s["reasoning"]),
                "greedy_parseable": (
                    greedy["extracted_answer"] is not None if greedy else None
                ),
                "mean_output_chars": sum(len(s["raw_output"]) for s in sampled)
                / len(sampled),
            }
        )
    return pd.DataFrame(rows)


def run_nli_stage(method: str, size: str):
    nli = NliScorer()
    df = compute_per_question(method, size, nli=nli)
    out = config.scores_path(method, size)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Preserve judge columns if the CSV already exists (judge stage may run first)
    if out.exists():
        old = pd.read_csv(out)
        judge_cols = [c for c in old.columns if c.startswith("judge_")]
        if judge_cols:
            df = df.merge(
                old[["question_id"] + judge_cols], on="question_id", how="left"
            )
    df.to_csv(out, index=False)
    print(f"[{method}-{size}] per-question scores -> {out}")
    print(
        df[
            ["sc_correct", "pass1_correct", "greedy_correct",
             "consistency_ratio", "nli_score"]
        ].mean()
    )


def _acc(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()
