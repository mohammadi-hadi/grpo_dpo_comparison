"""LLM-as-a-Judge via the OpenAI API.

Rubric text is byte-for-byte the one used with the original Gemma 3 27B
judge (src/judge_original/gsm8k_llm_judge_batch_*.py), including the
fallback conventions: no reasoning -> score 1, unparseable reply -> score 3.
Normalization: (score - 1) / 4 onto [0, 1].
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import config

JUDGE_RUBRIC = """Evaluate this mathematical solution on a scale of 1-5:

Problem: {problem}

Student Reasoning: {reasoning}

Student Answer: {answer}

Correct Answer: {correct_answer}

Scale:
1 = Poor reasoning AND wrong answer
2 = Some logical steps but significant issues OR good reasoning with wrong answer
3 = Generally sound reasoning with minor issues, answer may be wrong due to arithmetic errors
4 = Strong reasoning with very minor flaws, correct or nearly correct answer
5 = Excellent reasoning with correct answer

Consider both the logical flow of reasoning AND the final answer accuracy.

Respond with only a number from 1 to 5."""


def judge_cache_path(method, size):
    return config.PER_EXAMPLE_DIR / f"{config.run_name(method, size)}_judge.jsonl"


def _parse_score(text: str) -> int:
    match = re.search(r"[1-5]", text.strip())
    return int(match.group(0)) if match else 3  # original fallback


def _judge_one(client, rec):
    if not rec["reasoning"] or not rec["extracted_answer"]:
        return {**_key(rec), "score": 1, "raw": None}  # original convention
    prompt = JUDGE_RUBRIC.format(
        problem=rec["question"],
        reasoning=rec["reasoning"],
        answer=rec["extracted_answer"],
        correct_answer=rec["gold_answer"],
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=config.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            return {**_key(rec), "score": _parse_score(text), "raw": text.strip()[:50]}
        except Exception as e:
            if attempt == 4:
                return {**_key(rec), "score": None, "raw": f"ERROR: {e}"[:100]}
            time.sleep(2**attempt)


def _key(rec):
    return {"question_id": rec["question_id"], "sample_idx": rec["sample_idx"]}


def run_judge_stage(method: str, size: str, workers: int = 12):
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    client = OpenAI()

    gen_path = config.generations_path(method, size)
    records = [json.loads(l) for l in open(gen_path)]

    cache_path = judge_cache_path(method, size)
    done = set()
    if cache_path.exists():
        for line in open(cache_path):
            r = json.loads(line)
            if r.get("score") is not None:
                done.add((r["question_id"], r["sample_idx"]))
    todo = [r for r in records if (r["question_id"], r["sample_idx"]) not in done]
    print(f"[{method}-{size}] judging {len(todo)} generations "
          f"({len(done)} cached) with {config.JUDGE_MODEL}")

    with open(cache_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_judge_one, client, r) for r in todo]
            for i, fut in enumerate(as_completed(futures)):
                out_f.write(json.dumps(fut.result()) + "\n")
                if (i + 1) % 100 == 0:
                    out_f.flush()
                    print(f"  {i+1}/{len(todo)}", flush=True)

    _merge_into_scores(method, size)


def _merge_into_scores(method: str, size: str):
    """Aggregate judge scores per question and merge into the scores CSV."""
    cache = [json.loads(l) for l in open(judge_cache_path(method, size))]
    df = pd.DataFrame([r for r in cache if r.get("score") is not None])
    df = df.drop_duplicates(subset=["question_id", "sample_idx"], keep="last")

    per_q = []
    for qid, grp in df.groupby("question_id"):
        sampled = grp[grp.sample_idx >= 0]
        first = sampled[sampled.sample_idx == 0]
        greedy = grp[grp.sample_idx == -1]
        per_q.append(
            {
                "question_id": qid,
                "judge_score_first": (
                    (first.score.iloc[0] - 1) / 4 if len(first) else None
                ),
                "judge_score_mean": (
                    (sampled.score.mean() - 1) / 4 if len(sampled) else None
                ),
                "judge_score_greedy": (
                    (greedy.score.iloc[0] - 1) / 4 if len(greedy) else None
                ),
            }
        )
    judge_df = pd.DataFrame(per_q)

    scores_path = config.scores_path(method, size)
    if scores_path.exists():
        base = pd.read_csv(scores_path)
        base = base.drop(columns=[c for c in base.columns if c.startswith("judge_")])
        merged = base.merge(judge_df, on="question_id", how="left")
    else:
        merged = judge_df
    merged.to_csv(scores_path, index=False)
    print(f"[{method}-{size}] judge scores merged -> {scores_path}")
