"""Generation harness: k sampled generations + one greedy pass per question,
persisted incrementally as JSONL with resume support."""

import json
import time
from datetime import datetime, timezone

import torch
from datasets import load_dataset

from . import config
from .prompts import FORMATS, extract_hash_answer


def load_test_questions():
    """The same 200 questions as the original evaluation:
    GSM8K test split -> pandas -> df.sample(n=200, random_state=42)."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    df = ds.to_pandas()
    sampled = df.sample(n=config.N_QUESTIONS, random_state=config.DATASET_SEED)
    questions = []
    for qid, row in sampled.iterrows():
        questions.append(
            {
                "question_id": int(qid),  # row index in the full test split
                "question": row["question"],
                "gold_answer": extract_hash_answer(row["answer"]),
            }
        )
    return questions


def completed_keys(path):
    """(question_id, sample_idx) pairs already present in the JSONL."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["question_id"], rec["sample_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run_generation(method: str, size: str, merge: bool = False, limit: int = None):
    from .models import load_model

    fmt = FORMATS[method]
    out_path = config.generations_path(method, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = load_test_questions()
    if limit:
        questions = questions[:limit]

    sample_indices = list(range(config.NUM_SAMPLES))
    if config.GREEDY_PASS:
        sample_indices = [-1] + sample_indices  # -1 = deterministic pass

    done = completed_keys(out_path)
    todo = [
        (q, s)
        for q in questions
        for s in sample_indices
        if (q["question_id"], s) not in done
    ]
    if not todo:
        print(f"[{method}-{size}] all {len(questions)} questions already done")
        return

    print(f"[{method}-{size}] {len(todo)} generations to run "
          f"({len(done)} already present)")

    model, tokenizer, device = load_model(method, size, merge=merge)
    run_id = f"{method}_{size}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    batch_size = config.BATCH_SIZES[size]

    t0 = time.time()
    n_tokens = 0
    with open(out_path, "a") as out_f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            prompts = [fmt["format_prompt"](q["question"]) for q, _ in batch]
            greedy_mask = [s == -1 for _, s in batch]

            # Deterministic and sampled items need different generate() args,
            # so split the batch by decoding mode.
            for mode in ("greedy", "sampled"):
                idxs = [
                    i for i, g in enumerate(greedy_mask)
                    if (g and mode == "greedy") or (not g and mode == "sampled")
                ]
                if not idxs:
                    continue
                sub_prompts = [prompts[i] for i in idxs]
                inputs = tokenizer(
                    sub_prompts, return_tensors="pt", padding=True
                ).to(device)

                gen_kwargs = dict(
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
                if mode == "sampled":
                    # Deterministic per-(question, sample) seed.
                    seed = (
                        config.MASTER_SEED
                        + batch[idxs[0]][0]["question_id"] * 101
                        + batch[idxs[0]][1]
                    ) % (2**31)
                    torch.manual_seed(seed)
                    gen_kwargs.update(
                        do_sample=True,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                    )
                else:
                    seed = None
                    gen_kwargs.update(do_sample=False)

                with torch.no_grad():
                    output_ids = model.generate(**inputs, **gen_kwargs)

                for j, i in enumerate(idxs):
                    q, s = batch[i]
                    new_tokens = output_ids[j][inputs["input_ids"].shape[1]:]
                    n_tokens += len(new_tokens)
                    decoded = tokenizer.decode(output_ids[j], skip_special_tokens=True)
                    response = decoded.split("<|assistant|>")[-1].strip()
                    rec = {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "method": method,
                        "size": size,
                        "base_model": config.SIZES[size],
                        "adapter": config.ADAPTER_CHECKPOINTS[(method, size)],
                        "backend": f"transformers-{device}-bf16"
                        + ("-merged" if merge else ""),
                        "question_id": q["question_id"],
                        "question": q["question"],
                        "gold_answer": q["gold_answer"],
                        "sample_idx": s,
                        "seed": seed,
                        "gen_params": {
                            "temperature": None if s == -1 else config.TEMPERATURE,
                            "top_p": None if s == -1 else config.TOP_P,
                            "max_new_tokens": config.MAX_NEW_TOKENS,
                            "do_sample": s != -1,
                        },
                        "raw_output": response,
                        "reasoning": fmt["extract_reasoning"](response),
                        "extracted_answer": fmt["extract_answer"](response),
                    }
                    rec["is_correct"] = _check_accuracy(
                        rec["extracted_answer"], rec["gold_answer"]
                    )
                    out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

            elapsed = time.time() - t0
            done_n = start + len(batch)
            rate = n_tokens / elapsed if elapsed > 0 else 0
            print(
                f"[{method}-{size}] {done_n}/{len(todo)} generations | "
                f"{rate:.1f} tok/s | {elapsed/60:.1f} min elapsed",
                flush=True,
            )

    print(f"[{method}-{size}] DONE in {(time.time()-t0)/60:.1f} min -> {out_path}")


def _check_accuracy(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()
