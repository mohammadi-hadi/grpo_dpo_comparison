"""CLI orchestrator for the evaluation pipeline.

Examples:
    python -m src.eval.run_eval --method grpo --size 14b --stage generate
    python -m src.eval.run_eval --method grpo --size 14b --stage nli
    python -m src.eval.run_eval --method grpo --size 14b --stage judge
    python -m src.eval.run_eval --stage aggregate
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description="GRPO/DPO reasoning-answer entailment evaluation")
    ap.add_argument("--method", choices=["grpo", "dpo"])
    ap.add_argument("--size", choices=["1.5b", "3b", "7b", "14b"])
    ap.add_argument(
        "--stage",
        required=True,
        choices=["generate", "nli", "judge", "aggregate"],
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict to first N questions (smoke tests)")
    ap.add_argument("--merge", action="store_true",
                    help="merge LoRA into the base before generation")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent judge API calls")
    args = ap.parse_args()

    if args.stage == "aggregate":
        from .stats import run_aggregate_stage
        run_aggregate_stage()
        return

    if not args.method or not args.size:
        ap.error(f"--method and --size are required for stage {args.stage}")

    if args.stage == "generate":
        from .generate import run_generation
        run_generation(args.method, args.size, merge=args.merge, limit=args.limit)
    elif args.stage == "nli":
        from .metrics import run_nli_stage
        run_nli_stage(args.method, args.size)
    elif args.stage == "judge":
        from .judge_api import run_judge_stage
        run_judge_stage(args.method, args.size, workers=args.workers)


if __name__ == "__main__":
    main()
