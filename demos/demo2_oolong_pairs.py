#!/usr/bin/env python3
"""Demo 2: OOLONG-Pairs - pairwise aggregation (quadratic complexity)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rlm"))

from rlm import RLM
from rlm.logger import RLMLogger
from dataloader import get_oolong_pairs


def main():
    print("=" * 60)
    print("DEMO 2: OOLONG-Pairs (32K tokens)")
    print("=" * 60)

    data = get_oolong_pairs(32768, question_id="1")
    prompt = data["context"] + "\n\n" + data["question"]

    print(f"Context length: {len(data['context'])} chars")
    print(f"Question: {data['question'][:120]}...")
    print(f"Ground truth: {len(data['answer'])} pairs")
    print(f"Sample: {data['answer'][:3]}")
    print()

    rlm = RLM(
        backend="moonshot",
        backend_kwargs={"model_name": "kimi-k2.6"},
        environment="local",
        max_iterations=25,
        logger=RLMLogger(log_dir="./logs"),
        verbose=True,
    )

    result = rlm.completion(prompt)

    print()
    print("=" * 60)
    print(f"Ground truth: {len(data['answer'])} pairs")
    print(f"Sample: {data['answer'][:5]}")
    print()
    print(f"RLM answer ({len(str(result.response))} chars)")
    print(str(result.response)[:500])
    print(f"Execution time: {result.execution_time:.1f}s")
    print("Trajectory saved to ./logs/")


if __name__ == "__main__":
    main()
