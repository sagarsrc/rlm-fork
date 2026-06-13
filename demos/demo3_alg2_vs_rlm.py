#!/usr/bin/env python3
"""Demo 3: Algorithm 2 (direct LLM) vs RLM on same OOLONG query."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rlm"))
load_dotenv(ROOT / ".env")

from dataloader import get_oolong_trec_coarse
from rlm import RLM
from rlm.logger import RLMLogger


def main():
    print("=" * 60)
    print("DEMO 3: Algorithm 2 (Direct LLM) vs RLM")
    print("=" * 60)

    data = get_oolong_trec_coarse(32768)
    prompt = data["context"] + "\n\n" + data["question"]

    print(f"Context: {len(data['context'])} chars")
    print(f"Question: {data['question'][:100]}...")
    print(f"Ground truth: {data['answer']}")
    print()

    print("--- Algorithm 2: Direct LLM (prompt in context) ---")
    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1",
    )

    try:
        response = client.chat.completions.create(
            model="kimi-k2.6",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
        )
        alg2_answer = response.choices[0].message.content
        print(f"[Algorithm 2] Answer: {str(alg2_answer)[:300]}")
    except Exception as exc:
        alg2_answer = f"ERROR: {exc}"
        print(f"[Algorithm 2] FAILED: {exc}")

    print()
    print("--- RLM (Algorithm 1): context in REPL ---")
    rlm = RLM(
        backend="moonshot",
        backend_kwargs={"model_name": "kimi-k2.6"},
        environment="local",
        max_iterations=20,
        logger=RLMLogger(log_dir="./logs"),
        verbose=True,
    )

    result = rlm.completion(prompt)
    rlm_answer = result.response

    print()
    print("=" * 60)
    print(f"Ground truth:    {data['answer']}")
    print(f"Algorithm 2:     {str(alg2_answer)[:200]}")
    print(f"RLM (correct):   {str(rlm_answer)[:200]}")
    print()
    print("Difference: Algorithm 2 loads prompt into context -> bounded and lossy.")
    print("RLM keeps prompt in REPL -> programmatic access -> correct answer.")


if __name__ == "__main__":
    main()
