#!/usr/bin/env python3
"""Demo 1: OOLONG trec_coarse - semantic labeling + aggregation."""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rlm"))
load_dotenv(ROOT / ".env")

from rlm import RLM
from rlm.logger import RLMLogger
from dataloader import get_oolong_trec_coarse

def main():
    print("=" * 60)
    print("DEMO 1: OOLONG trec_coarse (32K tokens)")
    print("=" * 60)
    
    # Load data
    data = get_oolong_trec_coarse(32768)
    prompt = data["context"] + "\n\n" + data["question"]
    
    print(f"Context length: {len(data['context'])} chars")
    print(f"Question: {data['question'][:100]}...")
    print(f"Ground truth: {data['answer']}")
    print()
    
    # Configure RLM
    rlm = RLM(
        backend="moonshot",
        backend_kwargs={"model_name": "kimi-k2.6"},
        environment="local",
        max_iterations=20,
        logger=RLMLogger(log_dir="./logs"),
        verbose=True,
    )
    
    # Run
    result = rlm.completion(prompt)
    
    # Compare
    print()
    print("=" * 60)
    print(f"Ground truth: {data['answer']}")
    print(f"RLM answer:  {result.response}")
    
    # Simple comparison (case-insensitive contains check)
    gt = str(data['answer']).lower().strip()
    rlm_ans = str(result.response).lower().strip()
    if gt in rlm_ans or rlm_ans in gt:
        print("[PASS] RLM answer matches ground truth")
    else:
        print("[FAIL] RLM answer differs from ground truth")
    
    print(f"Execution time: {result.execution_time:.1f}s")
    print(f"Trajectory saved to ./logs/")

if __name__ == "__main__":
    main()
