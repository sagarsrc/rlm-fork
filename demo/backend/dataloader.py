from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CACHE_DIR = Path.home() / ".cache" / "rlm-demo"
_DEMO_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "demo-upload"


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _load_local_fallback(filename: str) -> dict[str, Any] | None:
    """Try to load pre-prepared data from demo-upload/ directory."""
    path = _DEMO_UPLOAD_DIR / filename
    if not path.exists():
        return None
    with path.open() as f:
        return json.loads(f.readline())


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle)


def get_oolong_trec_coarse(context_len: int = 32768) -> dict[str, Any]:
    cache_dir = _ensure_cache_dir()
    cache_path = cache_dir / f"oolong-trec-coarse-{context_len}.json"
    if cache_path.exists():
        return _read_json(cache_path)

    # Fallback: use pre-prepared local data from demo-upload/
    local = _load_local_fallback("oolong_trec_coarse_full.jsonl")
    if local is not None:
        _write_json(cache_path, local)
        return local

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets package required to download OOLONG data and no local fallback found. "
            "Install it: pip install datasets"
        )

    stream = load_dataset(
        "oolongbench/oolong-synth",
        split="validation",
        streaming=True,
        token=_token(),
        cache_dir=str(cache_dir),
    )
    for example in stream:
        if example.get("dataset") != "trec_coarse":
            continue
        if example.get("context_len") != context_len:
            continue

        result = {
            "context": example["context_window_text"],
            "question": example["question"],
            "answer": example["answer"],
        }
        _write_json(cache_path, result)
        return result

    raise ValueError(f"No trec_coarse example found for context_len={context_len}")


def get_oolong_pairs(
    context_len: int = 32768,
    question_id: str = "1",
) -> dict[str, Any]:
    cache_dir = _ensure_cache_dir()
    cache_path = cache_dir / f"oolong-pairs-{context_len}-{question_id}.json"
    if cache_path.exists():
        return _read_json(cache_path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub package required to download OOLONG pairs. "
            "Install it: pip install huggingface_hub"
        )

    answers_path = hf_hub_download(
        repo_id="mit-oasys/oolong-pairs",
        repo_type="dataset",
        filename=f"data/oolong-pairs-{context_len}.json",
        token=_token(),
        cache_dir=str(cache_dir),
    )
    with open(answers_path) as handle:
        records = json.load(handle)

    for record in records:
        if str(record.get("id")) != str(question_id):
            continue

        result = {
            "context": get_oolong_trec_coarse(context_len)["context"],
            "question": record["question"],
            "answer": list(record["answer"]),
        }
        _write_json(cache_path, result)
        return result

    raise ValueError(
        f"No OOLONG pairs example found for context_len={context_len}, question_id={question_id}"
    )
