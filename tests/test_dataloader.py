import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeStream:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _load_trec_result(monkeypatch, tmp_path):
    dataloader = importlib.import_module("dataloader")
    monkeypatch.setattr(dataloader, "_CACHE_DIR", Path(tmp_path))

    calls = []
    rows = [
        {
            "dataset": "other",
            "context_len": 32768,
            "context_window_text": "ignore",
            "question": "ignore",
            "answer": "ignore",
        },
        {
            "dataset": "trec_coarse",
            "context_len": 32768,
            "context_window_text": "context text",
            "question": "What label is most common?",
            "answer": "['abbreviation']",
        },
    ]

    def fake_load_dataset(repo_id, split, streaming, token=None, cache_dir=None):
        calls.append(
            {
                "repo_id": repo_id,
                "split": split,
                "streaming": streaming,
                "token": token,
                "cache_dir": cache_dir,
            }
        )
        return FakeStream(rows)

    monkeypatch.setattr(dataloader, "load_dataset", fake_load_dataset)

    first = dataloader.get_oolong_trec_coarse()
    second = dataloader.get_oolong_trec_coarse()
    return dataloader, first, second, calls


def test_get_oolong_trec_coarse_returns_dict_with_keys(monkeypatch, tmp_path):
    _, first, second, calls = _load_trec_result(monkeypatch, tmp_path)

    assert set(first.keys()) == {"context", "question", "answer"}
    assert first == second
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "oolongbench/oolong-synth"
    assert calls[0]["split"] == "validation"
    assert calls[0]["streaming"] is True


def test_context_is_string_and_non_empty(monkeypatch, tmp_path):
    _, result, _, _ = _load_trec_result(monkeypatch, tmp_path)

    assert isinstance(result["context"], str)
    assert result["context"].strip()


def test_question_is_non_empty_string(monkeypatch, tmp_path):
    _, result, _, _ = _load_trec_result(monkeypatch, tmp_path)

    assert isinstance(result["question"], str)
    assert result["question"].strip()


def test_get_oolong_pairs_returns_valid_answer_list(monkeypatch, tmp_path):
    dataloader = importlib.import_module("dataloader")
    monkeypatch.setattr(dataloader, "_CACHE_DIR", Path(tmp_path))
    monkeypatch.setattr(
        dataloader,
        "get_oolong_trec_coarse",
        lambda context_len=32768: {
            "context": "shared context",
            "question": "unused",
            "answer": "unused",
        },
    )

    answers_path = Path(tmp_path) / "oolong-pairs-32768.json"
    answers_path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "Find valid pairs.",
                    "answer": ["(22740, 35839)", "(35839, 52032)"],
                    "type": "list_of_answers",
                }
            ]
        )
    )

    calls = []

    def fake_hf_hub_download(repo_id, filename, repo_type=None, token=None, cache_dir=None):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "token": token,
                "cache_dir": cache_dir,
            }
        )
        return str(answers_path)

    monkeypatch.setattr(dataloader, "hf_hub_download", fake_hf_hub_download)

    result = dataloader.get_oolong_pairs(question_id="1")

    assert isinstance(result["answer"], list)
    assert result["answer"]
    assert all(isinstance(item, str) for item in result["answer"])
    assert result["context"] == "shared context"
    assert result["question"] == "Find valid pairs."
    assert calls == [
        {
            "repo_id": "mit-oasys/oolong-pairs",
            "filename": "data/oolong-pairs-32768.json",
            "repo_type": "dataset",
            "token": None,
            "cache_dir": str(Path(tmp_path)),
        }
    ]
