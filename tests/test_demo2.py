import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rlm"))


def test_demo2_script_loads_and_exposes_main():
    demo_path = Path(__file__).resolve().parent.parent / "demos" / "demo2_oolong_pairs.py"
    assert demo_path.exists()

    spec = importlib.util.spec_from_file_location("demo2_oolong_pairs", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_demo2_loads_pairs_data():
    from dataloader import get_oolong_pairs

    data = get_oolong_pairs(32768, question_id="1")
    assert "context" in data
    assert "question" in data
    assert "answer" in data
    assert isinstance(data["answer"], list)
    assert len(data["answer"]) > 0
    # Each answer should be a pair string
    assert "(" in data["answer"][0]


def test_demo2_rlm_configuration():
    from rlm import RLM
    from rlm.logger import RLMLogger

    rlm = RLM(
        backend="moonshot",
        backend_kwargs={"model_name": "kimi-k2.6"},
        environment="local",
        max_iterations=25,
        logger=RLMLogger(log_dir="./logs"),
    )
    assert rlm.backend == "moonshot"
    assert rlm.max_iterations == 25
