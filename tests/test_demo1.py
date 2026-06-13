import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rlm"))

def test_demo1_loads_data():
    """Verify demo can load OOLONG data."""
    from dataloader import get_oolong_trec_coarse
    data = get_oolong_trec_coarse(32768)
    assert "context" in data
    assert "question" in data
    assert "answer" in data
    assert len(data["context"]) > 1000
    assert len(data["question"]) > 10

def test_demo1_rlm_configuration():
    """Verify RLM can be instantiated with moonshot backend."""
    from rlm import RLM
    from rlm.logger import RLMLogger
    rlm = RLM(
        backend="moonshot",
        backend_kwargs={"model_name": "kimi-k2.6"},
        environment="local",
        max_iterations=20,
        logger=RLMLogger(log_dir="./logs"),
    )
    assert rlm.backend == "moonshot"
    assert rlm.max_iterations == 20
