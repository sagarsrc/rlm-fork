import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rlm"))


def test_demo3_script_loads_and_exposes_main():
    demo_path = ROOT / "demos" / "demo3_alg2_vs_rlm.py"
    assert demo_path.exists()

    spec = importlib.util.spec_from_file_location("demo3_alg2_vs_rlm", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_demo3_loads_data():
    from dataloader import get_oolong_trec_coarse

    data = get_oolong_trec_coarse(32768)
    assert len(data["context"]) > 1000


def test_demo3_direct_client_works():
    """Verify direct OpenAI client can be created with Moonshot base URL."""
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
    )
    assert str(client.base_url) == "https://api.moonshot.cn/v1/"


def test_demo3_rlm_configuration():
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
