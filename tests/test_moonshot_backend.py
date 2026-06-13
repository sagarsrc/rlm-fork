import sys
sys.path.insert(0, 'rlm')
import os
os.environ['MOONSHOT_API_KEY'] = 'sk-boNGsSxJp7e3CJJJ0j2CDx1CXww5krikfRkOscRrPnWLa8SA'

from rlm.clients import get_client
from rlm.clients.openai import OpenAIClient

def test_get_client_moonshot_returns_openai_client():
    """get_client('moonshot', ...) returns OpenAIClient with correct base_url."""
    client = get_client("moonshot", {"model_name": "kimi-k2.6"})
    assert isinstance(client, OpenAIClient), f"Expected OpenAIClient, got {type(client)}"
    assert client.base_url == "https://api.moonshot.ai/v1", f"Wrong base_url: {client.base_url}"
    assert client.model_name == "kimi-k2.6"

def test_get_client_unknown_backend_raises():
    """Unknown backend raises ValueError."""
    try:
        get_client("nonexistent_backend", {})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
