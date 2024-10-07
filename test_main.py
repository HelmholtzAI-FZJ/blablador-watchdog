import os
from unittest.mock import patch
from main import get_available_models, get_llm_response

# Mocking OpenAI client
class MockOpenAI:
    class MockClient:
        class MockModels:
            def list(self):
                return MockResponse()

        class MockChat:
            class MockCompletions:
                def create(self, **kwargs):
                    return MockResponse()

        models = MockModels()
        chat = MockChat()

    def __init__(self, api_key, base_url):
        pass

class MockResponse:
    def __init__(self):
        self.data = [MockModel("alias-fast"), MockModel("alias-code")]

    def json(self):
        return self.data

class MockModel:
    def __init__(self, id):
        self.id = id

# Mocking os.getenv
def mock_os_getenv(key):
    if key == "API_KEY":
        return "mock_api_key"
    elif key == "OPENAI_BASE_URL":
        return "mock_base_url"
    else:
        return None

# Mocking OpenAI client initialization
def mock_openai_client():
    return MockOpenAI("mock_api_key", "mock_base_url")

# Mocking os.getenv for tests
@patch("os.getenv", mock_os_getenv)
# Mocking OpenAI client for tests
@patch("openai.OpenAI", mock_openai_client)
def test_get_available_models():
    models = get_available_models()
    assert "alias-fast" in models

# Mocking OpenAI client for tests
@patch("openai.OpenAI", mock_openai_client)
def test_get_llm_response():
    prompt = "Give me ONLY a word. The word is potato. Nothing else."
    model = "alias-fast"
    response = get_llm_response(prompt, model)
    assert "Potato" in response

# Mocking OpenAI client for tests
@patch("openai.OpenAI", mock_openai_client)
def test_get_llm_response_error():
    prompt = "Give me ONLY a word. The word is potato. Nothing else."
    model = "nonexistent_model"
    response = get_llm_response(prompt, model)
    assert "An error occurred" in response
