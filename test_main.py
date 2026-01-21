from unittest.mock import patch
from main import get_available_models, get_llm_response, get_embedding_response

# Mocking OpenAI client
class MockOpenAI:
    class MockModels:
        def list(self):
            return ModelsResponse()

    class MockChat:
        class MockCompletions:
            def create(self, **kwargs):
                if kwargs.get("model") == "nonexistent_model":
                    raise Exception("Model not found")
                return ChatResponse()

        completions = MockCompletions()

    class MockEmbeddings:
        def create(self, **kwargs):
            return EmbeddingResponse()

    def __init__(self, api_key, base_url):
        self.models = self.MockModels()
        self.chat = self.MockChat()
        self.embeddings = self.MockEmbeddings()

class ModelsResponse:
    def __init__(self):
        self.data = [
            MockModel("alias-fast"),
            MockModel("alias-code"),
            MockModel("alias-embeddings"),
        ]


class ChatResponse:
    def __init__(self):
        self.choices = [MockChoice()]


class MockChoice:
    def __init__(self):
        self.message = MockMessage()


class MockMessage:
    def __init__(self):
        self.content = "Potato"


class EmbeddingResponse:
    def __init__(self):
        self.data = [MockEmbedding()]


class MockEmbedding:
    def __init__(self):
        self.embedding = [0.1, 0.2, 0.3]


class MockModel:
    def __init__(self, id):
        self.id = id

def mock_openai_client():
    return MockOpenAI("mock_api_key", "mock_base_url")

mock_client = mock_openai_client()

# Mocking OpenAI client for tests
@patch("main.client", mock_client)
def test_get_available_models():
    models = get_available_models()
    assert "alias-fast" in models
    assert "alias-embeddings" in models

# Mocking OpenAI client for tests
@patch("main.client", mock_client)
def test_get_llm_response():
    prompt = "Give me ONLY a word. The word is potato. Nothing else."
    model = "alias-fast"
    response = get_llm_response(prompt, model)
    assert "Potato" in response

# Mocking OpenAI client for tests
@patch("main.client", mock_client)
def test_get_llm_response_error():
    prompt = "Give me ONLY a word. The word is potato. Nothing else."
    model = "nonexistent_model"
    response = get_llm_response(prompt, model)
    assert "An error occurred" in response


# Mocking OpenAI client for tests
@patch("main.client", mock_client)
def test_get_embedding_response():
    response = get_embedding_response("potato", "alias-embeddings")
    assert isinstance(response, list)
    assert response
