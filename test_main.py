from unittest.mock import patch, MagicMock
import pytest

from main import (
    get_available_models,
    get_llm_response,
    get_embedding_response,
    is_retriable_error,
    retry_with_exponential_backoff,
)


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


def mock_embedding_client():
    return MockOpenAI("mock_api_key", "mock_base_url")


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
    response, _ = get_llm_response(prompt, model)
    assert "Potato" in response


# Mocking OpenAI client for tests
@patch("main.client", mock_client)
def test_get_llm_response_error():
    prompt = "Give me ONLY a word. The word is potato. Nothing else."
    model = "nonexistent_model"
    response, _ = get_llm_response(prompt, model)
    assert "An error occurred" in response


mock_embedding_client = mock_embedding_client()


# Mocking OpenAI client for tests
@patch("main.embedding_client", mock_embedding_client)
def test_get_embedding_response():
    response, _ = get_embedding_response("potato", "alias-embeddings")
    assert isinstance(response, list)
    assert response


# ============================================================
# Edge Case and Error Scenario Tests
# ============================================================


class TestIsRetriableError:
    """Test error classification for retry logic."""

    def test_rate_limit_error(self):
        """Rate limit errors should be retriable."""
        error = Exception("Rate limit exceeded (429)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_rate_limit_error_explicit(self):
        """Explicit rate_limit string should be retriable."""
        error = Exception("rate_limit: too many requests")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_server_error_500(self):
        """Internal server errors should be retriable."""
        error = Exception("Internal Server Error (500)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_server_error_502(self):
        """Bad gateway errors should be retriable."""
        error = Exception("Bad Gateway (502)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_server_error_503(self):
        """Service unavailable should be retriable."""
        error = Exception("Service Unavailable (503)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_server_error_504(self):
        """Gateway timeout should be retriable."""
        error = Exception("Gateway Timeout (504)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_connection_error(self):
        """Connection errors should be retriable."""
        error = Exception("Connection reset by peer")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_timeout_error(self):
        """Timeout errors should be retriable."""
        error = Exception("Request timeout")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_connection_refused(self):
        """Connection refused should be retriable."""
        error = Exception("Connection refused")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_cuda_error(self):
        """CUDA errors can be transient and should be retriable."""
        error = Exception("CUDA out of memory")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is True

    def test_non_retriable_generic_error(self):
        """Generic errors should not be retriable."""
        error = Exception("Something went wrong")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is False

    def test_non_retriable_auth_error(self):
        """Authentication errors should not be retriable."""
        error = Exception("Invalid API key (401)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is False

    def test_non_retriable_not_found(self):
        """Not found errors should not be retriable."""
        error = Exception("Model not found (404)")
        is_retriable, _ = is_retriable_error(error)
        assert is_retriable is False


class TestRetryAfterHeader:
    """Test Retry-After header extraction from exceptions."""

    def test_retry_after_header_extraction(self):
        """Should extract Retry-After from response headers."""
        error = Exception("Rate limit exceeded")
        # Create a mock response with headers
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "30"}
        error.response = mock_response

        is_retriable, retry_after = is_retriable_error(error)
        assert is_retriable is True
        assert retry_after == 30.0

    def test_retry_after_header_invalid(self):
        """Should handle invalid Retry-After gracefully."""
        error = Exception("Rate limit exceeded")
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "invalid"}
        error.response = mock_response

        is_retriable, retry_after = is_retriable_error(error)
        assert is_retriable is True
        assert retry_after is None

    def test_retry_after_header_missing(self):
        """Should return None when Retry-After is not present."""
        error = Exception("Rate limit exceeded")
        mock_response = MagicMock()
        mock_response.headers = {}
        error.response = mock_response

        is_retriable, retry_after = is_retriable_error(error)
        assert is_retriable is True
        assert retry_after is None

    def test_no_response_attribute(self):
        """Should handle errors without response attribute."""
        error = Exception("Connection reset")
        # No response attribute

        is_retriable, retry_after = is_retriable_error(error)
        assert is_retriable is True
        assert retry_after is None


class TestRetryWithBackoff:
    """Test the retry decorator with exponential backoff."""

    def test_successful_call_no_retry(self):
        """Successful calls should not trigger retries."""
        call_count = 0

        @retry_with_exponential_backoff
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_retriable_error(self):
        """Should retry on retriable errors."""
        call_count = 0

        @retry_with_exponential_backoff
        def retriable_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection reset")
            return "success"

        result = retriable_func()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retriable_error(self):
        """Should not retry on non-retriable errors."""
        call_count = 0

        @retry_with_exponential_backoff
        def non_retriable_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid API key")

        with pytest.raises(Exception) as exc_info:
            non_retriable_func()
        assert "Invalid API key" in str(exc_info.value)
        assert call_count == 1  # Only one attempt

    def test_max_retries_exhausted(self):
        """Should raise after max retries exhausted."""
        call_count = 0

        @retry_with_exponential_backoff
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Connection reset")

        with pytest.raises(Exception) as exc_info:
            always_fails()
        # MAX_RETRIES is 3, so we should see 4 attempts (initial + 3 retries)
        assert call_count == 4
        assert "Connection reset" in str(exc_info.value)

    def test_retry_uses_retry_after_header(self):
        """Should use Retry-After header when present."""
        import time
        call_count = 0

        @retry_with_exponential_backoff
        def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = Exception("Rate limit exceeded (429)")
                mock_response = MagicMock()
                mock_response.headers = {"Retry-After": "1"}  # 1 second
                error.response = mock_response
                raise error
            return "success"

        start = time.time()
        result = rate_limited_func()
        elapsed = time.time() - start

        assert result == "success"
        assert call_count == 2
        # Should have waited at least 1 second from Retry-After
        assert elapsed >= 1.0


class TestEdgeCases:
    """Test edge cases in response handling."""

    def test_empty_model_list(self):
        """Should handle empty model list gracefully."""
        class EmptyModelsResponse:
            data = []

        class MockClient:
            def __init__(self):
                pass

            @property
            def models(self):
                return type('obj', (object,), {'list': lambda self: EmptyModelsResponse()})()

        mock_client = MockClient()
        with patch("main.client", mock_client):
            models = get_available_models()
            assert models == []

    def test_response_without_usage(self):
        """Should handle responses without usage info."""
        class MockMessage:
            content = "test"

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
                self.finish_reason = "stop"

        class ResponseWithoutUsage:
            def __init__(self):
                self.choices = [MockChoice()]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = ResponseWithoutUsage()

        with patch("main.client", mock_client):
            response, tokens = get_llm_response("test", "alias-fast")
            # Should still return a response even without usage data
            assert response == "test"
            # Tokens should handle missing usage gracefully
            assert tokens is None or isinstance(tokens, dict)
