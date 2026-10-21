from unittest.mock import patch, MagicMock
import asyncio
import base64
import threading
import time
import pytest

from main import (
    detect_model_type,
    get_video_response,
    get_rerank_response,
    get_available_models,
    get_llm_response,
    get_embedding_response,
    check_model,
    model_timeout,
    is_retriable_error,
    retry_with_exponential_backoff,
    to_daemon_thread,
    load_model_state,
    save_model_state,
    model_tags,
    model_list_change_report,
    shorten_error,
    MAX_ERROR_REASON_CHARS,
    DEFAULT_MODEL_TIMEOUT,
    SLOW_REASONING_MODEL_TIMEOUT,
)


# Mocking OpenAI client
class MockOpenAI:
    class MockModels:
        def list(self, **kwargs):
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
    def __iter__(self):
        yield ChatChunk(content="Pot")
        yield ChatChunk(content="ato", finish_reason="stop")
        yield ChatChunk(total_tokens=12)


class ChatChunk:
    def __init__(self, content=None, reasoning=None, finish_reason=None, total_tokens=None):
        self.choices = (
            [MockChoice(content, reasoning, finish_reason)]
            if content is not None or reasoning is not None or finish_reason is not None
            else []
        )
        self.usage = MagicMock(total_tokens=total_tokens) if total_tokens is not None else None


class MockChoice:
    def __init__(self, content=None, reasoning=None, finish_reason=None):
        self.delta = MagicMock(
            content=content,
            reasoning_content=reasoning,
            reasoning=None,
        )
        self.finish_reason = finish_reason


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
    response, tokens = get_llm_response(prompt, model)
    assert "Potato" in response
    assert tokens == 12


@patch("main.client", mock_client)
def test_get_llm_response_requests_streaming():
    with patch.object(
        mock_client.chat.completions, "create", wraps=mock_client.chat.completions.create
    ) as create:
        get_llm_response("test", "alias-fast")

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}


@patch("main.client", mock_client)
def test_kimi_k3_uses_low_thinking_effort():
    with patch.object(
        mock_client.chat.completions, "create", wraps=mock_client.chat.completions.create
    ) as create:
        get_llm_response("test", "alias-kimi-k3-1m")

    assert create.call_args.kwargs["extra_body"] == {
        "chat_template_kwargs": {"thinking_effort": "low"}
    }


def test_detects_model_capabilities_from_names_and_metadata():
    endpoint = {
        "model_metadata": {
            "diffusion-model": {
                "task_type": "TI2V",
                "pipeline_class": "ExampleVideoPipeline",
            },
            "picture-model": {"pipeline_name": "ImageGenerationPipeline"},
        }
    }
    with patch("main.ENDPOINTS", [endpoint]):
        assert detect_model_type("alias-fast", 0) == "chat"
        assert detect_model_type("alias-embeddings", 0) == "embedding"
        assert detect_model_type("faster-whisper-large-v3", 0) == "audio"
        assert detect_model_type("MiniMax-h3", 0) == "video"
        assert detect_model_type("diffusion-model", 0) == "video"
        assert detect_model_type("picture-model", 0) == "image"


def test_kimi_k3_gets_a_longer_timeout():
    assert model_timeout("alias-kimi-k3-1m") == SLOW_REASONING_MODEL_TIMEOUT
    assert model_timeout("10 - Qwen 3.8-27B with DFlash") == SLOW_REASONING_MODEL_TIMEOUT
    assert model_timeout("15 - DeepSeek-V4.1-Flash on JUWELS Booster") == SLOW_REASONING_MODEL_TIMEOUT
    assert model_timeout("alias-deepseek-v41-flash-booster") == SLOW_REASONING_MODEL_TIMEOUT
    assert model_timeout("alias-fast") == DEFAULT_MODEL_TIMEOUT
    assert SLOW_REASONING_MODEL_TIMEOUT > DEFAULT_MODEL_TIMEOUT


def test_video_probe_falls_back_from_catalog_gateway():
    endpoints = [
        {
            "name": "Endpoint-1",
            "base_url": "https://catalog.example/v1",
            "api_key": "one",
            "models": ["MiniMax-h3"],
        },
        {
            "name": "Endpoint-2",
            "base_url": "https://inference.example/v1",
            "api_key": "two",
            "models": ["MiniMax-h3"],
        },
    ]
    not_found = MagicMock(status_code=404, is_success=False, text="Not Found")
    validation = MagicMock(
        status_code=400,
        is_success=False,
        text="prompt field required",
    )
    with patch("main.ENDPOINTS", endpoints), patch(
        "main.httpx.post", side_effect=[not_found, not_found, validation]
    ) as post:
        response, _ = get_video_response("MiniMax-h3", 0)

    assert response == "video endpoint healthy via Endpoint-2"
    # /videos and /videos/sync both 404 on Endpoint-1 before falling through.
    assert post.call_count == 3


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
                return type('obj', (object,), {'list': lambda self, **kwargs: EmptyModelsResponse()})()

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


class TestModelListState:
    """The model list of the previous run, used to report vanished models."""

    def test_round_trip_records_sorted_tags(self, tmp_path):
        state_path = str(tmp_path / "last_models.json")
        save_model_state([("alias-fast", 0), ("alias-code", 0)], path=state_path)

        assert load_model_state(path=state_path) == ["0:alias-code", "0:alias-fast"]

    def test_no_state_file_returns_empty_list(self, tmp_path):
        assert load_model_state(path=str(tmp_path / "missing.json")) == []

    def test_corrupt_state_file_returns_empty_list(self, tmp_path):
        state_path = tmp_path / "last_models.json"
        state_path.write_text("{not json")
        assert load_model_state(path=str(state_path)) == []

    def test_missing_models_are_reported(self):
        with patch("main.ENDPOINTS", [{"name": "Endpoint-1"}]):
            report = model_list_change_report(
                previous=["0:alias-fast", "0:alias-gone"],
                current=["0:alias-fast"],
            )

        assert report[0] == "Models missing since last run (1):"
        assert report[1] == "  [Endpoint-1] alias-gone"
        assert report[2] == "Models new since last run (0):"
        assert report[3] == "  - none"

    def test_new_models_are_reported(self):
        with patch("main.ENDPOINTS", [{"name": "Endpoint-1"}]):
            report = model_list_change_report(previous=[], current=["0:alias-new"])

        assert report == ["Model list: no previous run recorded, saved 1 models"]

    def test_empty_catalog_keeps_previous_list(self):
        report = model_list_change_report(previous=["0:alias-fast"], current=[])

        assert report == [
            "Model list: no models discovered this run (kept the previous list of 1)"
        ]

    def test_identical_lists_report_nothing_missing(self):
        with patch("main.ENDPOINTS", [{"name": "Endpoint-1"}]):
            report = model_list_change_report(previous=["0:a"], current=["0:a"])

        assert report == [
            "Models missing since last run (0):",
            "  - none",
            "Models new since last run (0):",
            "  - none",
        ]


def rerank_endpoint():
    """Endpoint dict shaped like a real /v1 gateway entry."""
    return {
        "name": "Endpoint-1",
        "base_url": "https://gateway.example/v1",
        "api_key": "secret",
        "model_metadata": {},
    }


class TestRerankingModels:
    """Reranking models answer on /rerank only: chat probes 404 and fail."""

    def test_reranking_models_are_detected(self):
        with patch("main.ENDPOINTS", [rerank_endpoint()]):
            assert detect_model_type("alias-qwen3-8b-reranking", 0) == "rerank"
            assert detect_model_type("alias-qwen3-4b-reranking", 0) == "rerank"
            assert detect_model_type("Qwen3-Reranker-8B", 0) == "rerank"
            assert detect_model_type("alias-qwen3-8b-embeddings", 0) == "embedding"
            assert detect_model_type("alias-fast", 0) == "chat"

    def test_rerank_probe_posts_query_and_documents(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.93},
                {"index": 1, "relevance_score": 0.00003},
            ]
        }
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", return_value=response
        ) as post:
            message, tokens = get_rerank_response("alias-qwen3-8b-reranking", 0)

        assert message == "rerank scores: relevant=0.9300 irrelevant=0.0000"
        assert tokens is None
        assert post.call_args.args[0] == "https://gateway.example/v1/rerank"
        body = post.call_args.kwargs["json"]
        assert body["model"] == "alias-qwen3-8b-reranking"
        assert body["query"] == "potato"
        assert len(body["documents"]) == 2
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_rerank_probe_rejects_wrong_ranking(self):
        response = MagicMock(status_code=200, is_success=True)
        response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.01},
                {"index": 1, "relevance_score": 0.99},
            ]
        }
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", return_value=response
        ):
            message, _ = get_rerank_response("alias-qwen3-8b-reranking", 0)

        assert message.startswith("Wrong ranking:")

    def test_rerank_probe_reports_route_errors(self):
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", side_effect=Exception("Connection refused")
        ):
            message, _ = get_rerank_response("alias-qwen3-8b-reranking", 0)

        assert message.startswith("An error occurred:")

    def test_check_model_routes_reranking_models_to_rerank_probe(self):
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.get_llm_response"
        ) as chat, patch(
            "main.get_rerank_response",
            return_value=("rerank scores: relevant=0.9 irrelevant=0.0", None),
        ) as rerank:
            ok, response, _ = check_model(
                "alias-qwen3-8b-reranking", "potato", "Give me ONLY a word.", 0
            )

        assert ok is True
        assert response.startswith("rerank scores:")
        assert rerank.call_args.args == ("alias-qwen3-8b-reranking", 0)
        assert not chat.called

    def test_check_model_fails_reranking_model_with_wrong_ranking(self):
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.get_rerank_response",
            return_value=("Wrong ranking: rerank scores: relevant=0.0 irrelevant=0.9", None),
        ):
            ok, response, _ = check_model(
                "alias-qwen3-4b-reranking", "potato", "Give me ONLY a word.", 0
            )

        assert ok is False
        assert response.startswith("Wrong ranking:")


class TruncatedResponse:
    """Thinking model that spent the token budget before writing the word."""

    def __iter__(self):
        yield ChatChunk(content="pot", finish_reason="length")
        yield ChatChunk(total_tokens=120)


class StillTruncatedResponse:
    def __iter__(self):
        yield ChatChunk(content="pot", finish_reason="length")
        yield ChatChunk(total_tokens=600)


class TestTruncatedThinkingOutput:
    """A 30-token budget spent on reasoning truncates "potato" to "pot"."""

    def make_client(self, *responses):
        client = mock_openai_client()
        client.chat.completions.create = MagicMock(side_effect=list(responses))
        return client

    def test_truncated_answer_is_retried_with_more_tokens(self):
        client = self.make_client(TruncatedResponse(), ChatResponse())
        with patch("main.client", client):
            response, _ = get_llm_response("Give me ONLY a word.", "alias-fast")

        assert "Potato" in response
        attempts = [
            call.kwargs["max_tokens"]
            for call in client.chat.completions.create.call_args_list
        ]
        assert attempts == [30, 512]

    def test_truncated_answer_is_reported_as_truncation_not_wrong_output(self):
        client = self.make_client(TruncatedResponse(), StillTruncatedResponse())
        with patch("main.client", client), patch("main.ENDPOINTS", [rerank_endpoint()]):
            ok, response, _ = check_model(
                "alias-fast", "potato", "Give me ONLY a word.", 0
            )

        assert ok is False
        assert response.startswith("An error occurred: Truncated output")

    def test_complete_answer_is_not_retried(self):
        client = self.make_client(ChatResponse())
        with patch("main.client", client):
            response, _ = get_llm_response("Give me ONLY a word.", "alias-fast")

        assert "Potato" in response
        assert client.chat.completions.create.call_count == 1


class Base64ImageResponse:
    """A model answering the chat probe with a generated image data URL."""

    def __init__(self, payload_chars: int = 700_000):
        self.payload = (
            "![generated image](data:image/jpeg;base64," + "A" * payload_chars + ")"
        )

    def __iter__(self):
        yield ChatChunk(content=self.payload, finish_reason="stop")
        yield ChatChunk(total_tokens=1024)


class WrongWordResponse:
    def __iter__(self):
        yield ChatChunk(content="rabbit", finish_reason="stop")
        yield ChatChunk(total_tokens=7)


class TestImageGeneratorModels:
    """Image generators render on /images/generations: a chat probe is not a test."""

    def test_flux_models_are_detected_as_image(self):
        with patch("main.ENDPOINTS", [rerank_endpoint()]):
            assert detect_model_type("alias-flux2", 0) == "image"
            assert (
                detect_model_type(
                    "999 - FLUX.2-dev - An image generator (Black Forest Labs)", 0
                )
                == "image"
            )
            assert detect_model_type("999 - Qwen-Image-2512 - An image generator", 0) == "image"
            assert detect_model_type("alias-image", 0) == "image"
            assert detect_model_type("alias-fast", 0) == "chat"

    def test_flux_alias_renders_a_real_image_on_the_image_route(self):
        rendered = MagicMock(status_code=200, is_success=True)
        rendered.json.return_value = {
            "data": [{"b64_json": base64.b64encode(b"\x89PNG" + b"0" * 60).decode()}]
        }
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", return_value=rendered
        ) as post:
            ok, response, _ = check_model(
                "alias-flux2", "potato", "Give me ONLY a word.", 0
            )

        assert ok is True
        assert response == "image rendered (64 bytes)"
        assert post.call_args.args[0].endswith("/images/generations")
        payload = post.call_args.kwargs["json"]
        assert payload["prompt"] == "a potato"
        assert payload["size"] == "50x50"
        assert payload["response_format"] == "b64_json"

    def test_image_route_answering_without_image_bytes_fails(self):
        not_an_image = MagicMock(status_code=200, is_success=True)
        not_an_image.json.return_value = {
            "data": [{"b64_json": base64.b64encode(b"<html>oops</html>").decode()}]
        }
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", return_value=not_an_image
        ):
            ok, response, _ = check_model("alias-image", "potato", "prompt", 0)

        assert ok is False
        assert response == "An error occurred: Image response is not an image"


class TestVoiceModels:
    """alias-voice-drama3 is a TTS model: the chat probe cannot exercise it,
    /audio/speech with a one-word input renders a ~74 KB WAV."""

    def test_voice_models_are_detected_as_speech(self):
        with patch("main.ENDPOINTS", [rerank_endpoint()]):
            assert detect_model_type("alias-voice-drama3", 0) == "speech"
            assert detect_model_type("faster-whisper-large-v3", 0) == "audio"

    def test_voice_model_synthesises_on_the_speech_route(self):
        wav = MagicMock(status_code=200, is_success=True, content=b"RIFF" + b"0" * 96)
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", return_value=wav
        ) as post:
            ok, response, _ = check_model("alias-voice-drama3", "potato", "prompt", 0)

        assert ok is True
        assert response == "speech synthesised (100 bytes)"
        assert post.call_args.args[0] == "https://gateway.example/v1/audio/speech"
        assert post.call_args.kwargs["json"] == {
            "model": "alias-voice-drama3",
            "input": "potato",
        }

    def test_speech_route_returning_json_is_not_audio(self):
        json_body = MagicMock(status_code=200, is_success=True, content=b'{"error":1}')
        with patch("main.ENDPOINTS", [rerank_endpoint()]), patch(
            "main.httpx.post", return_value=json_body
        ):
            ok, response, _ = check_model("alias-voice-drama3", "potato", "prompt", 0)

        assert ok is False
        assert response == "An error occurred: Speech response is not audio"


def decision_endpoint():
    endpoint = rerank_endpoint()
    endpoint["model_metadata"] = {
        "alias-clef": {
            "capabilities": ["systemone"],
            "decision_endpoint": "/v1/systemone",
        }
    }
    return endpoint


class TestDecisionModels:
    """alias-clef 404s on chat; the catalog advertises decision_endpoint
    /v1/systemone, which takes typed questions ("choice" needs a criteria
    object) instead of a prompt. A 200 is not enough: the probe must pick
    potato over quark."""

    def test_clef_is_detected_from_its_advertised_route(self):
        with patch("main.ENDPOINTS", [decision_endpoint()]):
            assert detect_model_type("alias-clef", 0) == "decision"

    @staticmethod
    def decided(choice):
        answered = MagicMock(status_code=200, is_success=True)
        answered.json.return_value = {
            "model": "alias-clef",
            "answers": {"word": {"type": "choice", "choice": choice, "confidence": 0.9925}},
        }
        return answered

    def test_clef_is_asked_a_choice_question_on_its_decision_route(self):
        with patch("main.ENDPOINTS", [decision_endpoint()]), patch(
            "main.httpx.post", return_value=self.decided("potato")
        ) as post:
            ok, response, _ = check_model("alias-clef", "potato", "prompt", 0)

        assert ok is True
        assert response == "decision: potato (confidence 0.9925)"
        assert post.call_args.args[0] == "https://gateway.example/v1/systemone"
        payload = post.call_args.kwargs["json"]
        assert payload["state"] == {"text": "The word is potato."}
        question = payload["questions"]["word"]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"potato", "quark"}

    def test_wrong_choice_fails_even_with_a_200(self):
        with patch("main.ENDPOINTS", [decision_endpoint()]), patch(
            "main.httpx.post", return_value=self.decided("quark")
        ):
            ok, response, _ = check_model("alias-clef", "potato", "prompt", 0)

        assert ok is False
        assert response == "Wrong decision: decision: quark (confidence 0.9925)"

    def test_unserved_decision_route_is_reported_by_name(self):
        missing = MagicMock(status_code=404, is_success=False, text="Not Found")
        with patch("main.ENDPOINTS", [decision_endpoint()]), patch(
            "main.httpx.post", return_value=missing
        ):
            ok, response, _ = check_model("alias-clef", "potato", "prompt", 0)

        assert ok is False
        assert response == (
            "An error occurred: Advertised route /v1/systemone is not served (404)"
        )


class TestOversizedFailureReasons:
    """A data URL answer used to dump 868 KB into the report and the metrics DB."""

    def make_client(self, *responses):
        client = mock_openai_client()
        client.chat.completions.create = MagicMock(side_effect=list(responses))
        return client

    def test_base64_image_answer_is_capped_before_reporting(self):
        stream = Base64ImageResponse()
        client = self.make_client(stream)
        with patch("main.client", client), patch("main.ENDPOINTS", [rerank_endpoint()]):
            ok, response, _ = check_model(
                "alias-fast", "potato", "Give me ONLY a word.", 0
            )

        assert ok is False
        assert response.startswith(
            "Wrong output: ![generated image](data:image/jpeg;base64,"
        )
        assert len(response) <= MAX_ERROR_REASON_CHARS + 40
        assert "+699" in response and "chars dropped" in response
        assert "A" * (MAX_ERROR_REASON_CHARS + 50) not in response

    def test_short_failure_reason_is_kept_verbatim(self):
        client = self.make_client(WrongWordResponse())
        with patch("main.client", client), patch("main.ENDPOINTS", [rerank_endpoint()]):
            ok, response, _ = check_model(
                "alias-fast", "potato", "Give me ONLY a word.", 0
            )

        assert ok is False
        assert response == "Wrong output: rabbit"

    def test_shorten_error_is_a_no_op_below_the_limit(self):
        assert shorten_error("Wrong output: rabbit") == "Wrong output: rabbit"
        long_reason = "x" * (MAX_ERROR_REASON_CHARS + 10)
        assert shorten_error(long_reason).endswith("[+10 chars dropped]")


class TestStalledProbesDoNotHangShutdown:
    """Probes that outlive their asyncio budget must not stall process exit.

    2026-10 incident: three timed-out chat probes stayed parked in their
    unbounded OpenAI SDK call on asyncio.to_thread's non-daemon executor
    threads. asyncio.run() then joined those threads for 300 seconds ("The
    executor did not finishing joining its threads within 300 seconds") and
    ./run.sh froze right after printing "Failures (3):".
    """

    def test_probe_work_runs_on_daemon_threads(self):
        seen = {}

        async def scenario():
            def probe():
                seen["daemon"] = threading.current_thread().daemon
                return "ok"
            return await to_daemon_thread(probe)

        assert asyncio.run(scenario()) == "ok"
        assert seen["daemon"] is True

    def test_chat_probe_sets_sdk_timeout_from_the_model_budget(self):
        client = mock_openai_client()
        client.chat.completions.create = MagicMock(side_effect=[ChatResponse(), ChatResponse()])
        with patch("main.client", client):
            get_llm_response("test message", "alias-fast")
            get_llm_response("test message", "alias-kimi-k3-1m")
        first, second = client.chat.completions.create.call_args_list
        assert first.kwargs["timeout"] == DEFAULT_MODEL_TIMEOUT
        assert second.kwargs["timeout"] == SLOW_REASONING_MODEL_TIMEOUT

    def test_model_listing_sets_sdk_timeout(self):
        client = mock_openai_client()
        client.models.list = MagicMock(return_value=ModelsResponse())
        with patch("main.client", client):
            get_available_models()
        assert client.models.list.call_args.kwargs["timeout"] == DEFAULT_MODEL_TIMEOUT

    def test_embedding_probe_sets_sdk_timeout(self):
        embedding_client = MockOpenAI("mock_api_key", "mock_base_url")
        embedding_client.embeddings.create = MagicMock(return_value=EmbeddingResponse())
        with patch("main.embedding_client", embedding_client):
            get_embedding_response("potato", "alias-embeddings")
        assert (
            embedding_client.embeddings.create.call_args.kwargs["timeout"]
            == DEFAULT_MODEL_TIMEOUT
        )

    def test_stalled_probe_does_not_stall_loop_shutdown(self):
        release = threading.Event()

        def stuck_probe():
            release.wait(30)
            return "too late"

        async def scenario():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(to_daemon_thread(stuck_probe), timeout=0.05)

        started = time.monotonic()
        asyncio.run(scenario())
        runtime = time.monotonic() - started
        release.set()  # let the abandoned daemon thread die quietly
        assert runtime < 5


class TestMetadataLessSpecialModels:
    """The gateway (2026-10-21) serves alias-clef and alias-video with no
    metadata at all, so both fell through to the potato chat prompt: clef
    404s there, and video answers "Use POST /v1/videos/sync for alias-video".
    /v1/videos 404s for alias-video; the sync route takes multipart form data
    and a model-only form gets the backend's "prompt is required" — routing
    proven without rendering a video."""

    endpoints = [{"name": "Endpoint-1", "base_url": "https://gw.example/v1", "api_key": "k"}]

    def test_detected_by_name_without_metadata(self):
        with patch("main.ENDPOINTS", self.endpoints):
            assert detect_model_type("alias-clef", 0) == "decision"
            assert detect_model_type("alias-video", 0) == "video"

    def test_video_output_modality_is_a_video_model(self):
        endpoints = [{**self.endpoints[0], "model_metadata": {
            "new-clip": {"modalities": {"output": ["video"]}}}}]
        with patch("main.ENDPOINTS", endpoints):
            assert detect_model_type("new-clip", 0) == "video"

    def test_alias_video_is_validated_on_the_sync_route(self):
        not_found = MagicMock(status_code=404, is_success=False, text='{"detail":"Not Found"}')
        validation = MagicMock(
            status_code=400, is_success=False, text='{"detail":"prompt is required"}'
        )
        with patch("main.ENDPOINTS", self.endpoints), patch(
            "main.httpx.post", side_effect=[not_found, validation]
        ) as post:
            ok, response, _ = check_model("alias-video", "potato", "prompt", 0)

        assert ok, response
        sync = post.call_args_list[1]
        assert sync.args[0] == "https://gw.example/v1/videos/sync"
        assert sync.kwargs["files"] == {"model": (None, "alias-video")}
        assert "prompt" not in sync.kwargs["files"]  # never start a render
