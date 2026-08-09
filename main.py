import asyncio
import os
import time
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import httpx
from openai import OpenAI
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static

from embedding_models import EMBEDDING_MODELS
from metrics import record_metric

load_dotenv()

# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds
BACKOFF_FACTOR = 2.0
DEFAULT_MODEL_TIMEOUT = 70.0
SLOW_REASONING_MODEL_TIMEOUT = 200.0


def model_timeout(model: str) -> float:
    normalized = model.lower()
    if "kimi-k3" in normalized or "kimi_k3" in normalized:
        return SLOW_REASONING_MODEL_TIMEOUT
    return DEFAULT_MODEL_TIMEOUT


def is_retriable_error(error: Exception) -> Tuple[bool, Optional[float]]:
    """Determine if an error is retriable. Returns (is_retriable, retry_after_seconds)."""
    error_str = str(error).lower()
    retry_after = None
    
    # Try to extract Retry-After from the error response
    # OpenAI SDK wraps the response in the exception
    if hasattr(error, 'response') and error.response is not None:
        response = error.response
        if hasattr(response, 'headers'):
            retry_after_header = response.headers.get('Retry-After')
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    pass
    
    # Rate limit errors
    if "rate_limit" in error_str or "rate limit" in error_str:
        return True, retry_after
    if "429" in error_str:
        return True, retry_after
    
    # Server errors (5xx)
    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return True, retry_after
    if "internal server error" in error_str:
        return True, retry_after
    if "bad gateway" in error_str:
        return True, retry_after
    if "service unavailable" in error_str:
        return True, retry_after
    if "gateway timeout" in error_str:
        return True, retry_after
    
    # Network errors
    if "connection" in error_str:
        return True, retry_after
    if "timeout" in error_str:
        return True, retry_after
    if "reset" in error_str or "refused" in error_str:
        return True, retry_after
    
    # CUDA errors (can be transient)
    if "cuda" in error_str:
        return True, retry_after
    
    return False, None


def retry_with_exponential_backoff(func):
    """Decorator to retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        last_exception = None
        delay = INITIAL_DELAY
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == MAX_RETRIES:
                    # Last attempt failed, don't retry
                    break
                
                is_retriable, retry_after = is_retriable_error(e)
                if not is_retriable:
                    # Non-retriable error, don't retry
                    break
                
                # Use Retry-After header if provided (for rate limits), otherwise use backoff
                if retry_after is not None and retry_after > 0:
                    sleep_time = min(retry_after, MAX_DELAY)
                    print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {sleep_time:.1f}s "
                          f"(from Retry-After header) due to: {type(e).__name__}: {str(e)[:100]}", flush=True)
                else:
                    # Add jitter to avoid thundering herd
                    jitter = random.uniform(0, 0.5)
                    sleep_time = min(delay + jitter, MAX_DELAY)
                    
                    print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {sleep_time:.1f}s "
                          f"due to: {type(e).__name__}: {str(e)[:100]}", flush=True)
                    
                    # Only increase delay if we didn't use Retry-After
                    if retry_after is None:
                        delay *= BACKOFF_FACTOR
                
                time.sleep(sleep_time)
        
        # All retries exhausted, raise the last exception
        raise last_exception
    
    return wrapper

# Multi-endpoint support
def get_endpoints() -> List[Dict]:
    """Parse multiple endpoints from environment variables."""
    endpoints = []
    
    # Support both old single URL and new multi-URL format
    base_urls_str = os.getenv("OPENAI_BASE_URLS") or os.getenv("OPENAI_BASE_URL")
    if not base_urls_str:
        return endpoints
    
    # Split by comma for multiple endpoints
    base_urls = [url.strip() for url in base_urls_str.split(",")]
    
    # Get API keys (can be single or comma-separated)
    api_keys_str = os.getenv("API_KEYS") or os.getenv("API_KEY")
    if api_keys_str:
        api_keys = [k.strip() for k in api_keys_str.split(",")]
    else:
        api_keys = []
    
    # Get embedding-specific keys/URLs
    embeddings_api_keys_str = os.getenv("EMBEDDINGS_API_KEYS") or os.getenv("EMBEDDINGS_API_KEY")
    if embeddings_api_keys_str:
        embeddings_api_keys = [k.strip() for k in embeddings_api_keys_str.split(",")]
    else:
        embeddings_api_keys = []
    
    embeddings_base_urls_str = os.getenv("OPENAI_EMBEDDINGS_BASE_URLS") or os.getenv("OPENAI_EMBEDDINGS_BASE_URL")
    if embeddings_base_urls_str:
        embeddings_base_urls = [url.strip() for url in embeddings_base_urls_str.split(",")]
    else:
        embeddings_base_urls = []
    
    # Create endpoint configurations
    for idx, base_url in enumerate(base_urls):
        if not base_url:
            continue
            
        # Use corresponding API key or fall back to first/any
        api_key = api_keys[idx] if idx < len(api_keys) else (api_keys[0] if api_keys else os.getenv("API_KEY"))
        
        # Use corresponding embedding config or fall back to main endpoint
        emb_api_key = embeddings_api_keys[idx] if idx < len(embeddings_api_keys) else (embeddings_api_keys[0] if embeddings_api_keys else api_key)
        emb_base_url = embeddings_base_urls[idx] if idx < len(embeddings_base_urls) else base_url
        
        endpoints.append({
            "id": idx,
            "name": f"Endpoint-{idx+1}",
            "base_url": base_url,
            "api_key": api_key,
            "embedding_api_key": emb_api_key,
            "embedding_base_url": emb_base_url,
            "client": OpenAI(api_key=api_key, base_url=base_url),
            "embedding_client": OpenAI(api_key=emb_api_key, base_url=emb_base_url),
            "models": [],
            "model_metadata": {},
        })
    
    return endpoints

# Global endpoint registry
ENDPOINTS = get_endpoints()
client = ENDPOINTS[0]["client"] if ENDPOINTS else None
embedding_client = ENDPOINTS[0]["embedding_client"] if ENDPOINTS else None
_registry_client = client
_registry_embedding_client = embedding_client


def get_available_models() -> List[str]:
    """Compatibility wrapper for the original single-endpoint API."""
    if client is not None and client is not _registry_client:
        models = client.models.list()
        return [model.id for model in models.data]
    if ENDPOINTS:
        return [model for model, _endpoint_id in get_all_models_from_endpoints()]
    if client is None:
        return []
    models = client.models.list()
    return [model.id for model in models.data]


def get_endpoint_clients(endpoint_id: int = 0):
    """Get client and embedding_client for a specific endpoint."""
    patched_client = client is not None and client is not _registry_client
    patched_embedding = (
        embedding_client is not None
        and embedding_client is not _registry_embedding_client
    )
    if endpoint_id == 0 and (patched_client or patched_embedding):
        active_client = client if patched_client else _registry_client
        active_embedding_client = (
            embedding_client if patched_embedding else _registry_embedding_client
        )
        return active_client, active_embedding_client or active_client
    if not ENDPOINTS and endpoint_id == 0 and client is not None:
        return client, embedding_client or client
    if endpoint_id >= len(ENDPOINTS):
        raise ValueError(f"Endpoint {endpoint_id} not found")
    ep = ENDPOINTS[endpoint_id]
    return ep["client"], ep["embedding_client"]


def is_embedding_model(model):
    if "embedding" in model.lower():
        return True
    return model in EMBEDDING_MODELS


def model_metadata(model: str, endpoint_id: int = 0) -> Dict:
    if endpoint_id >= len(ENDPOINTS):
        return {}
    return ENDPOINTS[endpoint_id].get("model_metadata", {}).get(model, {})


def detect_model_type(model: str, endpoint_id: int = 0) -> str:
    """Return the API capability to probe for a discovered model."""
    metadata = model_metadata(model, endpoint_id)
    model_lower = model.lower()
    task_type = str(metadata.get("task_type") or "").lower()
    pipeline = " ".join(
        str(metadata.get(key) or "").lower()
        for key in ("pipeline_name", "pipeline_class")
    )

    if is_embedding_model(model):
        return "embedding"
    if (
        "whisper" in model_lower
        or "transcription" in task_type
        or "speech-to-text" in task_type
        or "automatic-speech-recognition" in task_type
    ):
        return "audio"
    if (
        any(marker in task_type for marker in ("ti2v", "t2v", "video"))
        or "video" in pipeline
        or "minimax-h3" in model_lower
    ):
        return "video"
    if "image" in task_type or "image" in pipeline:
        return "image"
    return "chat"


def get_all_models_from_endpoints() -> List[Tuple[str, int]]:
    """Fetch models from all configured endpoints.
    Returns list of (model_id, endpoint_id) tuples.
    """
    all_models = []
    
    if not ENDPOINTS:
        print("DEBUG: No endpoints configured", flush=True)
        return all_models
    
    for ep in ENDPOINTS:
        try:
            client = ep["client"]
            base_url = ep["base_url"]
            print(f"DEBUG: Fetching models from {ep['name']} ({base_url})...", flush=True)
            models = client.models.list()
            model_list = [model.id for model in models.data]
            metadata = {}
            for model in models.data:
                if hasattr(model, "model_dump"):
                    metadata[model.id] = model.model_dump()
                else:
                    metadata[model.id] = dict(vars(model))
            msg = f"DEBUG: {ep['name']}: Found {len(model_list)} models: {model_list}"
            print(msg, flush=True)
            # Store models in endpoint registry
            ep["models"] = model_list
            ep["model_metadata"] = metadata
            # Add to all_models with endpoint ID
            for model_id in model_list:
                all_models.append((model_id, ep["id"]))
        except Exception as e:
            err_msg = f"DEBUG: {ep['name']}: Error fetching models: {type(e).__name__}: {str(e)}"
            print(err_msg, flush=True)
            import traceback
            traceback.print_exc()
            ep["models"] = []
            ep["model_metadata"] = {}
    
    return all_models


def extract_usage_tokens(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is not None:
        return total_tokens
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is not None or completion_tokens is not None:
        return (prompt_tokens or 0) + (completion_tokens or 0)
    return None


def consume_completion_stream(stream):
    # Keep compatibility with OpenAI-compatible gateways that ignore stream=True
    # and return a regular completion object.
    choices = getattr(stream, "choices", None)
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
            )
            return (
                content.strip() if isinstance(content, str) else "",
                reasoning.strip() if isinstance(reasoning, str) else "",
                getattr(choice, "finish_reason", None),
                extract_usage_tokens(stream),
            )

    content_parts = []
    reasoning_parts = []
    finish_reason = None
    tokens_used = None

    for chunk in stream:
        chunk_tokens = extract_usage_tokens(chunk)
        if chunk_tokens is not None:
            tokens_used = chunk_tokens

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = getattr(choice, "delta", None)
        if not delta:
            continue

        content = getattr(delta, "content", None)
        if isinstance(content, str):
            content_parts.append(content)

        reasoning = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
        )
        if isinstance(reasoning, str):
            reasoning_parts.append(reasoning)

    return (
        "".join(content_parts).strip(),
        "".join(reasoning_parts).strip(),
        finish_reason,
        tokens_used,
    )


@retry_with_exponential_backoff
def get_llm_response(prompt, model, endpoint_id: int = 0):
    response = None
    try:
        client, _ = get_endpoint_clients(endpoint_id)
        
        def is_qwen3_model(m):
            """Check if model is Qwen3 family that supports enable_thinking parameter."""
            return (
                "qwen3" in m.lower()
                or "alias-code" in m.lower()  # Qwen3-Coder
                or "alias-large" in m.lower()  # Qwen3-based
                or "alias-mis" in m.lower()   # Qwen3-based
            )
            
        def supports_thinking_toggle(m):
            """Check if model supports the enable_thinking parameter.
            
            Only Qwen3 family models support this. Other models (MiniMax, etc.)
            will return None content if this parameter is sent.
            """
            return is_qwen3_model(m)

        def is_kimi_k3_model(m):
            normalized = m.lower()
            return "kimi-k3" in normalized or "kimi_k3" in normalized

        def request_completion(max_tokens):
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "top_p": 0.8,
                "n": 1,
                "max_tokens": max_tokens,
                "stop": None,
                "stream": True,
                "stream_options": {"include_usage": True},
                "presence_penalty": 1.5,
                "frequency_penalty": 0,
            }
            if supports_thinking_toggle(model):
                kwargs["extra_body"] = {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            if is_kimi_k3_model(model):
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"thinking_effort": "low"},
                }
            return client.chat.completions.create(**kwargs)

        response = request_completion(30)
        content, reasoning, finish_reason, tokens_used = consume_completion_stream(response)
        if content:
            return content, tokens_used
        if reasoning and finish_reason == "length":
            # Thinking models: try with more tokens to get actual content.
            response = request_completion(512)
            content, reasoning_retry, _, tokens_used = consume_completion_stream(response)
            if content:
                return content, tokens_used
            if reasoning_retry:
                return reasoning_retry[:500], tokens_used
        # Some models (like alias-mis) only output to the reasoning field.
        if reasoning:
            return reasoning[:500], tokens_used
        return "An error occurred: Empty response content from LLM", None
    except UnboundLocalError:
        return "", None
    except Exception as e:
        extra = ""
        has_choices = (
            "response" in locals()
            and response
            and hasattr(response, "choices")
            and response.choices
        )
        if has_choices:
            try:
                first_choice = response.choices[0]
                message = getattr(first_choice, "message", None)
                content = getattr(message, "content", None)
                if message and isinstance(content, str):
                    extra = f" on {content}"
            except Exception:
                extra = ""
        return f"An error occurred: {str(e)}{extra}", None


@retry_with_exponential_backoff
def get_embedding_response(text, model, endpoint_id: int = 0):
    response = None
    try:
        _, embedding_client = get_endpoint_clients(endpoint_id)
        response = embedding_client.embeddings.create(
            model=model,
            input=text,
        )
        if response and hasattr(response, "data") and response.data:
            embedding = getattr(response.data[0], "embedding", None)
            if isinstance(embedding, list) and embedding:
                return embedding, extract_usage_tokens(response)
            return "An error occurred: Empty embedding response", None
        else:
            return "An error occurred: Invalid embedding response", None
    except UnboundLocalError:
        return "", None
    except Exception as e:
        return f"An error occurred: {str(e)}", None


def _endpoint_request_config(endpoint_id: int) -> Tuple[str, Dict[str, str]]:
    if endpoint_id >= len(ENDPOINTS):
        raise ValueError(f"Endpoint {endpoint_id} not found")
    endpoint = ENDPOINTS[endpoint_id]
    headers = {}
    if endpoint.get("api_key"):
        headers["Authorization"] = f"Bearer {endpoint['api_key']}"
    return endpoint["base_url"].rstrip("/"), headers


def _specialized_endpoint_candidates(model: str, endpoint_id: int):
    """Yield the requested endpoint, then alternate gateways advertising the model."""
    yield endpoint_id
    for candidate_id, endpoint in enumerate(ENDPOINTS):
        if candidate_id == endpoint_id:
            continue
        if model in endpoint.get("models", []):
            yield candidate_id


@retry_with_exponential_backoff
def get_audio_response(model: str, endpoint_id: int = 0):
    """Exercise an OpenAI-compatible transcription endpoint."""
    try:
        base_url, headers = _endpoint_request_config(endpoint_id)
        audio_path = Path(__file__).with_name("sample.wav")
        if not audio_path.exists():
            return f"An error occurred: Audio fixture not found: {audio_path}", None
        with audio_path.open("rb") as audio_file:
            response = httpx.post(
                f"{base_url}/audio/transcriptions",
                headers=headers,
                files={"file": (audio_path.name, audio_file, "audio/wav")},
                data={"model": model, "language": "en"},
                timeout=40.0,
            )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text")
        if isinstance(text, str):
            return text or "transcription endpoint responded", None
        return "An error occurred: Invalid transcription response", None
    except Exception as error:
        return f"An error occurred: {error}", None


@retry_with_exponential_backoff
def get_video_response(model: str, endpoint_id: int = 0):
    """Probe the video route without starting an expensive generation job."""
    try:
        last_response = None
        for candidate_id in _specialized_endpoint_candidates(model, endpoint_id):
            base_url, headers = _endpoint_request_config(candidate_id)
            response = httpx.post(
                f"{base_url}/videos",
                headers={**headers, "Content-Type": "application/json"},
                json={"model": model},
                timeout=20.0,
            )
            last_response = response
            endpoint_name = ENDPOINTS[candidate_id].get("name", f"Endpoint-{candidate_id + 1}")
            if response.is_success:
                return f"video endpoint accepted request via {endpoint_name}", None
            # Missing-prompt validation proves routing and the video API are
            # alive without consuming GPUs for a watchdog artifact.
            body = response.text.lower()
            if response.status_code in (400, 422) and "prompt" in body:
                return f"video endpoint healthy via {endpoint_name}", None
            # A catalog gateway may expose the model but not its specialized
            # route. Only that case should fall through to another endpoint.
            if response.status_code != 404:
                response.raise_for_status()
        if last_response is not None:
            last_response.raise_for_status()
        return "An error occurred: Invalid video response", None
    except Exception as error:
        return f"An error occurred: {error}", None


@retry_with_exponential_backoff
def get_image_response(model: str, endpoint_id: int = 0):
    """Probe the image route without starting an expensive generation job."""
    try:
        base_url, headers = _endpoint_request_config(endpoint_id)
        response = httpx.post(
            f"{base_url}/images/generations",
            headers={**headers, "Content-Type": "application/json"},
            json={"model": model},
            timeout=20.0,
        )
        if response.is_success:
            return "image endpoint accepted request", None
        body = response.text.lower()
        if response.status_code in (400, 422) and "prompt" in body:
            return "image endpoint healthy (prompt validation passed)", None
        response.raise_for_status()
        return "An error occurred: Invalid image response", None
    except Exception as error:
        return f"An error occurred: {error}", None


def check_model(model, word, prompt, endpoint_id: int = 0):
    model_type = detect_model_type(model, endpoint_id)
    if model_type == "embedding":
        response, tokens_used = get_embedding_response(word, model, endpoint_id)
        if response == "An error occurred: Empty embedding response":
            return False, response, tokens_used
        if response == "An error occurred: Invalid embedding response":
            return False, response, tokens_used
        if isinstance(response, list):
            return True, response, tokens_used
        if "CUDA error:" in response:
            return False, response, tokens_used
        if "Internal Server Error" in response:
            return False, response, tokens_used
        return False, response, tokens_used

    if model_type == "audio":
        response, tokens_used = get_audio_response(model, endpoint_id)
        return not response.startswith("An error occurred:"), response, tokens_used

    if model_type == "video":
        response, tokens_used = get_video_response(model, endpoint_id)
        return not response.startswith("An error occurred:"), response, tokens_used

    if model_type == "image":
        response, tokens_used = get_image_response(model, endpoint_id)
        return not response.startswith("An error occurred:"), response, tokens_used

    response, tokens_used = get_llm_response(prompt, model, endpoint_id)
    if response == "An error occurred: Empty response content from LLM":
        return False, response, tokens_used
    if response == "An error occurred: Invalid response from LLM":
        return False, response, tokens_used
    if word.lower() in response.lower():
        return True, response, tokens_used
    if "CUDA error:" in response:
        return False, response, tokens_used
    if "Internal Server Error" in response:
        return False, response, tokens_used
    return False, response, tokens_used


class ModelStatus(Static):
    def __init__(self, model):
        self.model = model
        self.status = "PENDING"
        self.elapsed = None
        super().__init__("", classes="pending")

    def render(self):
        max_width = 30
        if self.app:
            columns = 3
            gutter = 2
            grid_padding = 4
            cell_padding = 4
            border_width = 2
            available = self.app.size.width - grid_padding
            available -= (columns - 1) * gutter
            column_width = max(available // columns, 1)
            max_width = max(column_width - cell_padding - border_width, 4)

        def truncate_text(value: str) -> str:
            if len(value) <= max_width:
                return value
            if max_width <= 1:
                return "…"
            return f"{value[:max_width - 1]}…"

        model_text = truncate_text(self.model)
        status_text = truncate_text(self.status)
        return f"{model_text}\n{status_text}"

    def set_status(self, status):
        self.status = status
        self.update(self.render())
        self.remove_class("pending", "ok", "fail")
        if status == "OK":
            self.add_class("ok")
        elif status == "FAIL":
            self.add_class("fail")
        else:
            self.add_class("pending")

    def set_elapsed(self, elapsed):
        self.elapsed = elapsed
        if elapsed is None:
            self.border_title = ""
        else:
            self.border_title = f"{elapsed:.2f}s"
        self.refresh()


class WatchdogApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: #0f1217;
        color: #e6e6e6;
    }

    #title {
        height: 1;
        padding: 0 2;
        text-style: bold;
        background: #161b22;
    }

    #grid {
        layout: grid;
        grid-size: 3;
        grid-columns: 1fr 1fr 1fr;
        grid-gutter: 1 2;
        padding: 1 2;
    }

    #status {
        height: 1;
        padding: 0 2;
        color: #9aa4b2;
        background: #161b22;
    }

    ModelStatus {
        border: round #2a3240;
        padding: 1 2;
        height: 6;
        text-align: left;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        min-width: 0;
        overflow: hidden hidden;
        content-align: left middle;
    }

    ModelStatus.pending {
        background: #1b1f2a;
        color: #9aa4b2;
    }

    ModelStatus.ok {
        background: #0f2b1a;
        color: #7de6b4;
        border: round #1f5f3f;
    }

    ModelStatus.fail {
        background: #351315;
        color: #ff9a9a;
        border: round #6b2c2f;
    }
    """

    def __init__(self):
        super().__init__()
        self.model_widgets = {}
        self.models = []
        self.grid = None
        self.status_line = None

    def compose(self) -> ComposeResult:
        yield Static("Blablador Watchdog", id="title")
        self.grid = Container(id="grid")
        yield self.grid
        self.status_line = Static("Loading models...", id="status")
        yield self.status_line

    async def on_mount(self):
        await self.load_models()

    async def load_models(self):
        if self.status_line:
            self.status_line.update("Loading models...")
        models_with_endpoints = await asyncio.to_thread(get_all_models_from_endpoints)
        self.models_with_endpoints = models_with_endpoints
        self.model_widgets = {}

        if not models_with_endpoints:
            if self.status_line:
                self.status_line.update("No models available.")
            return

        # Group models by endpoint for display
        for model, endpoint_id in models_with_endpoints:
            endpoint_name = ENDPOINTS[endpoint_id]["name"] if endpoint_id < len(ENDPOINTS) else f"Endpoint-{endpoint_id}"
            widget = ModelStatus(f"{model} [{endpoint_name}]")
            widget.model = model  # Store original model name
            widget.endpoint_id = endpoint_id  # Store endpoint ID
            self.model_widgets[(model, endpoint_id)] = widget
            if self.grid:
                self.grid.mount(widget)

        if self.status_line:
            self.status_line.update(f"Testing {len(models_with_endpoints)} model-endpoint pairs...")
        self.run_worker(self.run_checks(), exclusive=True)

    async def run_checks(self):
        word = "potato"
        prompt = (
            f"Give me ONLY a word. The word is {word}. Nothing else. "
            "No sentences, no explanations, no definitions. Just the word."
        )
        successes = []
        failures = []

        def format_tokens_per_s(value):
            if value is None:
                return "n/a"
            return f"{value:.1f} tok/s"

        def tokens_sort_key(tokens_per_s):
            if tokens_per_s is None:
                return -1
            return tokens_per_s

        async def check_one(model: str, endpoint_id: int):
            start = time.monotonic()
            timeout = model_timeout(model)
            try:
                ok, response, tokens_used = await asyncio.wait_for(
                    asyncio.to_thread(check_model, model, word, prompt, endpoint_id),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                return model, endpoint_id, False, f"Timeout after {timeout:g}s", elapsed, None
            elapsed = time.monotonic() - start
            return model, endpoint_id, ok, response, elapsed, tokens_used

        tasks = [asyncio.create_task(check_one(model, ep_id)) for model, ep_id in self.models_with_endpoints]
        total = len(tasks)
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            if len(result) == 5:
                # Old format
                model, ok, response, elapsed, tokens_used = result
                endpoint_id = 0
            else:
                # New format with endpoint
                model, endpoint_id, ok, response, elapsed, tokens_used = result
            if self.status_line:
                endpoint_name = ENDPOINTS[endpoint_id]["name"] if endpoint_id < len(ENDPOINTS) else f"Endpoint-{endpoint_id}"
                self.status_line.update(f"Testing {index}/{total}: {model} [{endpoint_name}]")
            widget = self.model_widgets.get((model, endpoint_id))
            if widget:
                widget.set_status("OK" if ok else "FAIL")
                widget.set_elapsed(elapsed)
            tokens_per_s = None
            if tokens_used and elapsed and elapsed > 0:
                tokens_per_s = tokens_used / elapsed
            error_msg = None if ok else response
            await record_metric(
                model, ok, elapsed, tokens_used, tokens_per_s, error_msg
            )
            if ok:
                successes.append((model, endpoint_id, elapsed, tokens_per_s))
            else:
                failures.append((model, endpoint_id, response, elapsed, tokens_per_s))

        successes.sort(key=lambda item: tokens_sort_key(item[3]), reverse=True)
        failures.sort(key=lambda item: tokens_sort_key(item[4]), reverse=True)

        report_lines = [f"Successes ({len(successes)}):"]
        if successes:
            for model, endpoint_id, elapsed, tokens_per_s in successes:
                endpoint_name = ENDPOINTS[endpoint_id]["name"] if endpoint_id < len(ENDPOINTS) else f"Endpoint-{endpoint_id}"
                tps_str = format_tokens_per_s(tokens_per_s)
                line = f"[{endpoint_name}] {model} :: {elapsed:.2f}s :: {tps_str}"
                report_lines.append(line)
        else:
            report_lines.append("- none")
        report_lines.extend(["", f"Failures ({len(failures)}):"])
        if failures:
            for model, endpoint_id, error, elapsed, tokens_per_s in failures:
                endpoint_name = ENDPOINTS[endpoint_id]["name"] if endpoint_id < len(ENDPOINTS) else f"Endpoint-{endpoint_id}"
                tps_str = format_tokens_per_s(tokens_per_s)
                line = f"[{endpoint_name}] {model} :: {error} :: {elapsed:.2f}s :: {tps_str}"
                report_lines.append(line)
        else:
            report_lines.append("- none")
        report = "\n".join(report_lines)
        self.exit(message=report)


async def run_quiet():
    has_api_key = bool(os.getenv("API_KEY"))
    print(f"DEBUG: API_KEY set: {has_api_key}", flush=True)
    
    # Print configured endpoints
    if ENDPOINTS:
        print(f"DEBUG: Configured {len(ENDPOINTS)} endpoint(s):", flush=True)
        for ep in ENDPOINTS:
            print(f"  - {ep['name']}: {ep['base_url']}", flush=True)
    else:
        print("DEBUG: No endpoints configured", flush=True)
        return
    
    word = "potato"
    prompt = (
        f"Give me ONLY a word. The word is {word}. Nothing else. "
        "No sentences, no explanations, no definitions. Just the word."
    )
    models_with_endpoints = await asyncio.to_thread(get_all_models_from_endpoints)
    if not models_with_endpoints:
        print("No models available.")
        return

    async def check_one(model: str, endpoint_id: int):
        start = time.monotonic()
        timeout = model_timeout(model)
        endpoint_name = ENDPOINTS[endpoint_id]["name"] if endpoint_id < len(ENDPOINTS) else f"Endpoint-{endpoint_id}"
        try:
            ok, response, tokens_used = await asyncio.wait_for(
                asyncio.to_thread(check_model, model, word, prompt, endpoint_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            return model, endpoint_name, False, f"Timeout after {timeout:g}s", elapsed, None
        elapsed = time.monotonic() - start
        return model, endpoint_name, ok, response, elapsed, tokens_used

    tasks = [asyncio.create_task(check_one(model, ep_id)) for model, ep_id in models_with_endpoints]
    successes = []
    failures = []

    for task in asyncio.as_completed(tasks):
        result = await task
        if len(result) == 5:
            # Old format (shouldn't happen now)
            model, ok, response, elapsed, tokens_used = result
            endpoint_name = "unknown"
        else:
            # New format with endpoint
            model, endpoint_name, ok, response, elapsed, tokens_used = result
        tokens_per_s = None
        if tokens_used and elapsed and elapsed > 0:
            tokens_per_s = tokens_used / elapsed
        error_msg = None if ok else response
        await record_metric(
            model, ok, elapsed, tokens_used, tokens_per_s, error_msg
        )
        if ok:
            successes.append((model, endpoint_name, elapsed, tokens_per_s))
        else:
            failures.append((model, endpoint_name, response, elapsed, tokens_per_s))

    successes.sort(key=lambda x: x[2] if x[2] else -1, reverse=True)
    failures.sort(key=lambda x: x[3] if x[3] else -1, reverse=True)

    def fmt_tps(val):
        return f"{val:.1f} tok/s" if val else "n/a"

    print(f"Successes ({len(successes)}):")
    if successes:
        for model, endpoint_name, elapsed, tokens_per_s in successes:
            print(f"  [{endpoint_name}] {model} :: {elapsed:.2f}s :: {fmt_tps(tokens_per_s)}")
    else:
        print("  - none")

    print(f"\nFailures ({len(failures)}):")
    if failures:
        for model, endpoint_name, error, elapsed, tokens_per_s in failures:
            tps_str = fmt_tps(tokens_per_s)
            print(f"  [{endpoint_name}] {model} :: {error} :: {elapsed:.2f}s :: {tps_str}")
    else:
        print("  - none")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quiet":
        asyncio.run(run_quiet())
    else:
        WatchdogApp().run()
