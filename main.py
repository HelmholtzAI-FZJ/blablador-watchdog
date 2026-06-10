import asyncio
import os
import time
import random
from typing import Optional, Tuple, List, Dict

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
        })
    
    return endpoints

# Global endpoint registry
ENDPOINTS = get_endpoints()

def get_endpoint_clients(endpoint_id: int = 0):
    """Get client and embedding_client for a specific endpoint."""
    if endpoint_id >= len(ENDPOINTS):
        raise ValueError(f"Endpoint {endpoint_id} not found")
    ep = ENDPOINTS[endpoint_id]
    return ep["client"], ep["embedding_client"]


def is_embedding_model(model):
    if "embedding" in model:
        return True
    return model in EMBEDDING_MODELS


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
            msg = f"DEBUG: {ep['name']}: Found {len(model_list)} models: {model_list}"
            print(msg, flush=True)
            # Store models in endpoint registry
            ep["models"] = model_list
            # Add to all_models with endpoint ID
            for model_id in model_list:
                all_models.append((model_id, ep["id"]))
        except Exception as e:
            err_msg = f"DEBUG: {ep['name']}: Error fetching models: {type(e).__name__}: {str(e)}"
            print(err_msg, flush=True)
            import traceback
            traceback.print_exc()
            ep["models"] = []
    
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
                or "alias-huge" in m.lower()  # Qwen3.5-122B
                or "alias-mis" in m.lower()   # Qwen3-based
            )
            
        def supports_thinking_toggle(m):
            """Check if model supports the enable_thinking parameter.
            
            Only Qwen3 family models support this. Other models (MiniMax, etc.)
            will return None content if this parameter is sent.
            """
            return is_qwen3_model(m)

        def request_completion(max_tokens):
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "top_p": 0.8,
                "n": 1,
                "max_tokens": max_tokens,
                "stop": None,
                "stream": False,
                "presence_penalty": 1.5,
                "frequency_penalty": 0,
            }
            if supports_thinking_toggle(model):
                kwargs["extra_body"] = {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            return client.chat.completions.create(**kwargs)

        response = request_completion(30)
        if response and hasattr(response, 'choices') and response.choices:
            message = getattr(response.choices[0], "message", None)
            content = getattr(message, "content", None) if message else None
            if isinstance(content, str):
                return content.strip(), extract_usage_tokens(response)
            reasoning = None
            if message:
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                )
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if reasoning and finish_reason == "length":
                # Thinking models: try with more tokens to get actual content
                response = request_completion(512)
                has_choices = response and hasattr(response, "choices")
                if has_choices and response.choices:
                    message = getattr(response.choices[0], "message", None)
                    content = getattr(message, "content", None)
                    if message and isinstance(content, str):
                        return content.strip(), extract_usage_tokens(response)
                    # If still no content but has reasoning, accept reasoning as valid response
                    reasoning_retry = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                    if reasoning_retry:
                        return reasoning_retry[:500], extract_usage_tokens(response)
            # Some models (like alias-mis) only output to reasoning field
            if reasoning:
                return reasoning[:500], extract_usage_tokens(response)
            return "An error occurred: Empty response content from LLM", None
        else:
            return "An error occurred: Invalid response from LLM", None
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


def check_model(model, word, prompt, endpoint_id: int = 0):
    if is_embedding_model(model):
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
            self.model_widgets[model] = widget
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
            try:
                ok, response, tokens_used = await asyncio.wait_for(
                    asyncio.to_thread(check_model, model, word, prompt, endpoint_id),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                return model, endpoint_id, False, "Timeout after 45s", elapsed, None
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
            widget = self.model_widgets.get(model)
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
        endpoint_name = ENDPOINTS[endpoint_id]["name"] if endpoint_id < len(ENDPOINTS) else f"Endpoint-{endpoint_id}"
        try:
            ok, response, tokens_used = await asyncio.wait_for(
                asyncio.to_thread(check_model, model, word, prompt, endpoint_id),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            return model, endpoint_name, False, "Timeout after 45s", elapsed, None
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
