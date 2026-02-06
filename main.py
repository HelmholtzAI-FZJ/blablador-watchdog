import asyncio
import os
import time

from openai import OpenAI
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static

from embedding_models import EMBEDDING_MODELS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

embedding_client = OpenAI(
    api_key=os.getenv("EMBEDDINGS_API_KEY") or os.getenv("API_KEY"),
    base_url=os.getenv("OPENAI_EMBEDDINGS_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
)


def is_embedding_model(model):
    if "embedding" in model:
        return True
    return model in EMBEDDING_MODELS


def get_available_models():
    try:
        print(f"DEBUG: Fetching models from {os.getenv('OPENAI_BASE_URL')}...", flush=True)
        models = client.models.list()
        model_list = [model.id for model in models.data]
        print(f"DEBUG: Found {len(model_list)} models: {model_list}", flush=True)
        return model_list
    except Exception as e:
        print(f"DEBUG: Error fetching models: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return []


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


def get_llm_response(prompt, model):
    response = None
    try:
        def request_completion(max_tokens):
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=1,
                n=1,
                max_tokens=max_tokens,
                stop=None,
                stream=False,
                presence_penalty=0,
                frequency_penalty=0,
                reasoning_effort="low",
            )

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
                response = request_completion(256)
                if response and hasattr(response, "choices") and response.choices:
                    message = getattr(response.choices[0], "message", None)
                    content = getattr(message, "content", None) if message else None
                    if isinstance(content, str):
                        return content.strip(), extract_usage_tokens(response)
            return "An error occurred: Empty response content from LLM", None
        else:
            return "An error occurred: Invalid response from LLM", None
    except UnboundLocalError as e:
        return "", None
    except Exception as e:
        extra = ""
        if 'response' in locals() and response and hasattr(response, 'choices') and response.choices:
            try:
                message = getattr(response.choices[0], "message", None)
                content = getattr(message, "content", None) if message else None
                if isinstance(content, str):
                    extra = f" on {content}"
            except Exception:
                extra = ""
        return f"An error occurred: {str(e)}{extra}", None


def get_embedding_response(text, model):
    response = None
    try:
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
    except UnboundLocalError as e:
        return "", None
    except Exception as e:
        return f"An error occurred: {str(e)}", None


def check_model(model, word, prompt):
    if is_embedding_model(model):
        response, tokens_used = get_embedding_response(word, model)
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

    response, tokens_used = get_llm_response(prompt, model)
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
            available = self.app.size.width - grid_padding - (columns - 1) * gutter
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
        models = await asyncio.to_thread(get_available_models)
        self.models = models
        self.model_widgets = {}

        if not models:
            if self.status_line:
                self.status_line.update("No models available.")
            return

        for model in models:
            widget = ModelStatus(model)
            self.model_widgets[model] = widget
            if self.grid:
                self.grid.mount(widget)

        if self.status_line:
            self.status_line.update(f"Testing {len(models)} models...")
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

        async def check_one(model: str):
            start = time.monotonic()
            try:
                ok, response, tokens_used = await asyncio.wait_for(
                    asyncio.to_thread(check_model, model, word, prompt),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                return model, False, "Timeout after 45s", elapsed, None
            elapsed = time.monotonic() - start
            return model, ok, response, elapsed, tokens_used

        tasks = [asyncio.create_task(check_one(model)) for model in self.models]
        total = len(tasks)
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            model, ok, response, elapsed, tokens_used = await task
            if self.status_line:
                self.status_line.update(f"Testing {index}/{total}: {model}")
            widget = self.model_widgets.get(model)
            if widget:
                widget.set_status("OK" if ok else "FAIL")
                widget.set_elapsed(elapsed)
            tokens_per_s = None
            if tokens_used and elapsed and elapsed > 0:
                tokens_per_s = tokens_used / elapsed
            if ok:
                successes.append((model, elapsed, tokens_per_s))
            else:
                failures.append((model, response, elapsed, tokens_per_s))

        successes.sort(key=lambda item: tokens_sort_key(item[2]), reverse=True)
        failures.sort(key=lambda item: tokens_sort_key(item[3]), reverse=True)

        report_lines = [f"Successes ({len(successes)}):"]
        if successes:
            report_lines.extend(
                f"{model} :: {elapsed:.2f}s :: {format_tokens_per_s(tokens_per_s)}"
                for model, elapsed, tokens_per_s in successes
            )
        else:
            report_lines.append("- none")
        report_lines.extend(["", f"Failures ({len(failures)}):"])
        if failures:
            report_lines.extend(
                (
                    f"{model} :: {error} :: {elapsed:.2f}s :: "
                    f"{format_tokens_per_s(tokens_per_s)}"
                )
                for model, error, elapsed, tokens_per_s in failures
            )
        else:
            report_lines.append("- none")
        report = "\n".join(report_lines)
        self.exit(message=report)


async def run_quiet():
    print(f"DEBUG: API_KEY set: {bool(os.getenv('API_KEY'))}", flush=True)
    print(f"DEBUG: OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'not set')}", flush=True)
    word = "potato"
    prompt = (
        f"Give me ONLY a word. The word is {word}. Nothing else. "
        "No sentences, no explanations, no definitions. Just the word."
    )
    models = await asyncio.to_thread(get_available_models)
    if not models:
        print("No models available.")
        return

    async def check_one(model: str):
        start = time.monotonic()
        try:
            ok, response, tokens_used = await asyncio.wait_for(
                asyncio.to_thread(check_model, model, word, prompt),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            return model, False, "Timeout after 45s", elapsed, None
        elapsed = time.monotonic() - start
        return model, ok, response, elapsed, tokens_used

    tasks = [asyncio.create_task(check_one(model)) for model in models]
    successes = []
    failures = []

    for task in asyncio.as_completed(tasks):
        model, ok, response, elapsed, tokens_used = await task
        tokens_per_s = None
        if tokens_used and elapsed and elapsed > 0:
            tokens_per_s = tokens_used / elapsed
        if ok:
            successes.append((model, elapsed, tokens_per_s))
        else:
            failures.append((model, response, elapsed, tokens_per_s))

    successes.sort(key=lambda x: x[2] if x[2] else -1, reverse=True)
    failures.sort(key=lambda x: x[3] if x[3] else -1, reverse=True)

    def fmt_tps(val):
        return f"{val:.1f} tok/s" if val else "n/a"

    print(f"Successes ({len(successes)}):")
    if successes:
        for model, elapsed, tokens_per_s in successes:
            print(f"  {model} :: {elapsed:.2f}s :: {fmt_tps(tokens_per_s)}")
    else:
        print("  - none")

    print(f"\nFailures ({len(failures)}):")
    if failures:
        for model, error, elapsed, tokens_per_s in failures:
            print(f"  {model} :: {error} :: {elapsed:.2f}s :: {fmt_tps(tokens_per_s)}")
    else:
        print("  - none")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quiet":
        asyncio.run(run_quiet())
    else:
        WatchdogApp().run()
