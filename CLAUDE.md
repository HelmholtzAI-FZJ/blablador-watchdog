# CLAUDE.md — blablador-watchdog

Operating notes for agents working in this repo. Read before changing probe
behaviour; update it in the same commit as the change it describes.

## Committing is part of the task

Every change here lands as a commit **and** updates this file in the same
commit. Do not finish a task with "want me to commit?" or with a pending doc
edit — Alex asked for this explicitly, so stop asking.

- Stage the files of your change by name (`git add main.py test_main.py
  CLAUDE.md`). Never `git add -A` / `git commit -a`: `README.md`,
  `vllm_jobs.py`, `test_vllm_jobs.py`, `bench2.py`, the generated PNGs and
  `shinrai/` routinely hold unrelated in-progress work, so the tree is usually
  dirty at session start even when the workspace snapshot says clean.
- No pushing unless asked.
- `git` needs the CommandLineTools toolchain on this Mac — plain `/usr/bin/git`
  dies with "You have not agreed to the Xcode license agreements":
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools git status`. Same root
  cause as the unusable system `python3`; `.venv/bin/python` is unaffected.
- If a write to this file is refused because an approval prompt timed out, say
  so and leave the proposed text for the user — do not report the doc as
  updated.

## What lives here

| File | Role |
|---|---|
| `main.py` | Probe engine + Textual dashboard + headless report (`--quiet`) |
| `metrics.py` | SQLite metric storage (`~/.blablador_watchdog/metrics.db`) |
| `plot_metrics.py` | Charts from the metrics DB (`metrics_plots.png`, gitignored) |
| `vllm_jobs.py` | SSH/SLURM job monitor across the supercomputers |
| `monitor-cluster.sh` | Redis/kubectl gateway monitoring (users, throttling) |
| `run.sh` | Entry point for cron: `./run.sh --quiet` |
| `test_main.py` | Offline unit tests for the probe engine (mocked, no network) |

## Commands

```bash
uv pip install -e ".[dev]"     # uv ONLY — never pip / python -m pip
./run.sh                       # Textual dashboard
python3 main.py --quiet        # headless report (what cron and CI run)
python3 -m pytest -q           # full suite, no network needed
```

Use `uv` for every Python operation. The suite must stay offline: patch
`main.client` / `main.ENDPOINTS` / `main.httpx.post`, never call the gateway.

## Probe dispatch — one route per model type

`detect_model_type()` decides what a catalog entry is; `check_model()`
dispatches. Never send the potato chat prompt to a non-chat model.

| Type | Detected by | Probe |
|---|---|---|
| `chat` | everything else | `/v1/chat/completions`, "the word is potato" |
| `embedding` | `embedding` in name, or `EMBEDDING_MODELS` | `embeddings.create` must return a vector |
| `rerank` | `rerank` in name | `POST /v1/rerank` (`get_rerank_response`) |
| `audio` | `whisper`, `task_type` transcription/speech-to-text | `POST /v1/audio/transcriptions` with `sample.wav` |
| `video` | `task_type` ti2v/t2v/video, `minimax-h3` | `POST /videos` — missing-prompt validation counts as healthy |
| `image` | `qwen-image`, `flux`, `alias-image`, `task_type`/pipeline image | `POST /v1/images/generations`, same validation trick; skipped entirely in the UI/report |

Evidence for the rerank route on `api.blablador.fz-juelich.de`
(`alias-qwen3-{4b,8b}-reranking`): chat completions → 404, embeddings → 500,
`/v1/rerank` → 200. vLLM's `/v1/score` exists but takes a single `text_2`
string, so it cannot rank a document list.

A 200 is not proof of health for reranking: the probe asserts the potato
document outranks an unrelated one and reports `Wrong ranking:` otherwise.
Route-validation probes (audio/video/image) intentionally do not start
expensive generation jobs — keep it that way.

Image models must be detected by name, because the gateway cannot tell us:
`api.blablador.fz-juelich.de` returns no `task_type` or `pipeline_*` metadata,
so `flux` (and any new generator alias) belongs in the marker list in
`detect_model_type()`. This is not cosmetic — `alias-flux2` answers the potato
prompt on `/v1/chat/completions` with HTTP 200 in ~20s and
`content: "![generated image](data:image/jpeg;base64,…)"`, i.e. a real
1024x1024 render, which the probe then reports as `Wrong output:` with the whole
data URL attached (868 KB measured in `model_metrics`).
`MAX_ERROR_REASON_CHARS` (200, applied in `check_model()` for every model type)
bounds failure reasons so no model payload reaches a report or the database.

## Thinking models: two tiers, plus a token budget that truncates

1. **Thinking can be disabled** — Qwen3 family (`qwen3*`, `alias-code`,
   `alias-large`, `alias-mis`): send
   `chat_template_kwargs: {"enable_thinking": False}`.
2. **Always-think models reject that toggle** (`qwen3.8*`, `alias-qwen-trillion`,
   `alias-qwen-huge`, `alias-huge`, `kimi-k3`): send
   `chat_template_kwargs: {"thinking_effort": "low"}` instead and allow a
   longer timeout. The list lives in `LOW_REASONING_ONLY` in `main.py` —
   add new offenders there, never in a per-function copy.
3. Do not send either toggle to non-Qwen models (e.g. `alias-fast`): they
   return `None` content.

Check `needs_low_reasoning()` **before** `supports_thinking_toggle()`: names
like `Qwen3.8-2.4T` contain `qwen3` but reject `enable_thinking`.

### Truncation is not a wrong answer

The probe asks for 30 tokens first. A thinking model can spend all of them on
reasoning and come back with `finish_reason="length"` and content `"pot"` —
that is a healthy model, not a failure (measured on
`07 - Qwen3.8-27B-FP8` and `90 - DeepSeek-V4-Flash-0731`).

Rules in `get_llm_response()`:

- Return content directly only when it is **not** truncated.
- On `finish_reason == "length"`, retry once with 512 tokens.
- If the retry is still truncated, fail with
  `An error occurred: Truncated output after 512 tokens: '...'` so a budget
  problem is never reported as wrong output.

Some models only ever write to the `reasoning` field (`alias-mis`); the probe
accepts that as a last resort, truncated to 500 chars.

## Reports, state, and CI

- The report starts with the model-list diff (`Models missing since last run`),
  then `Successes (n)` / `Failures (n)`. Keep the diff first: a vanished model
  is the signal the user reads for.
- Every run stores the discovered catalog (before the image filter) in
  `~/.blablador_watchdog/last_models.json` as `<endpoint_id>:<model>` tags.
  Never save an empty discovery — a gateway/auth failure would make the next
  run report every model as new.
- Both the TUI and `--quiet` use the same helpers (`model_tags`,
  `load_model_state`, `save_model_state`, `model_list_change_report`), so the
  two modes never disagree.
- The headless run must not exit non-zero when models fail: the Telegram
  wrapper uses `set -euo pipefail` and would skip the plot uploads.
- CI (`.github/workflows/python-app.yml`) runs `pytest` and then
  `python3 main.py --quiet` against the real gateway. Keep `openai<3` pinned —
  openai 3.x dropped the `httpx` dependency that `main.py` imports directly.
- Failure reasons are printed verbatim and stored in `model_metrics.error`, so
  they are capped at `MAX_ERROR_REASON_CHARS` (200) in `check_model()`. Scope
  metrics-DB queries by row id, not by timestamp: `timestamp` is ISO-8601 with
  a `T` separator, so `timestamp > datetime('now','-30 minutes')` matches every
  row of the same day.

## Tests

- One test class per behaviour, named after the bug it prevents
  (`TestRerankingModels`, `TestTruncatedThinkingOutput`, `TestModelListState`).
- Mock `MagicMock` clients/streams rather than hitting the gateway; a
  truncation test feeds a stream that ends with `finish_reason="length"`.
- Assert the request as well as the result (URL, payload, `max_tokens`
  sequence) — the route and the budget are the things that break.

## Version control

- Conventional commit subjects (`feat(probe):`, `fix(probe):`, `docs:`), body
  explaining the measured evidence, not the code.
- One logical change per commit; every commit must pass `pytest`.
- `deploy-dsv41/`, `shinrai/`, `bench2.py` and the generated PNGs are local
  scratch/artifacts — never sweep them into a commit.
- `.env` holds API keys and is gitignored. Never commit or print its contents.
