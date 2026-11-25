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
| `decision` | metadata `decision_endpoint` or `capabilities: ["systemone"]`, or `clef` in name (`alias-clef`) | `POST` the advertised route with a `choice` question (`state` + `questions`, criteria potato/quark); must pick `potato`, else `Wrong decision:` |
| `speech` | `voice` / `drama` in name, `task_type` text-to-speech (`alias-voice-drama3`) | `POST /v1/audio/speech` with `input: "potato"`, body must start with an audio magic number |
| `audio` | `whisper`, `task_type` transcription/speech-to-text | `POST /v1/audio/transcriptions` with `sample.wav` |
| `video` | `task_type` ti2v/t2v/video, `modalities.output` video, `minimax-h3`, `alias-video` | `POST /videos` JSON; on 404 `POST /videos/sync` multipart with only `model` — missing-prompt validation counts as healthy |
| `image` | `qwen-image`, `flux`, `alias-image`, `task_type`/pipeline image | `POST /v1/images/generations`: one real 50x50, 4-step render (alias-image snaps it to 32x32) of "a potato"; decoded `b64_json` must be PNG/JPEG/WebP/GIF |
| `molecular` | `MOLECULAR_MODELS` (the five BioPortal aliases) or metadata `capabilities.molecular_jobs` | `POST /v1/responses` with `input` as a JSON *string* `{"action": "queue"}`; the reply must carry a `molecular_job` whose `object` is `molecular.queue` **and** `model_workers_ready > 0` |
| `commands` | `COMMAND_MODELS` (`vggt-omega-1b`, `alias-visual-geometry`), metadata `owned_by: vggt` or `capabilities.commands` | `POST /v1/chat/completions` whose single user message is the *command* `queue`; `content` must parse to `{"queue": {...}}` |

Evidence for the rerank route on `api.blablador.fz-juelich.de`
(`alias-qwen3-{4b,8b}-reranking`): chat completions → 404, embeddings → 500,
`/v1/rerank` → 200. vLLM's `/v1/score` exists but takes a single `text_2`
string, so it cannot rank a document list.

A 200 is not proof of health for reranking: the probe asserts the potato
document outranks an unrelated one and reports `Wrong ranking:` otherwise.
Video stays a route-validation probe (missing-prompt 400 counts as healthy) and
must not start a generation job. Image and speech models *do* generate, because
Alex asked for them to be really tested and the cost is small (measured
2026-10-19: alias-flux2 1.0s, alias-image 15.3s, alias-voice-drama3 4.2s,
~2 KB per image at 50x50, 90 KB per voice clip). Probes return only the byte count — never the payload.
Image models are no longer skipped in the UI/report.

`alias-clef` 404s on chat and only answers its advertised `/v1/systemone`
(listed in `openapi.json` since 2026-10). That route takes no prompt: it rejects anything
without `state` and a non-empty `questions` object whose entries are typed
`choice` (needs a `criteria` object), `noul` or `score` (needs a `criteria`
list). Measured 2026-10-19: potato chosen at 0.96–0.99 confidence in <1s. Measured 2026-10-21: the gateway served both `alias-clef` and `alias-video`
with **no metadata**, so both fell to the chat probe (clef → 404, video →
`Use POST /v1/videos/sync for alias-video`). Detection therefore falls back to
the name. `alias-video` 404s on `/v1/videos`; `/v1/videos/sync` takes
multipart form data and a model-only form returns `400 prompt is required`
from the backend (~3s) — never send a prompt there, it renders synchronously.

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

### Molecular job queues (BioPortal) and VGGT are not chat models

Neither family has ever answered the potato prompt; both used to spend the
whole run in `Failures` as false alarms. Measured 2026-11-18 on
`api.blablador.fz-juelich.de`:

- `alias-esmfold2`, `alias-bioemu2`, `alias-rfdiffusion3`, `alias-bindcraft`,
  `alias-proteina-complexa` answer `/v1/chat/completions` with `400 Use
  non-streaming molecular actions; jobs run asynchronously`. They take
  `POST /v1/responses` with `stream: false` and `input` as a **JSON string**
  holding an action (`submit`/`status`/`cancel`/`queue`/`artifact`), and only
  `molecular_job` in the reply is real — the router completes without the job.
  The probe therefore sends `{"action": "queue"}` (0.4–0.5 s, no GPU time, no
  job created) and requires `object: molecular.queue` with
  `model_workers_ready > 0`. **Never probe with `submit`**: that starts a real
  design run on someone's account. This protocol is BioPortal's
  (`../bioportal`, `jobs/blablador.py`); the capability is advertised as
  `capabilities.molecular_jobs`, but the aliases are also listed by name
  because the catalog has been served without metadata before.
- `vggt-omega-1b` and `alias-visual-geometry` *do* answer chat completions,
  but the message body is a command line: the potato prompt gets `400 unknown
  command 'give'; use step, finish, download, status or queue`. The probe sends
  the `queue` command (0.4 s) and requires the reply to parse to
  `{"queue": {"gpu_busy": …, "waiting": …}}`. A busy GPU is the model's own
  state, not a failure; a reply without the queue object is.

Both routes are additions to the dispatch in `_probe_model()`; keep their
detection *before* the prompt-based routes, since the potato prompt can never
succeed on either.

## Thinking models: two tiers, plus a token budget that truncates

1. **Thinking can be disabled** — Qwen3 family (`qwen3*`, `alias-code`,
   `alias-large`, `alias-mis`): send
   `chat_template_kwargs: {"enable_thinking": False}`.
2. **Always-think models reject that toggle** (`qwen3.8*`, `alias-qwen-trillion`,
   `alias-qwen-huge`, `alias-huge`, `kimi-k3`): send
   `chat_template_kwargs: {"thinking_effort": "low"}` instead and allow a
   longer timeout. The list lives in `LOW_REASONING_ONLY` in `main.py` —
   add new offenders there, never in a per-function copy. Alias spellings
   drop dot and space (`alias-qwen38-27b`), so they need their own marker;
   keep it as narrow as `qwen38-27b` — `alias-qwen38-quick` takes the toggle.
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

## Plots (`plot_metrics.py`)

With ~80 live models, one line per model is unreadable — never go back to it.
Charts are **one row per model**, sorted so problems sit on top:

- `metrics_plots.png`: fleet line (% of models passing per probe run, gaps >3h
  broken, runs <90% in red), a daily failure-rate grid (white→red, grey = not
  probed; currently failing models first, label in bold red with ✗), and a
  latency dot plot sharing the row order (7-day median, p10–p90 bar, previous
  week's median as a ring, ▲/▼ ratio when it moved ≥1.5×, log scale).
- `supercomputer_model_status.png`: hourly status timeline per
  `cluster: model` (status colours, last value carried forward) plus current
  job count.
- Only models probed in the last 3 days are shown; retired ones belong to the
  report's model-list diff. Aggregation lives in `summarize_models()` (pure).
- The supercomputer chart shows only deployments in each cluster's latest
  snapshot (within 10 min of its newest row) and drops names failing
  `vllm_jobs.is_valid_model_name()`, so renamed/stopped jobs and pre-fix junk
  rows don't linger for a week.

## Supercomputer job names (`vllm_jobs.py`)

`extract_model()` reads the launch script flattened to one line, so shell
syntax leaks into values. Before 2026-11-24 it stored `00`/`15`/`80` (quoted
names with spaces), `\` (line continuations), `$MODEL_ID`/`$ROOT`/`${names[@]}`
(variables) and `canary)` (last element of a bash `ARGS=(...)` array). Rules:
quoted values are read whole, `\` continuations skipped, `$VAR` resolved from
a `VAR=` assignment in the script, a lone trailing `)` stripped, and every
match is tried until one passes `is_valid_model_name()`. Fleet launchers that
serve several models from a loop report their `FLEET_MODEL`
(`trustllm-v2-8b-fleet`).

## Gateway user monitoring (`monitor-cluster.sh`)

Since 2026-11-20 `api.blablador.fz-juelich.de` is served by APISIX in
namespace `apisix-all-test` (context `blablador-import`; source:
gitlab.jsc.fz-juelich.de/rushchanskii2/blablador-apisix-api). The old
OpenResty Redis in `app-ns` gets no traffic — reading it gave empty top-user,
model, throttled and banned lists. Defaults: `KUBE_CONTEXT`,
`REDIS_NAMESPACE=apisix-all-test`, local port **6381** so a stale forward on
6379 to the old Redis is never reused.

- Rate limits are in the APISIX configmap (`rate_limits.default`: punished 1,
  external 20, internal 40, elevated 140 req/60s, superuser unlimited), not in
  `rate_limits:*` keys; `declare_default_limit` mirrors them. Update both
  together.
- The APISIX Redis has no persistence. `access_control:users` (bans,
  superuser/elevated grants) is filled by hand via the APISIX repo's
  `scripts/configure-users.sh` and vanishes on every pod restart; the summary
  prints a WARNING when the hash is missing. Measured 2026-11-24: missing
  after the 17h-ago restart.
- "Banned" has three sources, all shown by `banned-users` (and in `all`):
  `blacklisted` (403) / `punished` (1 req/min) entries in
  `access_control:users`, with request counts for those still active; the
  `access_control:blacklisted_domains` set (403 for the domain and every
  subdomain — `banned_domain_for` mirrors the APISIX suffix rule); and active
  users whose e-mail falls under a banned domain. Temporary `throttle:*`
  blocks stay in `throttled`. Seed lists live in the APISIX repo
  (`scripts/add_user_status.sh`, `scripts/add_domain_bans.sh`).
- Never print raw `rate:*` / `user:priority:*` key names: they contain users'
  API tokens (`glpat-…`). Show only prefixes or resolved e-mails.
