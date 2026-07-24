#!/usr/bin/env python3
import argparse
import asyncio
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from metrics import (
    DEFAULT_DB_PATH,
    get_recent_metrics,
    record_supercomputer_model_status,
)


DEFAULT_CLUSTERS = ("jureca", "booster", "jupiter", "haicluster1")
INFERENCE_SERVER_RE = re.compile(
    r"\b(vllm|api_server|openai[._-]api|whisper_openai_api)\b",
    re.I,
)
MODEL_ARG_RE = re.compile(r"(?:^|\s)--model(?:\s+|=)([^\s]+)")
MODEL_PATH_ARG_RE = re.compile(r"(?:^|\s)--model-path(?:\s+|=)([^\s]+)")
SERVED_MODEL_ARG_RE = re.compile(r"(?:^|\s)--served-model-name(?:\s+|=)([^\s]+)")
CONTAINER_MODEL_RE = re.compile(r"/(?:models|model|p/scratch|p/project)/([A-Za-z0-9_.:/@+-]+)")


@dataclass
class Job:
    cluster: str
    job_id: str
    state: str
    elapsed_seconds: int
    reason: str
    name: str
    command: str
    model: str
    detail: str = ""


@dataclass
class ModelStatus:
    cluster: str
    model: str
    status: str
    availability: bool
    concurrency: int
    jobs: list[Job] = field(default_factory=list)
    detail: str = ""

    @property
    def job_ids(self) -> str:
        return ",".join(job.job_id for job in self.jobs)


def shell_script() -> str:
    return r'''
set -o pipefail
now=$(date +%s)
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER}" -h -o "%i|%T|%M|%R|%j" 2>/dev/null | while IFS='|' read -r job state elapsed reason name; do
    command=$(scontrol show job "$job" 2>/dev/null | sed -n 's/^.*Command=//p' | sed 's/ WorkDir=.*$//' | head -n 1)
    command_detail="$command"
    if [ -f "$command" ] && [ -r "$command" ]; then
      script_detail=$(tr '\n|' '  ' < "$command")
      command_detail="$command $script_detail"
    fi
    printf 'ACTIVE|%s|%s|%s|%s|%s|%s\n' "$job" "$state" "$elapsed" "$reason" "$name" "$command_detail"
  done
fi
if command -v sacct >/dev/null 2>&1; then
  sacct -u "${USER}" -S now-12hours -X -n -P -o JobIDRaw,State,Elapsed,JobName 2>/dev/null \
    | awk -F'|' '$2 ~ /FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY/ {print "RECENT|" $0}'
fi
'''


def parse_elapsed(value: str) -> int:
    if not value or value == "Unknown":
        return 0
    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        days = int(day_part or 0)
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = 0, 0, parts[0]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def normalize_model(value: str) -> str:
    value = value.strip().strip("'\"")
    if not value:
        return "unknown-model"
    value = value.rstrip("/")
    if "/" in value and not value.startswith(("http://", "https://")):
        return value.split("/")[-1] or value
    return value


def extract_model(name: str, command: str) -> str:
    for pattern in (SERVED_MODEL_ARG_RE, MODEL_ARG_RE, MODEL_PATH_ARG_RE):
        match = pattern.search(command)
        if match:
            return normalize_model(match.group(1))
    match = CONTAINER_MODEL_RE.search(command)
    if match:
        return normalize_model(match.group(1))
    lowered_name = name.lower()
    if "vllm" in lowered_name or "model" in lowered_name:
        cleaned = re.sub(r"^(vllm|serve|model)[-_]?", "", name, flags=re.I)
        return normalize_model(cleaned)
    return "unknown-model"


def is_vllm_job(name: str, command: str) -> bool:
    return bool(INFERENCE_SERVER_RE.search(f"{name} {command}"))


def parse_job_rows(cluster: str, output: str) -> list[Job]:
    jobs = []
    for line in output.splitlines():
        fields = line.split("|", 6)
        if len(fields) < 5:
            continue
        kind = fields[0]
        if kind == "ACTIVE" and len(fields) == 6:
            _, job_id, state, elapsed, reason, name = fields
            command = ""
        elif kind == "ACTIVE" and len(fields) == 7:
            _, job_id, state, elapsed, reason, name, command = fields
        else:
            command = None
        if kind == "ACTIVE" and command is not None:
            if not is_vllm_job(name, command):
                continue
            jobs.append(
                Job(
                    cluster=cluster,
                    job_id=job_id,
                    state=state.upper(),
                    elapsed_seconds=parse_elapsed(elapsed),
                    reason=reason,
                    name=name,
                    command=command,
                    model=extract_model(name, command),
                )
            )
        elif kind == "RECENT" and len(fields) >= 5:
            _, job_id, state, elapsed, name = fields[:5]
            if not INFERENCE_SERVER_RE.search(name):
                continue
            jobs.append(
                Job(
                    cluster=cluster,
                    job_id=job_id,
                    state=state.upper(),
                    elapsed_seconds=parse_elapsed(elapsed),
                    reason="recent terminal state",
                    name=name,
                    command="",
                    model=extract_model(name, ""),
                )
            )
    return jobs


async def run_ssh(cluster: str, timeout: int) -> tuple[str, int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={min(timeout, 20)}",
        cluster,
        "bash",
        "-lc",
        shell_script(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return cluster, 124, "", f"timed out after {timeout}s"
    return cluster, proc.returncode, stdout.decode(), stderr.decode()


def build_statuses(
    jobs: Iterable[Job],
    pending_stuck_seconds: int,
    running_launch_seconds: int,
) -> list[ModelStatus]:
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job.cluster, job.model)].append(job)

    statuses = []
    for (cluster, model), model_jobs in sorted(grouped.items()):
        running = [job for job in model_jobs if job.state == "RUNNING"]
        pending = [job for job in model_jobs if job.state in {"PENDING", "CONFIGURING"}]
        dead = [
            job
            for job in model_jobs
            if any(token in job.state for token in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"))
        ]

        unknown_running = [
            job
            for job in running
            if job.model == "unknown-model" and job.elapsed_seconds >= running_launch_seconds
        ]
        old_pending = [
            job for job in pending if job.elapsed_seconds >= pending_stuck_seconds
        ]

        if unknown_running or old_pending:
            status = "STUCK"
            availability = False
            problem_jobs = unknown_running + old_pending
            detail = "; ".join(
                f"{job.job_id} {job.state.lower()} {job.elapsed_seconds}s {job.reason}".strip()
                for job in problem_jobs
            )
        elif running:
            status = "serving"
            availability = True
            detail = f"{len(running)} running job(s)"
        elif pending:
            status = "launching"
            availability = False
            detail = f"{len(pending)} pending/configuring job(s)"
        elif dead:
            status = "dead"
            availability = False
            detail = "; ".join(f"{job.job_id} {job.state.lower()}" for job in dead)
        else:
            status = "unknown"
            availability = False
            detail = "no classified jobs"

        statuses.append(
            ModelStatus(
                cluster=cluster,
                model=model,
                status=status,
                availability=availability,
                concurrency=len(running),
                jobs=model_jobs,
                detail=detail,
            )
        )
    return statuses


async def latest_failed_watchdog_models(db_path: str, limit: int) -> set[str]:
    latest = {}
    for row in await get_recent_metrics(limit=limit, db_path=db_path):
        latest.setdefault(row["model"], row)
    return {model for model, row in latest.items() if not row["success"]}


async def probe_watchdog_model(model: str, timeout: int) -> tuple[bool, str]:
    from main import ENDPOINTS, check_model, get_all_models_from_endpoints

    models_with_endpoints = await asyncio.to_thread(get_all_models_from_endpoints)
    word = "potato"
    prompt = (
        f"Give me ONLY a word. The word is {word}. Nothing else. "
        "No sentences, no explanations, no definitions. Just the word."
    )
    matches = [
        endpoint_id
        for model_id, endpoint_id in models_with_endpoints
        if model_id == model or model_id.endswith(f"/{model}")
    ]
    if not matches:
        return False, "model is not listed by configured OpenAI endpoints"
    for endpoint_id in matches:
        endpoint_name = ENDPOINTS[endpoint_id]["name"]
        try:
            ok, response, _ = await asyncio.wait_for(
                asyncio.to_thread(check_model, model, word, prompt, endpoint_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return False, f"{endpoint_name} probe timed out after {timeout}s"
        if ok:
            return True, f"{endpoint_name} responded"
        return False, f"{endpoint_name} probe failed: {response}"
    return False, "probe failed"


async def probe_failed_running_models(
    statuses: list[ModelStatus],
    db_path: str,
    recent_metric_limit: int,
    timeout: int,
) -> None:
    failed_models = await latest_failed_watchdog_models(db_path, recent_metric_limit)
    candidates = [
        status
        for status in statuses
        if status.status == "serving" and status.model in failed_models
    ]
    if not candidates:
        return
    results = await asyncio.gather(
        *(probe_watchdog_model(status.model, timeout) for status in candidates),
        return_exceptions=True,
    )
    for status, result in zip(candidates, results):
        if isinstance(result, Exception):
            ok = False
            detail = f"watchdog probe crashed: {result}"
        else:
            ok, detail = result
        if not ok:
            status.status = "STUCK"
            status.availability = False
            status.detail = f"running job exists, but watchdog probe says unavailable: {detail}"
        else:
            status.detail = f"{status.detail}; watchdog probe recovered: {detail}"


def format_status_line(status: ModelStatus) -> str:
    label = status.status.upper() if status.status == "STUCK" else status.status
    availability = "available" if status.availability else "unavailable"
    return (
        f"{status.cluster} | {status.model} | {label} | "
        f"concurrency={status.concurrency} | {availability} | "
        f"jobs={status.job_ids or '-'} | {status.detail}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check vLLM jobs on supercomputers.")
    parser.add_argument("--clusters", default=",".join(DEFAULT_CLUSTERS))
    parser.add_argument("--ssh-timeout", type=int, default=int(os.getenv("VLLM_SSH_TIMEOUT", "45")))
    parser.add_argument("--pending-stuck-minutes", type=int, default=int(os.getenv("VLLM_PENDING_STUCK_MINUTES", "30")))
    parser.add_argument("--running-launch-minutes", type=int, default=int(os.getenv("VLLM_RUNNING_LAUNCH_MINUTES", "10")))
    parser.add_argument("--probe-timeout", type=int, default=int(os.getenv("VLLM_PROBE_TIMEOUT", "45")))
    parser.add_argument("--recent-metric-limit", type=int, default=int(os.getenv("VLLM_RECENT_METRIC_LIMIT", "200")))
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--no-probe-red", action="store_true")
    args = parser.parse_args()

    clusters = [cluster.strip() for cluster in args.clusters.split(",") if cluster.strip()]
    started = time.monotonic()
    results = await asyncio.gather(
        *(run_ssh(cluster, args.ssh_timeout) for cluster in clusters),
        return_exceptions=True,
    )

    jobs = []
    errors = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
            continue
        cluster, returncode, stdout, stderr = result
        if returncode == 0:
            jobs.extend(parse_job_rows(cluster, stdout))
        else:
            errors.append(f"{cluster}: ssh failed ({returncode}) {stderr.strip()}")

    statuses = build_statuses(
        jobs,
        pending_stuck_seconds=args.pending_stuck_minutes * 60,
        running_launch_seconds=args.running_launch_minutes * 60,
    )
    if not args.no_probe_red:
        await probe_failed_running_models(
            statuses,
            db_path=args.db_path,
            recent_metric_limit=args.recent_metric_limit,
            timeout=args.probe_timeout,
        )

    if statuses:
        for status in statuses:
            print(format_status_line(status))
            await record_supercomputer_model_status(
                status.cluster,
                status.model,
                status.status,
                status.availability,
                status.concurrency,
                status.job_ids,
                status.detail,
                db_path=args.db_path,
            )
    else:
        print("No vLLM jobs found on configured supercomputers.")

    for error in errors:
        print(f"ERROR | {error}")
    elapsed = time.monotonic() - started
    print(f"Checked {len(clusters)} cluster(s) in {elapsed:.1f}s at {datetime.now(timezone.utc).isoformat()}")
    return 1 if any(status.status == "STUCK" for status in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
