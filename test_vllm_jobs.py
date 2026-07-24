from vllm_jobs import build_statuses, extract_model, parse_job_rows


def test_extract_model_prefers_served_model_name():
    command = "python -m vllm.entrypoints.openai.api_server --model /p/models/base --served-model-name alias-large"
    assert extract_model("vllm", command) == "alias-large"


def test_parse_running_vllm_job():
    output = "ACTIVE|123|RUNNING|00:12:03|None|vllm-alias-fast|python -m vllm.entrypoints.openai.api_server --model alias-fast\n"
    jobs = parse_job_rows("jureca", output)
    assert len(jobs) == 1
    assert jobs[0].cluster == "jureca"
    assert jobs[0].model == "alias-fast"
    assert jobs[0].elapsed_seconds == 723


def test_parse_job_when_vllm_is_only_visible_inside_launch_script():
    output = (
        "ACTIVE|6117|RUNNING|35-05:44:22|haicluster3|instinct|"
        "/p/FastChat/models/instinct/instinct.slurm "
        "srun vllm serve models/instinct --served-model-name alias-instinct --port 8014\n"
    )
    jobs = parse_job_rows("haicluster1", output)
    assert len(jobs) == 1
    assert jobs[0].model == "alias-instinct"


def test_parse_whisper_openai_endpoint():
    output = (
        "ACTIVE|6121|RUNNING|00:01:00|haicluster1|whisper-api|"
        "/p/FastChat/whisper.slurm python whisper_openai_api.py "
        "--model-path /p/FastChat/models/faster-whisper-large-v3 --port 8000\n"
    )
    jobs = parse_job_rows("haicluster1", output)
    assert len(jobs) == 1
    assert jobs[0].model == "faster-whisper-large-v3"


def test_running_job_is_serving_and_available():
    output = "ACTIVE|123|RUNNING|00:12:03|None|vllm-alias-fast|python -m vllm.entrypoints.openai.api_server --model alias-fast\n"
    statuses = build_statuses(parse_job_rows("booster", output), 1800, 600)
    assert len(statuses) == 1
    assert statuses[0].status == "serving"
    assert statuses[0].availability is True
    assert statuses[0].concurrency == 1


def test_pending_job_is_launching_before_threshold():
    output = "ACTIVE|124|PENDING|00:05:00|Resources|vllm-alias-code|python -m vllm.entrypoints.openai.api_server --model alias-code\n"
    statuses = build_statuses(parse_job_rows("jupiter", output), 1800, 600)
    assert statuses[0].status == "launching"
    assert statuses[0].availability is False


def test_old_pending_job_is_stuck():
    output = "ACTIVE|125|PENDING|00:45:00|Resources|vllm-alias-code|python -m vllm.entrypoints.openai.api_server --model alias-code\n"
    statuses = build_statuses(parse_job_rows("haicluster1", output), 1800, 600)
    assert statuses[0].status == "STUCK"
    assert statuses[0].availability is False


def test_recent_failed_vllm_job_is_dead():
    output = "RECENT|126|FAILED|00:03:00|vllm-alias-mis\n"
    statuses = build_statuses(parse_job_rows("jureca", output), 1800, 600)
    assert statuses[0].model == "alias-mis"
    assert statuses[0].status == "dead"
    assert statuses[0].availability is False
