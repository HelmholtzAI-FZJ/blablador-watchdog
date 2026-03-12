import asyncio
import os
import tempfile

import pytest

from metrics import init_db, record_metric, get_recent_metrics


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.asyncio
async def test_init_db(db_path):
    await init_db(db_path)
    assert os.path.exists(db_path)


@pytest.mark.asyncio
async def test_record_metric_success(db_path):
    await init_db(db_path)
    await record_metric(
        model="test-model",
        success=True,
        elapsed_seconds=1.5,
        tokens_used=100,
        tokens_per_second=66.67,
        db_path=db_path,
    )
    metrics = await get_recent_metrics(db_path=db_path)
    assert len(metrics) == 1
    assert metrics[0]["model"] == "test-model"
    assert metrics[0]["success"] == 1
    assert metrics[0]["elapsed_seconds"] == 1.5
    assert metrics[0]["tokens_used"] == 100
    assert metrics[0]["tokens_per_second"] == 66.67


@pytest.mark.asyncio
async def test_record_metric_failure(db_path):
    await init_db(db_path)
    await record_metric(
        model="test-model-fail",
        success=False,
        elapsed_seconds=0.5,
        tokens_used=None,
        tokens_per_second=None,
        error="Model not found",
        db_path=db_path,
    )
    metrics = await get_recent_metrics(db_path=db_path)
    assert len(metrics) == 1
    assert metrics[0]["model"] == "test-model-fail"
    assert metrics[0]["success"] == 0
    assert metrics[0]["error"] == "Model not found"


@pytest.mark.asyncio
async def test_get_recent_metrics_limit(db_path):
    await init_db(db_path)
    for i in range(5):
        await record_metric(
            model=f"model-{i}",
            success=True,
            elapsed_seconds=1.0,
            tokens_used=10,
            tokens_per_second=10.0,
            db_path=db_path,
        )
    metrics = await get_recent_metrics(limit=3, db_path=db_path)
    assert len(metrics) == 3