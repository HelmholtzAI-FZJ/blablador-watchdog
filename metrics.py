import aiosqlite
import os
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.expanduser("~/.blablador_watchdog/metrics.db")


async def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                success INTEGER NOT NULL,
                elapsed_seconds REAL,
                tokens_used INTEGER,
                tokens_per_second REAL,
                error TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_timestamp
            ON model_metrics(model, timestamp)
            """
        )
        await db.commit()


async def record_metric(
    model: str,
    success: bool,
    elapsed_seconds: float | None,
    tokens_used: int | None,
    tokens_per_second: float | None,
    error: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                success INTEGER NOT NULL,
                elapsed_seconds REAL,
                tokens_used INTEGER,
                tokens_per_second REAL,
                error TEXT
            )
            """
        )
        await db.commit()
        await db.execute(
            """
            INSERT INTO model_metrics (
                timestamp, model, success, elapsed_seconds,
                tokens_used, tokens_per_second, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                model,
                1 if success else 0,
                elapsed_seconds,
                tokens_used,
                tokens_per_second,
                error,
            ),
        )
        await db.commit()


async def get_recent_metrics(
    model: str | None = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    if not os.path.exists(db_path):
        return []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]