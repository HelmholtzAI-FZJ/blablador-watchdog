#!/usr/bin/env python3
import asyncio
import aiosqlite
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_DB_PATH = os.path.expanduser("~/.blablador_watchdog/metrics.db")


async def get_all_metrics(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM model_metrics ORDER BY timestamp ASC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


def shorten_model_name(name: str, max_len: int = 25) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + "..."


async def main():
    metrics = await get_all_metrics()
    if not metrics:
        print("No data to plot")
        return

    for m in metrics:
        m["timestamp_dt"] = datetime.fromisoformat(m["timestamp"])
        m["short_model"] = shorten_model_name(m["model"])

    models = list(set(m["short_model"] for m in metrics))
    colors = plt.cm.tab20(range(len(models)))
    model_colors = dict(zip(models, colors))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Metrics Dashboard", fontsize=14, fontweight="bold")

    ax1 = axes[0, 0]
    for model in models:
        model_data = [m for m in metrics if m["short_model"] == model]
        ax1.plot(
            [m["timestamp_dt"] for m in model_data],
            [m["elapsed_seconds"] for m in model_data],
            marker="o",
            label=model,
            color=model_colors[model],
        )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Response Time (seconds)")
    ax1.set_title("Response Time over Time")
    ax1.legend(loc="upper left", fontsize=6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for model in models:
        model_data = [m for m in metrics if m["short_model"] == model]
        ax2.plot(
            [m["timestamp_dt"] for m in model_data],
            [m["tokens_per_second"] for m in model_data],
            marker="s",
            label=model,
            color=model_colors[model],
        )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Tokens/second")
    ax2.set_title("Throughput (Tokens/sec) over Time")
    ax2.legend(loc="upper left", fontsize=6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    success_counts = {}
    for model in models:
        model_data = [m for m in metrics if m["short_model"] == model]
        success = sum(1 for m in model_data if m["success"])
        total = len(model_data)
        success_counts[model] = success / total * 100 if total > 0 else 0
    bars = ax3.bar(success_counts.keys(), success_counts.values(), color=[model_colors[m] for m in success_counts])
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Success Rate (%)")
    ax3.set_title("Success Rate by Model")
    ax3.tick_params(axis="x", rotation=45, labelsize=7)
    ax3.set_ylim(0, 110)
    for bar, val in zip(bars, success_counts.values()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    ax4 = axes[1, 1]
    avg_times = {}
    for model in models:
        model_data = [m for m in metrics if m["short_model"] == model]
        times = [m["elapsed_seconds"] for m in model_data if m["elapsed_seconds"] is not None]
        avg_times[model] = sum(times) / len(times) if times else 0
    bars = ax4.barh(avg_times.keys(), avg_times.values(), color=[model_colors[m] for m in avg_times])
    ax4.set_xlabel("Avg Response Time (seconds)")
    ax4.set_title("Average Response Time by Model")
    ax4.tick_params(axis="y", labelsize=7)
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("metrics_plots.png", dpi=150, bbox_inches="tight")
    print("Saved plots to metrics_plots.png")


if __name__ == "__main__":
    asyncio.run(main())