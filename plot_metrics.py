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


async def get_supercomputer_status(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    if not os.path.exists(db_path):
        return []
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        try:
            async with db.execute(
                """
                SELECT * FROM supercomputer_model_status
                ORDER BY timestamp ASC
                """
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.OperationalError:
            return []


def shorten_model_name(name: str, max_len: int = 10) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + "..."


# Different marker shapes for different models (enough for 30 models)
MARKERS = ["o", "s", "^", "v", "D", "*", "P", "X", "h", "8", ">", "<", "d", "p", "H", "x", "1", "2", "3", "4", "+", "o", "s", "^", "v", "D", "*", "P", "X", "h"]


async def main():
    metrics = await get_all_metrics()
    if metrics:
        for m in metrics:
            m["timestamp_dt"] = datetime.fromisoformat(m["timestamp"])
            m["short_model"] = shorten_model_name(m["model"])

        # Get unique short model names for display
        models_short = list(set(m["short_model"] for m in metrics))
        
        # Create color and marker maps based on short model names (for display)
        num_models = len(models_short)
        colors = plt.cm.tab20(range(num_models))
        model_colors = dict(zip(models_short, colors))
        model_markers = dict(zip(models_short, MARKERS[:num_models]))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Metrics Dashboard", fontsize=14, fontweight="bold")

        ax1 = axes[0, 0]
        for model_short in models_short:
            model_data = [m for m in metrics if m["short_model"] == model_short]
            ax1.plot(
                [m["timestamp_dt"] for m in model_data],
                [m["elapsed_seconds"] for m in model_data],
                marker=model_markers[model_short],
                label=model_short,
                color=model_colors[model_short],
            )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Response Time (seconds)")
        ax1.set_title("Response Time over Time")
        ax1.legend(loc="upper left", fontsize=6)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        for model_short in models_short:
            model_data = [m for m in metrics if m["short_model"] == model_short]
            ax2.plot(
                [m["timestamp_dt"] for m in model_data],
                [m["tokens_per_second"] for m in model_data],
                marker=model_markers[model_short],
                label=model_short,
                color=model_colors[model_short],
            )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Tokens/second")
        ax2.set_title("Throughput (Tokens/sec) over Time")
        ax2.legend(loc="upper left", fontsize=6)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        success_counts = {}
        for model_short in models_short:
            model_data = [m for m in metrics if m["short_model"] == model_short]
            success = sum(1 for m in model_data if m["success"])
            total = len(model_data)
            success_counts[model_short] = success / total * 100 if total > 0 else 0
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
        for model_short in models_short:
            model_data = [m for m in metrics if m["short_model"] == model_short]
            times = [m["elapsed_seconds"] for m in model_data if m["elapsed_seconds"] is not None]
            avg_times[model_short] = sum(times) / len(times) if times else 0
        ax4.barh(avg_times.keys(), avg_times.values(), color=[model_colors[m] for m in avg_times])
        ax4.set_xlabel("Avg Response Time (seconds)")
        ax4.set_title("Average Response Time by Model")
        ax4.tick_params(axis="y", labelsize=7)
        ax4.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig("metrics_plots.png", dpi=150, bbox_inches="tight")
        print("Saved plots to metrics_plots.png")
    else:
        print("No model metrics to plot")

    supercomputer_status = await get_supercomputer_status()
    if not supercomputer_status:
        print("No supercomputer model status data to plot")
        return

    for row in supercomputer_status:
        row["timestamp_dt"] = datetime.fromisoformat(row["timestamp"])
        row["label"] = f"{row['cluster']}:{shorten_model_name(row['model'], 18)}"

    labels = sorted(set(row["label"] for row in supercomputer_status))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle("Supercomputer vLLM Availability and Concurrency", fontsize=14, fontweight="bold")

    for label in labels:
        rows = [row for row in supercomputer_status if row["label"] == label]
        ax1.step(
            [row["timestamp_dt"] for row in rows],
            [row["availability"] for row in rows],
            where="post",
            label=label,
            linewidth=1.8,
        )
        ax2.step(
            [row["timestamp_dt"] for row in rows],
            [row["concurrency"] for row in rows],
            where="post",
            label=label,
            linewidth=1.8,
        )

    stuck_rows = [row for row in supercomputer_status if row["status"] == "STUCK"]
    if stuck_rows:
        ax1.scatter(
            [row["timestamp_dt"] for row in stuck_rows],
            [row["availability"] for row in stuck_rows],
            color="red",
            marker="x",
            s=55,
            label="STUCK",
            zorder=5,
        )

    ax1.set_ylabel("Available")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["no", "yes"])
    ax1.set_title("Availability by Supercomputer Model")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Running Jobs")
    ax2.set_xlabel("Time")
    ax2.set_title("vLLM Job Concurrency by Supercomputer Model")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    handles, labels_for_legend = ax2.get_legend_handles_labels()
    if len(labels_for_legend) <= 20:
        ax2.legend(handles, labels_for_legend, loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig("supercomputer_model_status.png", dpi=150, bbox_inches="tight")
    print("Saved plots to supercomputer_model_status.png")


if __name__ == "__main__":
    asyncio.run(main())
