#!/usr/bin/env python3
"""Charts from the watchdog metrics DB.

Designed for ~80 live models: one row per model, sorted so problems sit on
top, instead of one coloured line per model.

metrics_plots.png
  * Fleet health: share of probed models passing, per run (one line).
  * Health grid: failure rate per model per day (white = healthy, red = failing,
    grey = not probed). Currently failing models first.
  * Latency: median of the last 7 days (dot), p10-p90 (bar), median of the
    7 days before (hollow ring) -> a slowdown is a dot right of its ring.

supercomputer_model_status.png
  * Status timeline per cluster:model (hourly), plus current running jobs.
"""
import asyncio
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DEFAULT_DB_PATH = os.path.expanduser("~/.blablador_watchdog/metrics.db")

HEALTH_DAYS = 14          # columns in the health grid
ACTIVE_DAYS = 3           # a model is "live" if probed this recently
LATENCY_DAYS = 7          # window for current vs previous latency
STATUS_DAYS = 7           # supercomputer timeline span

INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d0d7de"
NO_DATA = "#eef1f4"
GOOD = "#2da44e"
BAD = "#cf222e"
LATENCY = "#0969da"

FAIL_CMAP = LinearSegmentedColormap.from_list(
    "fail", ["#ffffff", "#ffd8b5", "#fb8f44", "#cf222e", "#82071e"]
)
FAIL_CMAP.set_bad(NO_DATA)

STATUS_ORDER = ["serving", "launching", "unknown", "STUCK", "dead"]
STATUS_COLORS = {
    "serving": GOOD,
    "launching": "#54aeff",
    "unknown": "#afb8c1",
    "STUCK": "#bf8700",
    "dead": BAD,
}


def _rows(db_path: str, query: str) -> list[dict]:
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(query)]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def get_all_metrics(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    return _rows(db_path, "SELECT * FROM model_metrics ORDER BY id ASC")


async def get_supercomputer_status(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    return _rows(db_path, "SELECT * FROM supercomputer_model_status ORDER BY id ASC")


def shorten_model_name(name: str, max_len: int = 34) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


# ---------------------------------------------------------------- model probes
def summarize_models(metrics: list[dict], now: datetime) -> dict:
    """Pure aggregation, kept separate from drawing so it can be tested."""
    days = [(now - timedelta(days=i)).date() for i in range(HEALTH_DAYS - 1, -1, -1)]
    day_index = {d: i for i, d in enumerate(days)}
    active_since = now - timedelta(days=ACTIVE_DAYS)
    cur_since = now - timedelta(days=LATENCY_DAYS)
    prev_since = now - timedelta(days=2 * LATENCY_DAYS)

    per_model = defaultdict(list)
    for m in metrics:
        m["dt"] = _ts(m["timestamp"])
        per_model[m["model"]].append(m)

    models = {}
    for name, rows in per_model.items():
        if rows[-1]["dt"] < active_since:
            continue  # retired model: the report's diff already covers it
        ok = np.zeros(HEALTH_DAYS)
        total = np.zeros(HEALTH_DAYS)
        cur, prev = [], []
        for r in rows:
            i = day_index.get(r["dt"].date())
            if i is not None:
                total[i] += 1
                ok[i] += bool(r["success"])
            if r["success"] and r["elapsed_seconds"]:
                if r["dt"] >= cur_since:
                    cur.append(r["elapsed_seconds"])
                elif r["dt"] >= prev_since:
                    prev.append(r["elapsed_seconds"])
        with np.errstate(invalid="ignore", divide="ignore"):
            fail = np.where(total > 0, 1 - ok / np.maximum(total, 1), np.nan)
        recent = [r for r in rows if r["dt"] >= cur_since]
        models[name] = {
            "fail": fail,
            "last_ok": bool(rows[-1]["success"]),
            "last_error": rows[-1].get("error") or "",
            "fail_7d": 1 - (sum(bool(r["success"]) for r in recent) / len(recent)) if recent else 0.0,
            "cur": np.percentile(cur, [10, 50, 90]) if cur else None,
            "prev_median": float(np.median(prev)) if prev else None,
        }

    # Broken now first, then flaky (by 7-day failure rate), then by name.
    order = sorted(models, key=lambda n: (models[n]["last_ok"], -models[n]["fail_7d"], n.lower()))

    # Fleet health per probe run: runs are minutes apart, bucket by minute.
    runs = defaultdict(lambda: [0, 0])
    since = now - timedelta(days=HEALTH_DAYS)
    for m in metrics:
        if m["dt"] >= since:
            key = m["dt"].replace(second=0, microsecond=0)
            runs[key][0] += bool(m["success"])
            runs[key][1] += 1
    run_times = sorted(t for t, (_, n) in runs.items() if n >= 5)
    fleet = [(t, 100 * runs[t][0] / runs[t][1], runs[t][1] - runs[t][0]) for t in run_times]

    return {"days": days, "models": models, "order": order, "fleet": fleet}


def plot_models(metrics: list[dict], out: str = "metrics_plots.png", now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    s = summarize_models(metrics, now)
    order, models, days = s["order"], s["models"], s["days"]
    if not order:
        print("No recently probed models to plot")
        return

    n = len(order)
    row_h = 0.19
    fig = plt.figure(figsize=(15, 2.6 + n * row_h + 1.2))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.2, n * row_h], width_ratios=[1.25, 1],
        hspace=0.12 if n > 30 else 0.25, wspace=0.03,
    )
    broken = sum(not models[m]["last_ok"] for m in order)
    fig.suptitle(
        f"Blablador model health — {n} live models, {broken} failing on the last probe"
        f"   ·   {now:%Y-%m-%d %H:%M} UTC",
        x=0.01, y=1 - 0.25 / fig.get_figheight(), ha="left", fontsize=14,
        fontweight="bold", color=INK,
    )
    fig.subplots_adjust(top=1 - 0.7 / fig.get_figheight())

    # Fleet health line
    ax0 = fig.add_subplot(gs[0, :])
    _style(ax0)
    if s["fleet"]:
        # Break the line where no run happened for >3h instead of drawing a
        # diagonal across the gap, and show every run as a dot.
        t, pct = [], []
        for i, (ti, pi, _) in enumerate(s["fleet"]):
            if t and ti - s["fleet"][i - 1][0] > timedelta(hours=3):
                t.append(ti - timedelta(minutes=1))
                pct.append(np.nan)
            t.append(ti)
            pct.append(pi)
        ax0.plot(t, pct, color=LATENCY, lw=1.2)
        ax0.plot(t, pct, "o", color=LATENCY, ms=3)
        bad = [(ti, pi) for ti, pi in zip(t, pct) if pi == pi and pi < 90]
        if bad:
            ax0.plot(*zip(*bad), "o", color=BAD, ms=4, zorder=3)
        pct = [p for p in pct if p == p]
        last_t, last_pct, last_fail = s["fleet"][-1]
        ax0.annotate(f"{last_pct:.0f}% passing\n({last_fail} failing)", (last_t, last_pct),
                     xytext=(6, 0), textcoords="offset points", va="center",
                     fontsize=9, color=INK)
        ax0.set_ylim(max(0, min(pct) - 5), 101)
    ax0.set_title("Share of models passing per probe run (last 14 days)", loc="left",
                  fontsize=10, color=INK)
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax0.grid(axis="y", color=GRID, lw=0.6)

    labels = []
    for m in order:
        mark = "✗ " if not models[m]["last_ok"] else ""
        labels.append(mark + shorten_model_name(m))

    # Health grid
    ax1 = fig.add_subplot(gs[1, 0])
    grid = np.ma.masked_invalid(np.vstack([models[m]["fail"] for m in order]))
    ax1.imshow(grid, aspect="auto", cmap=FAIL_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels, fontsize=7.5)
    for tick, m in zip(ax1.get_yticklabels(), order):
        tick.set_color(BAD if not models[m]["last_ok"] else INK)
        if not models[m]["last_ok"]:
            tick.set_fontweight("bold")
    step = 2 if len(days) > 10 else 1
    ax1.set_xticks(range(0, len(days), step))
    ax1.set_xticklabels([d.strftime("%b %d") for d in days][::step], fontsize=7.5, rotation=0)
    ax1.xaxis.tick_top()
    ax1.set_xticks(np.arange(-0.5, len(days)), minor=True)
    ax1.set_yticks(np.arange(-0.5, n), minor=True)
    ax1.grid(which="minor", color="white", lw=1)
    ax1.tick_params(which="minor", length=0)
    ax1.tick_params(colors=MUTED, length=0)
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.set_title("Daily failure rate (white = always passed · red = always failed · grey = not probed)",
                  loc="left", fontsize=9, color=INK, pad=18)

    # Latency
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    _style(ax2)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.set_xscale("log")
    for y, m in enumerate(order):
        info = models[m]
        if y % 2 == 0:
            ax2.axhspan(y - 0.5, y + 0.5, color="#f6f8fa", lw=0, zorder=0)
        if info["cur"] is None:
            ax2.text(0.02, y, "no successful probe in 7 days", transform=ax2.get_yaxis_transform(),
                     va="center", fontsize=7, color=MUTED)
            continue
        p10, med, p90 = info["cur"]
        ax2.plot([p10, p90], [y, y], color=LATENCY, alpha=0.35, lw=3, solid_capstyle="round", zorder=2)
        prev = info["prev_median"]
        if prev:
            ax2.plot([prev], [y], "o", mfc="white", mec=MUTED, ms=6, mew=1.2, zorder=3)
        ax2.plot([med], [y], "o", color=LATENCY, ms=5, zorder=4)
        label = f"{med:.1f}s" if med < 10 else f"{med:.0f}s"
        if prev and (med / prev >= 1.5 or prev / med >= 1.5):
            ratio = med / prev
            label += f"  {'▲' if ratio > 1 else '▼'}{ratio:.1f}×"
            color = BAD if ratio > 1 else GOOD
        else:
            color = MUTED
        ax2.annotate(label, (max(p90, med), y), xytext=(5, 0), textcoords="offset points",
                     va="center", fontsize=7, color=color)
    ax2.grid(axis="x", which="major", color=GRID, lw=0.6)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}s"))
    ax2.xaxis.tick_top()
    ax2.set_title("Response time, last 7 days (● median, bar p10–p90, ○ previous week's median)",
                  loc="left", fontsize=9, color=INK, pad=18)
    ax2.set_ylim(n - 0.5, -0.5)

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plots to {out}")


# ------------------------------------------------------- supercomputer status
def plot_supercomputers(rows: list[dict], out: str = "supercomputer_model_status.png",
                        now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    start = (now - timedelta(days=STATUS_DAYS)).replace(minute=0, second=0, microsecond=0)
    hours = int((now - start).total_seconds() // 3600) + 1

    series = defaultdict(lambda: [None] * hours)
    latest = {}
    for r in rows:
        dt = _ts(r["timestamp"])
        if dt < start:
            continue
        key = f"{r['cluster']}: {r['model']}"
        series[key][int((dt - start).total_seconds() // 3600)] = r["status"]
        latest[key] = r
    if not series:
        print("No supercomputer model status data to plot")
        return

    # Carry the last known status forward across hours without a sample.
    for vals in series.values():
        for i in range(1, hours):
            if vals[i] is None:
                vals[i] = vals[i - 1]

    rank = {s: i for i, s in enumerate(STATUS_ORDER)}
    keys = sorted(series, key=lambda k: (latest[k]["status"] == "serving", k.split(":")[0], k.lower()))
    statuses = STATUS_ORDER + sorted({v for vals in series.values() for v in vals
                                       if v and v not in rank})
    idx = {s: i for i, s in enumerate(statuses)}
    cmap = ListedColormap([STATUS_COLORS.get(s, "#8250df") for s in statuses])
    cmap.set_bad(NO_DATA)
    grid = np.ma.masked_invalid(np.array(
        [[idx[v] if v else np.nan for v in series[k]] for k in keys], dtype=float))

    n = len(keys)
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(15, 1.8 + n * 0.26),
                                  gridspec_kw={"width_ratios": [12, 1], "wspace": 0.02})
    not_serving = sum(latest[k]["status"] != "serving" for k in keys)
    fig.suptitle(f"Supercomputer vLLM jobs — {n} deployments, {not_serving} not serving now",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)

    x0, x1 = mdates.date2num(start), mdates.date2num(start + timedelta(hours=hours))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(statuses) - 0.5,
              interpolation="nearest", extent=(x0, x1, n - 0.5, -0.5))
    ax.set_yticks(range(n))
    ax.set_yticklabels([shorten_model_name(k, 44) for k in keys], fontsize=8)
    for tick, k in zip(ax.get_yticklabels(), keys):
        if latest[k]["status"] != "serving":
            tick.set_color(BAD)
            tick.set_fontweight("bold")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", axis="y", color="white", lw=1.5)
    ax.grid(which="major", axis="x", color="white", lw=0.8)
    ax.tick_params(which="both", length=0, colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(color=cmap(idx[s]), label=s) for s in statuses],
              loc="upper left", bbox_to_anchor=(0, -0.04 * 12 / max(n, 12)), ncol=len(statuses),
              frameon=False, fontsize=8)
    ax.set_title(f"Status per hour, last {STATUS_DAYS} days", loc="left", fontsize=10, color=INK)

    conc = [latest[k]["concurrency"] for k in keys]
    axc.barh(range(n), conc, color=LATENCY, alpha=0.7, height=0.6)
    for y, c in enumerate(conc):
        axc.text(c, y, f" {c}", va="center", fontsize=7.5, color=INK)
    axc.set_ylim(n - 0.5, -0.5)
    axc.set_xlim(0, max(conc + [1]) * 1.6)
    axc.axis("off")
    axc.set_title("jobs now", fontsize=9, color=INK)

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plots to {out}")


async def main():
    metrics = await get_all_metrics()
    if metrics:
        plot_models(metrics)
    else:
        print("No model metrics to plot")
    plot_supercomputers(await get_supercomputer_status())


if __name__ == "__main__":
    asyncio.run(main())
