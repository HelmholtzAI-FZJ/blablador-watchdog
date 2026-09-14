#!/bin/bash
source .venv/bin/activate
# Default: interactive Textual dashboard.
# ./run.sh --quiet  (or -q) prints the plain-text report instead (used by cron).
python3 main.py "$@"
python3 vllm_jobs.py
python3 plot_metrics.py
./monitor-cluster.sh all
