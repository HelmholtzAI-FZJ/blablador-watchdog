#!/bin/bash
source .venv/bin/activate
python3 main.py
python3 vllm_jobs.py
python3 plot_metrics.py
./monitor-cluster.sh all
