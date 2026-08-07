#!/usr/bin/env bash
#
# Run the entropy-vs-time benchmark across the three default Qwen3 models on six
# RTX 4090s, packing them onto disjoint GPUs so nothing collides, then draw the
# combined figures from cache.
#
# The 32B model shards over 4 GPUs (TP=4) and the 14B over 2 (TP=2); together
# they fill all six, so they run in parallel first. The 8B model then runs alone
# on a freed GPU. Running the 8B on cuda:0 *while* the 32B also holds cuda:0
# would OOM, since each engine grabs ~90% of every card it uses -- hence phases.
#
# Per-run stdout/stderr goes to logs/<model>.log, because concurrent tqdm bars
# would otherwise interleave in the terminal. Follow one with, e.g.:
#     tail -f logs/qwen3-32b.log
#
# Usage:  bash run.sh          (or: chmod +x run.sh && ./run.sh)

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "[phase 1] qwen3-32b (cuda:0-3) + qwen3-14b (cuda:4-5) in parallel"
python entropy_vs_time.py --models qwen3-32b --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    > logs/qwen3-32b.log 2>&1 &
pid_32b=$!
python entropy_vs_time.py --models qwen3-14b --devices cuda:4 cuda:5 \
    > logs/qwen3-14b.log 2>&1 &
pid_14b=$!
wait "$pid_32b"
wait "$pid_14b"

echo "[phase 2] qwen3-8b (cuda:0)"
python entropy_vs_time.py --models qwen3-8b --devices cuda:0 > logs/qwen3-8b.log 2>&1

echo "[phase 3] draw combined figures from cache (all three models overlaid)"
python entropy_vs_time.py --plot-only

echo "done. figures in figures/, per-run logs in logs/"
