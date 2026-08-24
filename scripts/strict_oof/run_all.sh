#!/bin/bash
# 严格 OOF 唯一启动入口；写 ENABLED 后旧项目3 cron 看门狗会优先委派。
set -u
EXP="${SR_EXP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${SR_PYTHON:-python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GPU_LOCK="/run/user/$(id -u)/kidney-gpu-training.lock"
OUT="$EXP/results/strict_oof_v1"
LOG="$EXP/logs/strict_oof_v1"
mkdir -p "$OUT/markers" "$LOG" "$EXP/big/strict_oof_v1/locks"
tmp="$OUT/ENABLED.tmp"
printf '%s\n' "$(date --iso-8601=seconds)" > "$tmp" && mv "$tmp" "$OUT/ENABLED"

for pid in $(pgrep -f "strict_oof[./]orchestrat""or" 2>/dev/null); do
  [ "$(readlink /proc/"$pid"/cwd 2>/dev/null)" = "$EXP" ] || continue
  echo "strict OOF orchestrator already running (pid=$pid)"
  exit 0
done

cd "$EXP" || exit 1
setsid nohup flock -n "$GPU_LOCK" "$PY" -m strict_oof.orchestrator \
  >> "$LOG/orchestrator.out" 2>&1 &
echo "strict OOF orchestrator started pid=$!"
