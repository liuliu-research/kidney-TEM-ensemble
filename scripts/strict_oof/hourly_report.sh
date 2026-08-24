#!/bin/bash
# 严格 OOF 补训整点飞书播报。由旧 hourly_report.sh 在 ENABLED 后委派。
set -u
EXP="${SR_EXP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${SR_PYTHON:-python}"
LOG="$EXP/logs/strict_oof_v1"
mkdir -p "$LOG"

# 防止 cron 重入造成同一整点重复卡片。
exec 9>"$LOG/.hourly_report.lock"
flock -n 9 || exit 0

cd "$EXP" || exit 0
exec "$PY" -m strict_oof.hourly_report --send
