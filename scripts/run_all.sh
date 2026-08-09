#!/bin/bash
# SR 大修重做实验总入口（远程 liuliu1 执行）
# 用法: bash run_all.sh
cd "$(dirname "$0")"
PY="${SR_PYTHON:-python}"
mkdir -p logs results
# 防重复启动：检查已有 orchestrator.py 进程的 cwd 是否为本目录
for pid in $(pgrep -f "orchestrator.py" 2>/dev/null); do
  if [ "$(readlink /proc/$pid/cwd 2>/dev/null)" = "$(pwd)" ]; then
    echo "orchestrator already running (pid=$pid)"; exit 0
  fi
done
setsid nohup $PY orchestrator.py > logs/orchestrator.out 2>&1 &
echo "orchestrator started pid=$!"
