#!/bin/bash
# 严格 OOF 看门狗：由旧项目3 cron 每10分钟委派，也可由 systemd timer 独立调用。
set -u
EXP="${SR_EXP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${SR_PYTHON:-python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT="$EXP/results/strict_oof_v1"
MARK="$OUT/markers"
LOG="$EXP/logs/strict_oof_v1"
BIG="$EXP/big/strict_oof_v1"
PUSH="$EXP/push.sh"
AUTO_RETRY=3
STALL_MIN=120
ANALYSIS_HARD_STALL_MIN=720
mkdir -p "$MARK" "$LOG" "$BIG/mirror/results" "$BIG/mirror/logs"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG/watchdog.log"; }
push() { bash "$PUSH" "$1" "$2" "${3:-blue}" >/dev/null 2>&1 || true; }

exec 9>"$LOG/.watchdog.lock"
flock -n 9 || exit 0

# 隔离镜像只增不减；断电/同步故障时可追溯，不覆盖旧 results。
cp -au "$OUT/." "$BIG/mirror/results/" 2>/dev/null || true
cp -au "$LOG/." "$BIG/mirror/logs/" 2>/dev/null || true

if [ -f "$OUT/READY.json" ] && [ -f "$MARK/finalize_strict_oof.done" ]; then
  # 存在不等于完成：每次都重算marker元数据、命令、交付全集哈希及ACTIVE。
  if (cd "$EXP" && "$PY" -m strict_oof.validate \
      --marker finalize_strict_oof >/dev/null 2>&1); then
    # 完成卡由整点播报器唯一发送；看门狗不另发第二张。
    exit 0
  fi
  say "READY/finalize marker deep validation failed; restart self-healing chain"
fi

own_step_pids() {
  local out="" p cmd
  for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ "$(readlink /proc/"$p"/cwd 2>/dev/null)" = "$EXP" ] || continue
    cmd=$(tr '\0' ' ' < /proc/"$p"/cmdline 2>/dev/null)
    case "$cmd" in *strict_oof/orchestrat*or.py*|*strict_oof.orchestrat*or*) continue ;; esac
    case "$cmd" in *strict_oof/*.py*|*strict_oof.*) out="$out $p" ;; esac
  done
  echo "$out"
}

# 持久失败：三轮看门狗级重试，超限后保留现场、每6小时告警。
for failed in "$MARK"/*.FAILED; do
  [ -e "$failed" ] || continue
  name=$(basename "$failed" .FAILED)
  counter="$MARK/$name.retry_count"
  count=$(tr -cd '0-9' < "$counter" 2>/dev/null); [ -z "$count" ] && count=0
  if [ "$count" -lt "$AUTO_RETRY" ]; then
    next=$((count + 1))
    echo "$next" > "$counter"
    mkdir -p "$OUT/failure_history"
    mv "$failed" "$OUT/failure_history/$name.$(date '+%Y%m%d_%H%M%S').FAILED"
    say "retry $name $next/$AUTO_RETRY"
    push "🔁项目3严格 OOF 自动重试" \
      "步骤 $name 第 $next/$AUTO_RETRY 次看门狗级重试；训练从 epoch 断点恢复。" orange
  else
    notified="$failed.notified"
    if [ ! -f "$notified" ] || \
       [ $(( ( $(date +%s) - $(stat -c %Y "$notified") ) / 3600 )) -ge 6 ]; then
      touch "$notified"
      push "⛔项目3严格 OOF 持久失败" \
        "步骤 $name 超过 $AUTO_RETRY 次自动重试，已失败关闭；查 $LOG/$name.log。" red
      say "persistent failure $name"
    fi
    exit 0
  fi
done

# 卡死恢复：训练/推理有断点，120min无输出可杀；分析只在12h硬阈值处理。
last=$(grep -E ' START [A-Za-z0-9_]+| DONE [A-Za-z0-9_]+' "$LOG/status.log" \
       2>/dev/null | tail -1)
event=$(awk '{print $3}' <<< "$last")
step=$(awk '{print $4}' <<< "$last")
step_log="$LOG/$step.log"
cool="$MARK/.last_stall_kill"
cool_age=99999
[ -f "$cool" ] && cool_age=$(( ( $(date +%s) - $(stat -c %Y "$cool") ) / 60 ))
if [ "$event" = "START" ] && [ -f "$step_log" ] && [ "$cool_age" -ge "$STALL_MIN" ]; then
  age=$(( ( $(date +%s) - $(stat -c %Y "$step_log") ) / 60 ))
  live=$(own_step_pids)
  if [ -n "$live" ]; then
    case "$step" in
      outer_train_*|outer_pred_*|terminal_train_*|terminal_pred_*|ablation_train_*|ablation_pred_*)
        if [ "$age" -gt "$STALL_MIN" ]; then
          kill -9 $live 2>/dev/null || true
          touch "$cool"
          say "stalled resumable step $step (${age}min), killed $live"
          push "🩺项目3严格 OOF 卡死自愈" \
            "$step 已 ${age}分钟无输出，已杀停；将从断点恢复。" orange
        fi
        ;;
      *)
        if [ "$age" -gt "$ANALYSIS_HARD_STALL_MIN" ]; then
          kill -9 $live 2>/dev/null || true
          touch "$cool"
          say "analysis hard stall $step (${age}min), killed $live"
          push "🩺项目3严格 OOF 分析活锁恢复" \
            "$step 超过12小时无输出，已杀停整步重试。" orange
        fi
        ;;
    esac
  fi
fi

alive=0
for pid in $(pgrep -f "strict_oof[./]orchestrat""or" 2>/dev/null); do
  [ "$(readlink /proc/"$pid"/cwd 2>/dev/null)" = "$EXP" ] && alive=1
done
if [ "$alive" = "0" ]; then
  orphan=$(own_step_pids)
  if [ -n "$orphan" ]; then
    say "orphan strict workers:$orphan; do not double-start"
    push "⚠️项目3严格 OOF 孤儿进程" \
      "编排器不在，但子进程$orphan仍存活；为避免双开未拉起新编排器。" red
  else
    bash "$EXP/strict_oof/run_all.sh" >> "$LOG/watchdog.log" 2>&1
    say "orchestrator absent -> restarted"
  fi
fi
